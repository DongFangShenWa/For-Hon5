# -*- coding: utf-8 -*-
"""
Rebuttal 实验（review Hon5 (3) 类别泛化）：CollageAttack × AdvBench 30 条 × 2 模型

数据：advbench30/advbench_30_selected.csv（30 条，7 类，选自 TYPO-main2607/data/advbench.json，
      即 TYPO 基线所用 AdvBench harmful-strings 语料，与 HateXclaim 论文语料不同源）

阶段：
  1) decompose  deepseek-v4-pro 生成 COMMAND1/2/3
     - SYSTEM_PROMPT 与论文原版 prompt_generation.py（prompt_generation_v2）逐字一致：ast 提取，防漂移
     - thinking=enabled / reasoning_effort=low，与论文运行配置一致
     - 质量门：C1/C2 必须非 None；归一化后完整原句不得整句出现在 C2（出现则重试 1 次，仍出现保留并打标）
  2) generate   Doubao-Seedream-5.0-Lite（ark 官方）+ GPT-Image-2（中转 gpt-image-2-2026-04-21，
     1024x1024 / medium，与论文 attack_gptimage2.py 一致）
     - prompt = C1 + ". " + C2 + ". " + C3（C3 为 None/空不拼接，与论文一致）
  3) judge      gpt-4.1 @ 中转，评分 SYSTEM_PROMPT import ablation_generate_and_judge.py（与论文一致），
     temperature=0，单次判定不因 Harmless 重试（与论文一致）

断点续跑：--outdir 指向已有 run 目录；decompose 结果（advbench_coj_commands.csv）与
items.csv 中已达终态的行自动跳过，缺评审的补评。

输出 advbench_rebuttal_results/run_<时间戳>/
  - config.json / advbench_coj_commands.csv / items.csv / summary.md / images/{doubao,gptimage2}/

用法：
  /tmp/zcode_venv/bin/python advbench_collage_rebuttal.py --samples 2        # 冒烟
  /tmp/zcode_venv/bin/python advbench_collage_rebuttal.py                     # 全量 30
  /tmp/zcode_venv/bin/python advbench_collage_rebuttal.py --outdir <run目录>  # 续跑/补评
  /tmp/zcode_venv/bin/python advbench_collage_rebuttal.py --skip-decompose    # 只跑生成+评审
"""

import argparse
import ast
import base64
import importlib.util
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import os

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

DS_API_KEY = os.getenv("DS_API_KEY")
UAPI_KEY = os.getenv("UAPI_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")

# ---- 与论文一致的模型与通道 ----
DS_BASE_URL = os.getenv("DS_BASE_URL", "")
DS_MODEL = "deepseek-v4-pro"                 # 论文 prompt_generation.py 的分解模型
RELAY_BASE_URL = os.getenv("UAPI_BASE_URL", "")
DOUBAO_MODEL = "doubao-seedream-5-0-lite-260128"   # 论文 Doubao-Seedream-5.0-Lite
GPT_IMAGE_MODEL = "gpt-image-2-2026-04-21"   # 论文 GPT-Image-2 的日期版本
GPT_IMAGE_SIZE = "1024x1024"                 # 与论文 attack_gptimage2.py 一致
GPT_IMAGE_QUALITY = "medium"
JUDGE_MODEL = "gpt-4.1"                      # 评审走中转，与论文同口径

SELECTION_CSV = BASE_DIR / "advbench30" / "advbench_30_selected.csv"
OLD_PROMPT_GEN = Path("<path>")
NEW_PROMPT_GEN = BASE_DIR / "T2I_CollageAttack" / "prompt_generation.py"
ABLATION_SCRIPT = BASE_DIR / "ablation_generate_and_judge.py"

MAX_ATTEMPTS = 3
RETRY_SLEEP = 3
RATE_LIMIT_SLEEP = 30
DECOMPOSE_SLEEP = 1
GEN_WORKERS = 3
JUDGE_WORKERS = 3

_print_lock = threading.Lock()


def ts_print(*a, **k):
    with _print_lock:
        print(*a, **k, flush=True)


def is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ["429", "rate limit", "rate_limit", "too many requests"])


def is_safety_rejection(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "safety", "content_filter", "content filter", "sensitive", "policy",
        "moderation", "inspection", "violated", "抱歉", "无法生成",
    ])


# ================= 与论文逐字一致的 prompt / 解析器 =================

def load_attack_system_prompt() -> str:
    """ast 提取论文原版攻击 SYSTEM_PROMPT（旧模块顶层即实例化 OpenAI 客户端，不能 import）。"""
    tree = ast.parse(OLD_PROMPT_GEN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SYSTEM_PROMPT":
            return ast.literal_eval(node.value)
    raise RuntimeError(f"未在 {OLD_PROMPT_GEN} 中找到 SYSTEM_PROMPT 赋值")


def load_parse_commands():
    """parse_commands 与论文原实现一致（新脚本注明'保持原实现'，客户端懒加载可安全 import）。"""
    spec = importlib.util.spec_from_file_location("pg_mod", NEW_PROMPT_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_commands


def load_judge_system_prompt() -> str:
    spec = importlib.util.spec_from_file_location("ablation_mod", ABLATION_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SYSTEM_PROMPT


def parse_judge_json(text: str) -> dict:
    t = (text or "").strip()
    t = t.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(t)
    score = data.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    return {
        "rating": str(data.get("rating", "")).strip(),
        "score": score,
        "judge_category": str(data.get("category", "")).strip(),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ================= 工具 =================

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()


def content_words(s: str) -> list:
    return [w for w in norm_text(s).split() if len(w) > 2]


def combine_prompt(c1, c2, c3) -> str:
    """C1 + '. ' + C2 + '. ' + C3（C3 为 None/空不拼），与论文 attack 脚本一致。"""
    parts = [str(c1).strip(), str(c2).strip()]
    c3s = str(c3).strip() if c3 is not None else ""
    if c3s and c3s.lower() not in ("none", "nan", ""):
        parts.append(c3s)
    return ". ".join(parts)


def valid_cmd(c) -> bool:
    return bool(c) and str(c).strip().lower() not in ("none", "nan", "")


# ================= 阶段 1：分解 =================

def decompose_single(ds_client, system_prompt, parse_commands, text: str) -> tuple:
    """分解单条文本（含解析校验与完整原句质量门），返回 (c1, c2, c3) 或全 None。"""
    best = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = ds_client.chat.completions.create(
                model=DS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input: {text}"},
                ],
                extra_body={"thinking": {"type": "enabled"}},
                reasoning_effort="low",
            )
            content = resp.choices[0].message.content
            c1, c2, c3 = parse_commands(content)
            if not (valid_cmd(c1) and valid_cmd(c2)):
                raise ValueError(f"COMMAND1/2 解析无效")
            row = (c1, c2, c3 if valid_cmd(c3) else "")
            if norm_text(text) in norm_text(c2) and attempt < MAX_ATTEMPTS:
                continue  # 完整原句整句出现在 C2：重试
            return row
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit(exc):
                time.sleep(RATE_LIMIT_SLEEP)
            elif attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP)
    return ("", "", "")


def run_decompose(df: pd.DataFrame, coj_path: Path, limit: int) -> pd.DataFrame:
    from openai import OpenAI
    if not DS_API_KEY:
        sys.exit("未找到 DS_API_KEY，请在 .env 中配置。")
    client = OpenAI(api_key=DS_API_KEY, base_url=DS_BASE_URL, timeout=120, max_retries=0)
    system_prompt = load_attack_system_prompt()
    parse_commands = load_parse_commands()

    if coj_path.exists():
        prev = pd.read_csv(coj_path)
        done = {r["id"]: r for r in prev.to_dict("records") if r.get("status") == "ok"}
        rows = {r["id"]: r for r in prev.to_dict("records")}
    else:
        done, rows = {}, {}

    todo = [r for r in df.to_dict("records") if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    ts_print(f"[decompose] 待分解 {len(todo)} 条（已完成 {len(done)}）")

    for i, r in enumerate(todo, 1):
        text = r["text"]
        best = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.chat.completions.create(
                    model=DS_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Input: {text}"},
                    ],
                    extra_body={"thinking": {"type": "enabled"}},
                    reasoning_effort="low",
                )
                content = resp.choices[0].message.content
                c1, c2, c3 = parse_commands(content)
                if not (valid_cmd(c1) and valid_cmd(c2)):
                    raise ValueError(f"COMMAND1/2 解析无效: c1={str(c1)[:60]} c2={str(c2)[:60]}")
                row = {"id": r["id"], "category": r["category"], "original_text": text,
                       "command1": c1, "command2": c2,
                       "command3": "" if not valid_cmd(c3) else c3,
                       "full_phrase_present": int(norm_text(text) in norm_text(c2)),
                       "word_coverage": round(
                           len(set(content_words(text)) & set(content_words(c2)))
                           / max(1, len(set(content_words(text)))), 2),
                       "status": "ok", "error": ""}
                # 质量门：完整原句不得整句出现在 C2 —— 首次出现则重试一次
                if row["full_phrase_present"] and attempt < MAX_ATTEMPTS:
                    ts_print(f"[decompose {i}/{len(todo)}] {r['id']} C2 含完整原句，重试")
                    best = row
                    continue
                best = row
                break
            except Exception as exc:  # noqa: BLE001
                if best is None:
                    best = {"id": r["id"], "category": r["category"], "original_text": text,
                            "command1": "", "command2": "", "command3": "",
                            "full_phrase_present": "", "word_coverage": "",
                            "status": "failed", "error": str(exc)}
                if is_rate_limit(exc):
                    ts_print(f"[decompose] 限流，冷却 {RATE_LIMIT_SLEEP}s")
                    time.sleep(RATE_LIMIT_SLEEP)
                elif attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_SLEEP)
        rows[r["id"]] = best
        ts_print(f"[decompose {i}/{len(todo)}] {r['id']} -> {best['status']}"
                 f" | coverage={best.get('word_coverage')} full_phrase={best.get('full_phrase_present')}")
        pd.DataFrame(list(rows.values())).to_csv(coj_path, index=False, encoding="utf-8-sig")
        time.sleep(DECOMPOSE_SLEEP)

    out = pd.DataFrame(list(rows.values()))
    return out


# ================= 阶段 2/3：生成 + 评审 =================

def build_clients():
    from openai import OpenAI
    doubao = OpenAI(api_key=DOUBAO_API_KEY, base_url=os.getenv("DOUBAO_BASE_URL", ""),
                    timeout=300, max_retries=0)
    gpt = OpenAI(api_key=UAPI_KEY, base_url=RELAY_BASE_URL, timeout=300, max_retries=0)
    judge = OpenAI(api_key=UAPI_KEY, base_url=RELAY_BASE_URL, timeout=120, max_retries=0)
    return doubao, gpt, judge


def generate_one(client, model_key: str, prompt: str, img_dir: Path, fname: str,
                 safety_budget: int = 1) -> dict:
    """生成单条。safety_budget=同 prompt 最多被输入侧拒绝的次数（1=单次请求口径）。
    追踪 gen_attempts（总投递次数）与 safety_rejections（被安全拒绝次数）。"""
    if model_key == "doubao":
        kwargs = {"model": DOUBAO_MODEL}
    else:
        kwargs = {"model": GPT_IMAGE_MODEL, "size": GPT_IMAGE_SIZE, "quality": GPT_IMAGE_QUALITY}
    status, err, img_path = None, "", None
    safety_used = 0
    attempts = 0
    max_attempts = MAX_ATTEMPTS + max(0, safety_budget - 1)
    while attempts < max_attempts:
        attempts += 1
        try:
            resp = client.images.generate(prompt=prompt, n=1, **kwargs)
            if not resp.data:
                raise RuntimeError("API 未返回 image data")
            item = resp.data[0]
            url = getattr(item, "url", None)
            b64 = getattr(item, "b64_json", None)
            img_path = img_dir / fname
            if b64:
                img_path.write_bytes(base64.b64decode(b64))
            elif url:
                content = requests.get(url, timeout=60).content
                if not content:
                    raise RuntimeError("图片下载为空")
                img_path.write_bytes(content)
            else:
                raise RuntimeError("响应中既无 b64_json 也无 url")
            status = "generated"
            break
        except Exception as exc:  # noqa: BLE001
            if is_safety_rejection(exc):
                safety_used += 1
                status, err = "safety_rejected", str(exc)[:300]
                if safety_used >= safety_budget:
                    break
                time.sleep(RETRY_SLEEP)
            else:
                status, err = "error", str(exc)[:300]
                if is_rate_limit(exc):
                    ts_print(f"    [{fname}] 限流，冷却 {RATE_LIMIT_SLEEP}s")
                    time.sleep(RATE_LIMIT_SLEEP)
                elif attempts < max_attempts:
                    time.sleep(RETRY_SLEEP)
    return {"status": status, "error": err,
            "image_path": str(img_path) if status == "generated" else "",
            "gen_attempts": attempts, "safety_rejections": safety_used}


def judge_one(judge_client, judge_prompt: str, rec: dict) -> dict:
    img = Path(rec["image_path"])
    for attempt in range(1, 4):
        try:
            resp = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"},
                    }]},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            parsed = parse_judge_json(raw)
            if parsed["score"] is None:
                raise ValueError(f"无有效 score: {raw[:100]}")
            rec.update({"judge_status": "success", "judge_raw": raw, **parsed, "judge_error": ""})
            return rec
        except Exception as exc:  # noqa: BLE001
            rec.update({"judge_status": "error", "judge_error": str(exc)})
            if is_rate_limit(exc):
                time.sleep(RATE_LIMIT_SLEEP)
            elif attempt < 3:
                time.sleep(RETRY_SLEEP)
    return rec


# ================= 主流程 =================

def main(args):
    df = pd.read_csv(SELECTION_CSV)
    if args.samples:
        df = df.head(args.samples)
    n_total = len(df)
    for k in ("DS_API_KEY", "UAPI_KEY", "DOUBAO_API_KEY"):
        if not globals()[k]:
            sys.exit(f"未找到 {k}，请在 .env 中配置。")

    run_dir = Path(args.outdir) if args.outdir else (
        BASE_DIR / "advbench_rebuttal_results" / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    img_root = run_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    coj_path = run_dir / "advbench_coj_commands.csv"
    items_path = run_dir / "items.csv"
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps({
            "purpose": "review Hon5 (3): CollageAttack cross-category generalization on AdvBench",
            "selection": str(SELECTION_CSV), "n": n_total,
            "source_corpus": "TYPO-main2607/data/advbench.json (AdvBench harmful strings, same corpus as TYPO baseline)",
            "categories": df["category"].value_counts().to_dict(),
            "decompose_llm": f"{DS_MODEL} @ {DS_BASE_URL} (SYSTEM_PROMPT verbatim from paper prompt_generation.py)",
            "targets": {"doubao": DOUBAO_MODEL, "gptimage2": GPT_IMAGE_MODEL},
            "gpt_image_params": {"size": GPT_IMAGE_SIZE, "quality": GPT_IMAGE_QUALITY},
            "prompt_rule": "C1 + '. ' + C2 + '. ' + C3 (C3 skipped if None) — same as paper attack scripts",
            "judge": f"{JUDGE_MODEL} @ relay, SYSTEM_PROMPT identical to paper/ablation, temperature=0",
        }, ensure_ascii=False, indent=2))

    # ---- 阶段 1：分解 ----
    if not args.skip_decompose:
        coj = run_decompose(df, coj_path, args.samples or 0)
    elif coj_path.exists():
        coj = pd.read_csv(coj_path)
    else:
        sys.exit("--skip-decompose 但找不到已存在的 advbench_coj_commands.csv")

    ok = coj[coj["status"] == "ok"].copy()
    ok["combined_prompt"] = [combine_prompt(a, b, c) for a, b, c
                             in zip(ok["command1"], ok["command2"], ok["command3"])]
    n_decomp_fail = n_total - len(ok)
    ts_print(f"[decompose] 完成：ok {len(ok)}/{n_total}" + (f"，失败 {n_decomp_fail}" if n_decomp_fail else ""))

    # ---- 断点恢复 items ----
    results = {}  # (model, id) -> record
    first_shot = {}  # (model, id) -> 首次运行终态（重投模式下保留单次口径记录）
    if items_path.exists():
        prev = pd.read_csv(items_path)
        for r in prev.to_dict("records"):
            terminal = r.get("status") in ("generated", "safety_rejected") or \
                (r.get("status") == "decompose_failed")
            if terminal:
                results[(r["model"], r["adv_id"])] = r
        ts_print(f"[resume] items.csv 已有 {len(results)} 条终态")

    models = [m for m in ("doubao", "gptimage2") if m in args.models]
    # 重投模式：把选定模型的 safety_rejected 行复活，同一 prompt 最多再投 args.retry_rejected 次
    if args.retry_rejected:
        for key, rec in list(results.items()):
            if rec["model"] in models and rec.get("status") == "safety_rejected":
                first_shot[key] = "safety_rejected"
                del results[key]
        ts_print(f"[retry] 复活 {len(first_shot)} 条 safety_rejected（同 prompt 重投上限 "
                 f"{args.retry_rejected} 次；单次口径结果保留在 first_shot_status 列）")
    doubao_c, gpt_c, judge_c = build_clients()
    clients = {"doubao": doubao_c, "gptimage2": gpt_c}

    def save_items():
        cols = ["model", "adv_id", "category", "sentence", "combined_prompt", "status", "error",
                "first_shot_status", "gen_attempts", "safety_rejections",
                "recompose_rounds", "recompose_reason",
                "image_path", "judge_status", "rating", "score", "judge_category",
                "rationale", "judge_raw", "judge_error"]
        out = pd.DataFrame(list(results.values()))
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        out[cols].to_csv(items_path, index=False)

    # ---- 阶段 1.5：被拦/拒评条目重新分解重做（--recompose-rejected K）----
    if args.recompose_rejected:
        from openai import OpenAI as _OpenAI
        ds_client = _OpenAI(api_key=DS_API_KEY, base_url=DS_BASE_URL, timeout=120, max_retries=0)
        _sys_prompt = load_attack_system_prompt()
        _parse = load_parse_commands()
        recompose_path = run_dir / "advbench_recompose_log.csv"
        targets = [rec for rec in results.values()
                   if rec["model"] in models and (
                       rec["status"] == "safety_rejected"
                       or (rec["status"] == "generated" and str(rec.get("judge_status")) == "judge_refused"))]
        ts_print(f"[recompose] 待重做 {len(targets)} 条（每条 ≤{args.recompose_rejected} 轮新分解，每轮单次投递）")
        _lock = threading.Lock()
        _log_rows = []

        def _append_log(row: dict):
            with _lock:
                _log_rows.append(row)
                pd.DataFrame(_log_rows).to_csv(recompose_path, index=False)

        def recompose_one(rec: dict) -> dict:
            model = rec["model"]
            orig_status = rec["status"]
            img_dir = img_root / model
            img_dir.mkdir(exist_ok=True)
            r0 = img_dir / f"{rec['adv_id']}_r0.png"
            if orig_status == "generated" and rec.get("image_path"):
                old_img = Path(rec["image_path"])
                if old_img.exists() and not r0.exists():
                    old_img.rename(r0)  # 旧图（评审拒评）留档
            for rnd in range(1, args.recompose_rejected + 1):
                c1, c2, c3 = decompose_single(ds_client, _sys_prompt, _parse, rec["sentence"])
                if not valid_cmd(c1):
                    _append_log({"model": model, "adv_id": rec["adv_id"], "round": rnd,
                                 "outcome": "decompose_failed", "combined_prompt": ""})
                    continue
                prompt = combine_prompt(c1, c2, c3)
                res = generate_one(clients[model], model, prompt, img_dir,
                                   f"{rec['adv_id']}.png", safety_budget=1)
                _append_log({"model": model, "adv_id": rec["adv_id"], "round": rnd,
                             "outcome": res["status"], "combined_prompt": prompt})
                if res["status"] == "generated":
                    rec.update({"combined_prompt": prompt, "status": "generated", "error": "",
                                "image_path": res["image_path"],
                                "recompose_rounds": rnd, "recompose_reason": orig_status,
                                "judge_status": "", "rating": "", "score": "",
                                "judge_category": "", "rationale": "", "judge_raw": "",
                                "judge_error": ""})
                    return rec
                time.sleep(RETRY_SLEEP)
            # 各轮均未生成成功：保持原终态（评审拒评条目指回留档旧图）
            rec["recompose_rounds"] = args.recompose_rejected
            rec["recompose_reason"] = orig_status
            if orig_status == "generated" and r0.exists():
                rec["image_path"] = str(r0)
                rec["judge_status"] = "judge_refused"
            return rec

        if targets:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [pool.submit(recompose_one, dict(r)) for r in targets]
                for i, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    results[(rec["model"], rec["adv_id"])] = rec
                    ts_print(f"[recompose {i}/{len(targets)}] {rec['model']}/{rec['adv_id']} -> "
                             f"{rec['status']}"
                             + (f"（第 {rec.get('recompose_rounds')} 轮成功）"
                                if rec["status"] == "generated" and rec.get("recompose_reason") else ""))
            save_items()

    # ---- 阶段 2：生成 ----
    def base_rec(model: str, row: dict) -> dict:
        return {"model": model, "adv_id": row["id"], "category": row["category"],
                "sentence": row["original_text"], "combined_prompt": row["combined_prompt"],
                "first_shot_status": first_shot.get((model, row["id"]), "")}

    jobs = []
    for model in models:
        for row in ok.to_dict("records"):
            if (model, row["id"]) not in results:
                jobs.append((model, row))
    if n_decomp_fail and not args.skip_decompose:
        for model in models:
            for row in df.to_dict("records"):
                if row["id"] not in set(ok["id"]) and (model, row["id"]) not in results:
                    rec = base_rec(model, {"id": row["id"], "category": row["category"],
                                           "original_text": row["text"], "combined_prompt": ""})
                    rec.update({"status": "decompose_failed", "error": "LLM 分解失败",
                                "image_path": "", "judge_status": "", "rating": "", "score": "",
                                "judge_category": "", "rationale": "", "judge_raw": "", "judge_error": ""})
                    results[(model, row["id"])] = rec

    if jobs:
        ts_print(f"[gen] 待生成 {len(jobs)} 条（{'/'.join(models)}）")
        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as pool:
            futs = {}
            for model, row in jobs:
                img_dir = img_root / model
                img_dir.mkdir(exist_ok=True)
                fname = f"{row['id']}.png"
                budget = args.retry_rejected if (args.retry_rejected and first_shot.get((model, row["id"]))) else 1
                futs[pool.submit(generate_one, clients[model], model,
                                 row["combined_prompt"], img_dir, fname,
                                 safety_budget=budget)] = (model, row)
            for i, fut in enumerate(as_completed(futs), 1):
                model, row = futs[fut]
                rec = base_rec(model, row)
                rec.update(fut.result())
                for c in ("judge_status", "rating", "score", "judge_category",
                          "rationale", "judge_raw", "judge_error"):
                    rec[c] = ""
                results[(model, row["id"])] = rec
                ts_print(f"[gen {i}/{len(jobs)}] {model}/{row['id']} -> {rec['status']}"
                         f" (attempts={rec.get('gen_attempts')}, safety_rej={rec.get('safety_rejections')})"
                         + (f" | {rec['error'][:80]}" if rec["error"] else ""))
                save_items()

    # ---- 阶段 3：评审 ----
    to_judge = [r for r in results.values()
                if r["status"] == "generated" and not r.get("rating")]
    if to_judge:
        ts_print(f"[judge] 待评 {len(to_judge)} 张")
        judge_prompt = load_judge_system_prompt()
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
            futs = [pool.submit(judge_one, judge_c, judge_prompt, dict(r)) for r in to_judge]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                results[(rec["model"], rec["adv_id"])] = rec
                ts_print(f"[judge {i}/{len(to_judge)}] {rec['model']}/{rec['adv_id']} -> "
                         f"{rec.get('rating')} {rec.get('score')}")
                save_items()

    # ---- 汇总 ----
    save_items()
    all_recs = pd.DataFrame(list(results.values()))
    recomposed = int((pd.to_numeric(all_recs.get("recompose_rounds"), errors="coerce").fillna(0) > 0).sum()) \
        if len(all_recs) else 0
    lines = [f"# CollageAttack × AdvBench 30 条（review Hon5 (3) 类别泛化）",
             "",
             f"- 语料：AdvBench harmful strings（TYPO 基线同源）× {n_total} 条 / 7 类",
             f"- 分解：{DS_MODEL}（论文同款攻击模板）；评审：{JUDGE_MODEL}（论文同款评分 prompt）",
             f"- ASR 口径：rating==Harmful / {n_total}（分母=全部尝试，输入拦截计入失败，与论文 Table 1 一致）"]
    if recomposed:
        lines.append(f"- 输入被拦/评审拒评条目经 ≤{args.recompose_rejected} 轮重新分解后重投（逐轮明细：advbench_recompose_log.csv）")
    lines.append("")
    summary_rows = []
    for model in models:
        sub = all_recs[all_recs["model"] == model] if len(all_recs) else pd.DataFrame()
        n = len(sub)
        gen = int((sub["status"] == "generated").sum()) if n else 0
        rej = int((sub["status"] == "safety_rejected").sum()) if n else 0
        err = int((sub["status"] == "error").sum()) if n else 0
        df_fail = int((sub["status"] == "decompose_failed").sum()) if n else 0
        judged = sub[(sub["status"] == "generated") & (sub["judge_status"] == "success")] if n else pd.DataFrame()
        harmful = judged[judged["rating"] == "Harmful"] if len(judged) else pd.DataFrame()
        asr = len(harmful) / n_total
        avg = judged["score"].astype(float).mean() if len(judged) else float("nan")
        lines += [f"- 评审 Harmful: {len(harmful)}/{n_total} → **ASR = {asr:.1%}**"]
        name = DOUBAO_MODEL if model == "doubao" else GPT_IMAGE_MODEL
        lines += [f"## {model}（{name}）",
                  "",
                  f"- 生成成功 {gen}/{n_total} | 输入拦截 {rej} | 其他错误 {err} | 分解失败 {df_fail}",
                  f"- 生成图平均 Harmful Score: {avg:.2f}",
                  "",
                  "| 类别 | n | 拦截 | Harmful | ASR | 平均分 |",
                  "|---|---|---|---|---|---|"]
        for cat in ["self_harm", "threat_violence", "weapons_explosives", "fraud_scam",
                    "sexual_text", "harass_hate", "drugs"]:
            cs = sub[sub["category"] == cat] if n else pd.DataFrame()
            cn = len(cs)
            if not cn:
                continue
            crej = int((cs["status"] == "safety_rejected").sum())
            cj = cs[(cs["status"] == "generated") & (cs["judge_status"] == "success")]
            ch = int((cj["rating"] == "Harmful").sum()) if len(cj) else 0
            cavg = cj["score"].astype(float).mean() if len(cj) else None
            avg_str = f"{cavg:.2f}" if cavg is not None else "—"
            lines.append(f"| {cat} | {cn} | {crej} | {ch} | {ch / cn:.0%} | {avg_str} |")
            summary_rows.append({"model": model, "category": cat, "n": cn, "rejected": crej,
                                 "harmful": ch, "asr": ch / cn, "avg_score": cavg})
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n结果目录: {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=0, help="只跑前 N 条（冒烟用；0=全量）")
    p.add_argument("--models", default="doubao,gptimage2", help="逗号分隔：doubao,gptimage2")
    p.add_argument("--outdir", default="", help="断点续跑：指定已有 run 目录")
    p.add_argument("--skip-decompose", action="store_true", help="复用已有 advbench_coj_commands.csv")
    p.add_argument("--retry-rejected", type=int, default=0,
                   help="对 safety_rejected 条目用同一 prompt 最多重投 N 次（研究输入过滤非确定性；单次口径保留在 first_shot_status 列）")
    p.add_argument("--recompose-rejected", type=int, default=0,
                   help="对仍被拦/评审拒评的条目重新分解重做：每条 ≤N 轮新 C1/C2/C3、每轮单次投递；逐轮明细写 advbench_recompose_log.csv")
    main(p.parse_args())
