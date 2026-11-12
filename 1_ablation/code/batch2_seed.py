# -*- coding: utf-8 -*-
"""batch2_seed20261111.py — 第二批 30 条：新种子从未用条目抽取，v2 prompt（锚定+氛围）
生成三模块拆分，Surface 校验 + 修复 + 候选替换，写出工作簿与溯源表。"""
import importlib.util
import random
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(".")
SEED = 20261111

spec = importlib.util.spec_from_file_location("exp", BASE / "expand_to_30.py")
exp = importlib.util.module_from_spec(spec)
sys.modules["exp"] = exp
spec.loader.exec_module(exp)


def gen_one(text):
    ok, content = exp.call_llm(f"Input: {text}")
    if not ok:
        return None, "llm_error", "LLM 调用失败"
    parsed = exp.parse_three(content)
    v = exp.validate(parsed)
    for _ in range(2):
        if v["validation_status"] in exp.abl.VALIDATION_OK_STATES:
            return parsed, v["validation_status"], ""
        err = v["validation_error"]
        ok2, content2 = exp.call_llm(f"Input: {text}" + exp.REPAIR_SUFFIX.format(error=err))
        if not ok2:
            break
        parsed = exp.parse_three(content2)
        v = exp.validate(parsed)
    return parsed, v["validation_status"], v["validation_error"]


def main():
    df200 = pd.read_csv(BASE / "T2I_CollageAttack/Dataset/sampled_200_hate.csv")
    used = set()
    for wb_name in ["CollageAttack_commands_split_30samples.xlsx"]:
        used |= set(pd.read_excel(BASE / wb_name, sheet_name="command3_rows")["original_text"].astype(str).str.strip())
    pool = df200[~df200["text"].astype(str).str.strip().isin(used)].copy()
    pool["orig_index"] = pool.index
    pool = pool.reset_index(drop=True)
    print(f"未用池: {len(pool)} 条 | 种子: {SEED}")

    rng = random.Random(SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)

    need, got, tried, fails = 30, [], 0, []
    for idx in order:
        if len(got) >= need:
            break
        tried += 1
        row = pool.iloc[idx]
        text = str(row["text"]).strip()
        print(f"[候选{tried}] ({len(got) + 1}/{need}) {text[:60]}")
        parsed, status, err = gen_one(text)
        if parsed is None or status not in exp.abl.VALIDATION_OK_STATES:
            print(f"    弃用 [{status}]: {err}")
            fails.append((text[:40], err))
            continue
        print(f"    通过 [{status}]")
        got.append({"original_text": text, "command1": parsed["command1"],
                    "command2": parsed["command2"], "command3": parsed["command3"],
                    "source_index_in_200": int(pool.iloc[idx]["orig_index"])})
        time.sleep(2)

    print(f"\n成功 {len(got)}/{need}（尝试 {tried}，弃用 {len(fails)}）")
    if len(got) < need:
        raise SystemExit("候选不足，请检查失败原因后重跑")

    out = BASE / "CollageAttack_commands_split_30samples_seed20261111.xlsx"
    rows = [{k: g[k] for k in ["original_text", "command1", "command2", "command3"]} for g in got]
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="command3_rows", index=False)
        prov = pd.DataFrame([{"sample_id": i + 1, "batch": "seed2", "seed": SEED,
                              "original_text": g["original_text"],
                              "source_index_in_200": g["source_index_in_200"]}
                             for i, g in enumerate(got)])
        prov.to_excel(w, sheet_name="provenance", index=False)
    print(f"已写出: {out}")


if __name__ == "__main__":
    main()
