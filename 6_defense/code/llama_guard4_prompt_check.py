#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-Guard-4-12B 前置文本护栏实验(YUQu W4): 检测五种方法提交给 T2I API 的完整 prompt。
输入: guard_prompt_pack.csv (method, idx, original_text, prompt)
输出: guard_results.csv (…, guard_raw, guard_label, guard_categories)
运行: CUDA_VISIBLE_DEVICES=0,1 python llama_guard4_prompt_check.py \
        --model /hub/huggingface/models/meta/Llama-Guard-4-12B \
        --input guard_prompt_pack.csv --output guard_results.csv
协议: 单次判定, greedy(do_sample=False), 与论文输出侧 judge 的三次重试规则不同(输入侧拦截只需一次)。
"""
import argparse, json, re, sys, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor

# 服务器 checkpoint 是 Meta 初版格式(config.model_type=llama4), 环境里也无 llama_guard4 专用模块;
# Llama Guard 4 = Llama4ForConditionalGeneration 权重 + 仓库自带 guard chat_template, 行为一致。
try:
    from transformers import LlamaGuard4ForConditionalGeneration as GuardModelClass
except ImportError:
    from transformers import Llama4ForConditionalGeneration as GuardModelClass

CAT_RE = re.compile(r"\bS\d+\b")


def parse_guard(text: str):
    t = (text or "").strip()
    label = "unsafe" if t.lower().startswith("unsafe") else ("safe" if t.lower().startswith("safe") else "parse_error")
    return label, sorted(set(CAT_RE.findall(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/hub/huggingface/models/meta/Llama-Guard-4-12B")
    ap.add_argument("--input", default="guard_prompt_pack.csv")
    ap.add_argument("--output", default="guard_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="每组 method 只取前 N 条(冒烟用), 0=全量")
    ap.add_argument("--methods", default="", help="逗号分隔过滤, 空=全部")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.methods:
        df = df[df["method"].isin(args.methods.split(","))].copy()
    if args.limit:
        df = df.groupby("method", group_keys=False).head(args.limit).copy()

    done = set()
    if Path(args.output).exists():  # 断点续跑
        prev = pd.read_csv(args.output)
        done = set(zip(prev["method"], prev["idx"]))
        print(f"[resume] 已完成 {len(done)} 条, 跳过", flush=True)

    print(f"[load] {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = GuardModelClass.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()
    print(f"[load] ok, device_map={getattr(model, 'hf_device_map', '?')}", flush=True)

    fout = Path(args.output)
    t0 = time.time()
    for n, (_, r) in enumerate(df.iterrows(), 1):
        key = (r["method"], int(r["idx"]))
        if key in done:
            continue
        conversation = [{"role": "user", "content": [{"type": "text", "text": str(r["prompt"])}]}]
        raw, label, cats, err = "", "", "", ""
        try:
            try:  # 标准路径: 直接返回 BatchFeature; 部分版本纯文本时返回 str, 走回退
                inputs = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True
                )
                if isinstance(inputs, str):
                    raise TypeError("apply_chat_template returned str")
            except Exception:
                rendered = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(text=[rendered], return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            else:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[-1]:]
            raw = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
            label, cats = parse_guard(raw)
            cats = ",".join(cats)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            label = "error"
        pd.DataFrame([{**r.to_dict(), "guard_raw": raw, "guard_label": label,
                       "guard_categories": cats, "error": err}]).to_csv(
            fout, mode="a", header=not fout.exists(), index=False)
        if n % 20 == 0 or label in ("parse_error", "error"):
            print(f"[{n}/{len(df)}] {r['method']}#{r['idx']} -> {label} {cats} {err[:80]}", flush=True)

    res = pd.read_csv(fout)
    ok = res[res["guard_label"].isin(["safe", "unsafe"])]
    print(f"\n[done] {len(res)} rows, {time.time()-t0:.0f}s, errors={len(res)-len(ok)}")
    if len(ok):
        summ = ok.pivot_table(index="method", values="guard_label",
                              aggfunc=lambda s: f"{(s=='unsafe').sum()}/{len(s)} ({(s=='unsafe').mean()*100:.1f}%)")
        print(summ)


if __name__ == "__main__":
    main()
