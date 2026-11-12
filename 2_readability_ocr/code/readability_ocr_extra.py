#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可读性实验 · 补充 OCR 引擎（与 macOS Vision 同口径对比）

引擎：
  qwen     Qwen-VL（DashScope OpenAI 兼容接口，qwen-vl-max）独立 VLM 转写
  rapidocr RapidOCR（PaddleOCR PP-OCRv4 模型，onnxruntime 本地推理）

输入：For-Hon5/readability_check/readability_ocr.csv（Vision 那次已固定
30 张图与顺序）、images/ 下的图片
输出：readability_ocr_<engine>.csv（逐词恢复，与 Vision 表同列结构）
用法：/tmp/zcode_venv/bin/python -I readability_ocr_extra.py --engine qwen
      /tmp/zcode_venv/bin/python -I readability_ocr_extra.py --engine rapidocr
"""
import argparse
import base64
import csv
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(".")
OUT = Path("readability_out")
SRC = OUT / "readability_ocr.csv"
IMG = OUT / "images"

QWEN_PROMPT = "Read all the text in the image."  # qwen-vl-ocr 官方推荐 prompt


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def words_of(original: str):
    return [t for t in re.findall(r"[A-Za-z]+", original) if len(t) >= 2]


def load_picked():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    return [(int(r["图片序号"]), r["图片文件"], r["原始语句"]) for r in rows]


def transcribe_qwen(path: Path, model: str = "qwen-vl-ocr") -> str:
    from openai import OpenAI
    load_dotenv(BASE / ".env")
    client = OpenAI(api_key=os.getenv("QWEN_API_KEY"),
                    base_url=os.getenv("QWEN_BASE_URL", ""),
                    timeout=120, max_retries=0)
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" and data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        mime = "image/png"
    b64 = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    last = ""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64}},
                    {"type": "text", "text": QWEN_PROMPT},
                ]}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            time.sleep(5 * (attempt + 1))
    return f"__ERROR__ {last[:120]}"


def transcribe_rapidocr(path: Path) -> str:
    from rapidocr_onnxruntime import RapidOCR
    global _RCR
    try:
        _RCR
    except NameError:
        _RCR = RapidOCR()
    result, _ = _RCR(str(path))
    if not result:
        return ""
    return "\n".join(item[1] for item in result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["qwen", "rapidocr"], required=True)
    args = ap.parse_args()

    picked = load_picked()
    rows = []
    for i, fname, original in picked:
        path = IMG / fname
        text = (transcribe_qwen if args.engine == "qwen"
                else transcribe_rapidocr)(path)
        ws = words_of(original)
        tnorm = norm(text)
        recovered = [w for w in ws if norm(w) and norm(w) in tnorm]
        rows.append({
            "图片序号": i, "图片文件": fname, "原始语句": original,
            "OCR全文": text.replace("\n", " | "),
            "词数": len(ws), "OCR逐词恢复": len(recovered),
            "恢复率": f"{len(recovered)}/{len(ws)}",
            "恢复的词": ",".join(recovered),
        })
        print(f"[{args.engine} {i}/30] {fname} -> {len(recovered)}/{len(ws)}",
              flush=True)

    out_csv = OUT / f"readability_ocr_{args.engine}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot = sum(r["词数"] for r in rows)
    ok = sum(r["OCR逐词恢复"] for r in rows)
    print(f"完成：{ok}/{tot} = {ok / tot:.1%} -> {out_csv}")


if __name__ == "__main__":
    main()
