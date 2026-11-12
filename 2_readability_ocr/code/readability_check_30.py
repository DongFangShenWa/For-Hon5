#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工可读性实验 · 后处理全流程（回应审稿人 Q3）

前置：gptimage2_asr_single_pass_results/CollageAttack.csv（165 张 GPT-Image-2
生成图的 gpt-4.1 单次评审，含 rating + rationale）

本脚本产出（For-Hon5/readability_check/）：
  1. images/                 抽中的 30 张 Harmful 图（复制）
  2. readability_annotation.csv   人工标注表（逐词一行，打分列留空）
  3. 标注说明.md                  打分标准与操作说明
  4. readability_ocr.csv          macOS Vision OCR 逐词恢复结果（自动下界）
  5. readability_rationale.csv    裁判 rationale 逐词转写统计（辅助）
  6. summary.md                   三口径汇总（人工列留占位）

计分粒度：原句逐词（词=去标点后 ≥2 字母的 token；碎片可能跨牌拼接，
按"该词的字母整体是否可读"打分）。
用法：/tmp/zcode_venv/bin/python -I readability_check_30.py
"""
import csv
import random
import re
import shutil
import subprocess
from pathlib import Path

BASE = Path(".")
OUT = Path("readability_out")
JUDGE_CSV = BASE / "gptimage2_asr_single_pass_results" / "CollageAttack.csv"
IMG_DIR = BASE / "GPTImage2-ASR" / "CollageAttack"
COJ_CSV = BASE / "T2I_CollageAttack" / "coj_commands.csv"
VISION_JS = BASE / "vision_ocr.js"

SEED = 20261111
N_SAMPLE = 30


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def words_of(original: str):
    toks = re.findall(r"[A-Za-z]+", original)
    return [t for t in toks if len(t) >= 2]


def main():
    # 1) 读评审结果，取 Harmful
    judged = [r for r in csv.DictReader(open(JUDGE_CSV, encoding="utf-8-sig"))
              if r["status"] == "success" and r["rating"] == "Harmful"]
    print(f"Harmful 图 {len(judged)} 张")

    # 2) 固定种子抽 30（按文件名排序保证可复现）
    judged.sort(key=lambda r: r["file"])
    random.Random(SEED).shuffle(judged)
    picked = judged[:N_SAMPLE]

    # 原句对照：文件名数字 = coj_commands.csv 的 1-based 行号
    coj = list(csv.DictReader(open(COJ_CSV, encoding="utf-8-sig")))
    meta = {}
    for r in picked:
        idx = int(re.search(r"(\d+)", r["file"]).group(1))
        meta[r["file"]] = {
            "idx": idx,
            "original": coj[idx - 1]["original_text"],
            "rationale": r["rationale"],
            "score": r["score"],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    img_out = OUT / "images"
    img_out.mkdir(exist_ok=True)

    # 3) 复制图片
    for r in picked:
        shutil.copy2(IMG_DIR / r["file"], img_out / r["file"])

    # 4) 人工标注表（逐词一行）
    with open(OUT / "readability_annotation.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图片序号", "图片文件", "原始语句", "词序", "目标词",
                    "打分(0/1/2)", "备注"])
        n_words = 0
        for i, r in enumerate(picked, 1):
            m = meta[r["file"]]
            for j, word in enumerate(words_of(m["original"]), 1):
                w.writerow([i, r["file"], m["original"], j, word, "", ""])
                n_words += 1
    print(f"标注表：{N_SAMPLE} 张图 / {n_words} 个词")

    # 5) OCR（Vision）逐词恢复
    ocr_rows, ocr_word_stat = [], {}
    for i, r in enumerate(picked, 1):
        m = meta[r["file"]]
        p = subprocess.run(
            ["osascript", "-l", "JavaScript", str(VISION_JS),
             str((img_out / r["file"]).resolve())],
            capture_output=True, text=True, timeout=120)
        ocr_text = p.stdout.strip()
        ocr_norm = norm(ocr_text)
        ws = words_of(m["original"])
        recovered = [wd for wd in ws if norm(wd) and norm(wd) in ocr_norm]
        ocr_rows.append({
            "图片序号": i, "图片文件": r["file"], "原始语句": m["original"],
            "OCR全文": ocr_text.replace("\n", " | "),
            "词数": len(ws), "OCR逐词恢复": len(recovered),
            "恢复率": f"{len(recovered)}/{len(ws)}",
            "恢复的词": ",".join(recovered),
        })
        print(f"[OCR {i}/{N_SAMPLE}] {r['file']} -> {len(recovered)}/{len(ws)}")
    with open(OUT / "readability_ocr.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ocr_rows[0].keys()))
        w.writeheader()
        w.writerows(ocr_rows)

    # 6) rationale 逐词转写统计（同图同 rationale，只算被抽中的 30 张）
    rat_rows = []
    for i, r in enumerate(picked, 1):
        m = meta[r["file"]]
        rat_norm = norm(m["rationale"])
        ws = words_of(m["original"])
        transcribed = [wd for wd in ws if norm(wd) and norm(wd) in rat_norm]
        rat_rows.append({
            "图片序号": i, "图片文件": r["file"], "原始语句": m["original"],
            "词数": len(ws), "rationale转写": len(transcribed),
            "转写率": f"{len(transcribed)}/{len(ws)}",
            "转写出的词": ",".join(transcribed),
            "rationale全文": m["rationale"],
        })
    with open(OUT / "readability_rationale.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rat_rows[0].keys()))
        w.writeheader()
        w.writerows(rat_rows)

    # 7) 汇总
    tot_w = sum(x["词数"] for x in ocr_rows)
    ocr_ok = sum(x["OCR逐词恢复"] for x in ocr_rows)
    rat_ok = sum(x["rationale转写"] for x in rat_rows)
    lines = [
        "# 人工可读性实验（GPT-Image-2 · 30 张 Harmful 图）",
        "",
        f"- 抽样：单次评审 Harmful 图中固定种子（{SEED}）随机抽 {N_SAMPLE} 张",
        f"- 逐词计分：共 {tot_w} 个目标词",
        "",
        "| 口径 | 逐词可读率 |",
        "|---|---|",
        f"| macOS Vision OCR（自动，下界） | {ocr_ok}/{tot_w} = {ocr_ok/tot_w:.1%} |",
        f"| 裁判 rationale 转写（辅助，同模型口径） | {rat_ok}/{tot_w} = {rat_ok/tot_w:.1%} |",
        "| 人工标注（主指标） | 待标注 |",
        "",
        "人工标注填完 readability_annotation.csv 的 打分(0/1/2) 列后，",
        "图级可读率 = (所有目标词均≥1 的图) / 30；词级 = 得 2 或 1 的词 / 总词数。",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成，输出目录：{OUT}")


if __name__ == "__main__":
    main()
