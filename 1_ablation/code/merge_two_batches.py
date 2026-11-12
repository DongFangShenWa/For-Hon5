#!/usr/bin/env python3
"""Merge the two independent 30-sample ablation batches into the N=60 summary.

Batch 1 (single-pass judging):  batch1_single_pass/detail_results_single_shot_backup.csv
Batch 2:                        batch2/detail_results.csv
Output:                         merged_n60/summary_by_variant_n60.csv

Per-sample columns used: variant, rating, score. Harmful = rating == "Harmful".
Scores are averaged over judged rows with a numeric score.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BATCHES = [HERE / "batch1_single_pass" / "detail_results_single_shot_backup.csv",
           HERE / "batch2" / "detail_results.csv"]

def main():
    stats = defaultdict(lambda: {"n": 0, "harmful": 0, "score_sum": 0.0, "score_n": 0})
    for path in BATCHES:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = r.get("variant")
                if not v:
                    continue
                s = stats[v]
                s["n"] += 1
                if r.get("rating") == "Harmful":
                    s["harmful"] += 1
                try:
                    x = float(r["score"])
                    if not math.isnan(x):
                        s["score_sum"] += x
                        s["score_n"] += 1
                except (TypeError, ValueError):
                    pass
    out = HERE / "merged_n60" / "summary_by_variant_n60.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "harmful", "harmful_rate", "avg_score"])
        for v, s in stats.items():
            avg = s["score_sum"] / s["score_n"] if s["score_n"] else float("nan")
            w.writerow([v, s["n"], s["harmful"], f'{s["harmful"] / s["n"]:.4f}', f"{avg:.2f}"])
    print(out.read_text())

if __name__ == "__main__":
    main()
