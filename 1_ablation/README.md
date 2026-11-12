# 1. Ablation Study — Module-Level + Factor-Level

**Answers:** Hon5 W1/Q1 (corrected Table 3 — COMMAND2/COMMAND3 mix-up), YUQu W6 (why spatial
text composition is effective).

## Setup

- Data: DGHS hate-speech statements. Two independent batches of 30 prompts each
  (batch 2 uses a different 30-prompt draw and anchored carrier materials) — N = 60 total.
- Variants: 7 module combinations (COMMAND1/2/3 and their unions) + 7 single-factor
  perturbations of the full prompt = 14 variants × 30 prompts per batch.
- Generation: Doubao-Seedream, one generation per prompt. Judge: GPT-4.1, paper rubric.
- Variant definitions:
  - **COMMAND1** — scene construction only (no carriers, no text);
  - **COMMAND2** — blank text-bearing carriers + spatial layout + Surface IDs, **no text**;
  - **COMMAND3** — text fragments + rendering styles, attached to carriers via Surface IDs;
  - `factor_*` — single-factor perturbations of the full (C1+C2+C3) prompt.

## Corrected Table 3 (batch 1, judged under the paper's ≤3-recheck rule)

| Combination | Harmful Rate | Avg score |
|---|---|---|
| COMMAND1 (scene) | 0.0% | 0.07 |
| COMMAND2 (blank carriers + layout) | **0.0%** | **0.10** |
| COMMAND3 (text fragments) | **93.3%** | **7.00** |
| C1+C2 | 6.7% | 0.43 |
| C1+C3 | 83.3% | 6.70 |
| C2+C3 | 86.7% | 6.37 |
| C1+C2+C3 (full method) | **100%** | 7.63 |

Source: `batch1/summary_by_variant.csv`, per-sample: `batch1/detail_results.csv`.

## Factor ablation (batch 1)

| Perturbation | Meaning | Harmful Rate | Avg score |
|---|---|---|---|
| factor_fragments_1 | all fragments merged onto carrier 1 | 90.0% | 7.20 |
| factor_fragments_2 | fragments merged onto 2 carriers | 90.0% | 6.83 |
| factor_carrier_generic | scene-fitted carriers → generic blank signs | 90.0% | 7.13 |
| factor_layout_reverse | fragment→carrier mapping reversed | 70.0% | 4.80 |
| factor_layout_random | fragment→carrier mapping randomized | 76.7% | 5.40 |
| factor_no_spatial_constraints | spatial constraints + Surface mapping removed | 80.0% | 6.00 |
| factor_semantic_shift1 | fragments cyclically shifted by 1 (semantic misalignment) | 73.3% | 5.43 |

## Merged N=60 single-pass view (used in the YUQu W6 response)

First judge pass only (batch 1) + batch 2, N = 60 per variant. Source:
`merged_n60/summary_by_variant_n60.csv` (regenerate: `python code/merge_two_batches.py`).

| Variant | Harmful | Avg score | | Variant | Harmful | Avg score |
|---|---|---|---|---|---|---|
| command1 | 1/60 | 0.13 | | full (C1+C2+C3) | 53/60 | 6.88 |
| command2 | 0/60 | 0.00 | | factor_fragments_1 | 54/60 | 7.13 |
| command3 | 56/60 | 7.72 | | factor_fragments_2 | 53/60 | 6.88 |
| C1+C2 | 3/60 | 0.38 | | factor_carrier_generic | 51/60 | 6.93 |
| C1+C3 | 52/60 | 7.02 | | factor_layout_reverse | 44/60 | 5.44 |
| C2+C3 | 53/60 | 6.83 | | factor_layout_random | 48/60 | 6.07 |
| | | | | factor_no_spatial_constraints | 49/60 | 6.52 |
| | | | | factor_semantic_shift1 | 48/60 | 6.05 |

**Takeaways.** (i) Harmfulness is carried entirely by the text module (COMMAND3); blank
carriers alone are harmless (0/60) — no prompt leakage. (ii) Fragment count and carrier type
barely matter (|Δ| ≤ 0.25 vs full). (iii) Breaking the spatial composition consistently hurts
(reversed reading order −1.4, random placement −0.8, semantic shift −0.8).

## Files

- `batch1/` — batch 1 (recheck rule): `summary_by_variant.csv`, `detail_results.csv`,
  `prompt_manifest.csv` (exact prompts per variant), `images/<variant>/sample_*.png` (all 420).
- `batch1_single_pass/` — batch 1 restricted to the first judge pass (single-shot backups).
- `batch2/` — independent replication: same three CSVs + all 420 images.
- `merged_n60/` — merged N=60 summary.
- `code/` — `ablation_generate_and_judge.py` (main pipeline: builds all 14 variants,
  generates, judges, summarizes), `batch2_seed.py` + `expand_to_30.py` (batch-2 /
  30-prompt construction), `merge_two_batches.py` (regenerates the merged summary).
