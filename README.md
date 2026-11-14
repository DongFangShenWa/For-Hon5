# Rebuttal Evidence Package — CollageAttack

Supplementary experimental evidence for the reviewer responses. Each folder contains code and result files; generated images are included where available.

| Folder | Answers | What it contains |
|---|---|---|
| `1_ablation/` | Hon5 W1/Q1, YUQu W6 | Module-level and factor-level ablations |
| `2_readability_ocr/` | Hon5 Q3 | Human word-level readability annotation and multi-engine OCR cross-check |
| `3_advbench_categories/` | Hon5 W3/Q4 | Cross-category generalization on AdvBench across seven safety categories and two target models |
| `4_baselines_comparison/` | 5yoR W1/Q1, YUQu W1/W5 | MPJ (ACL 2025), PGJ, DACA, and TYPO comparisons; Llama-Guard-4 input-filter rates |
| `5_additional_models/` | 5yoR W2/Q2 | Gemini Nano Banana, Midjourney, SDXL, and SD3.5-Large |
| `6_defense/` | YUQu W4 | Llama-Guard-4-12B input-side guard experiment |

## Notes

- **Images.** Generated images are included for selected experiments. For larger runs, images are omitted to keep the package size manageable; per-sample judgments (status, rating, score, rationale) are available in `items.csv` or judge result files. Representative images can be supplied separately.
- **Shared experiments.** The Llama-Guard-4 experiment lives in `6_defense/`; its result (`guard_results.csv`) is duplicated in `4_baselines_comparison/mpj_3models/results/` for the 5yoR W1/Q1 response. Gemini results used for both 5yoR W1/Q1 and W2 are stored in `4_baselines_comparison/mpj_3models/results/` and referenced from `5_additional_models/`.
- **Judge.** Unless stated otherwise, judging uses GPT-4.1 with the paper's harmfulness rubric (1–10 scale; `rating ∈ {Harmful, Harmless}`), temperature = 0. The judge prompt is embedded in each script under `code/`.
- **ASR convention.** ASR = harmful judgments divided by all attempts; input-side rejections count as failures. For the local SD pipelines (no safety checker loaded), there is no input rejection; ASR is calculated over generated images only.
- **Sanitization.** No endpoint URL or API key appears in this package; local absolute paths were stripped from result files. Scripts expect environment variables documented in `.env.example`.
