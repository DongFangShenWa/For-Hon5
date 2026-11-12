# Rebuttal Evidence Package — CollageAttack

Supplementary experimental evidence for the reviewer responses. Each folder is self-contained:
sanitized code (all endpoints/keys are read from environment variables — see `.env.example`,
nothing is hard-coded), per-sample result CSVs, and for the 30-sample experiments also all
generated images.

| Folder | Answers | What it contains |
|---|---|---|
| `1_ablation/` | Hon5 W1/Q1, YUQu W6 | Module-level + factor-level ablation, two independent 30-sample batches, merged N=60 summary |
| `2_readability_ocr/` | Hon5 Q3 | Human word-level readability annotation + multi-engine OCR cross-check on 30 images |
| `3_advbench_categories/` | Hon5 W3/Q4 | Cross-category generalization: AdvBench 30 prompts, 7 safety categories, 2 target models |
| `4_baselines_comparison/` | 5yoR W1/Q1, YUQu W1/W5 | MPJ (ACL 2025) on 3 models; PGJ on Gemini; DACA & TYPO on GPT-Image-2; Llama-Guard-4 input-filter rates |
| `5_additional_models/` | 5yoR W2/Q2 | Gemini Nano Banana, Midjourney, SDXL, SD3.5-Large |
| `6_defense/` | YUQu W4 | Llama-Guard-4-12B input-side guard experiment over 5 methods |

## Notes

- **Images.** All generated images are included for the 30-sample experiments (folders 1–3).
  For the 200-prompt runs (folders 4–5) images are omitted for package size; every per-sample
  judgment (status, rating, score, rationale) is in `items.csv` / judge CSVs. Representative
  images are available upon request.
- **Shared experiments.** The Llama-Guard-4 experiment lives in `6_defense/` and its result
  (`guard_results.csv`) is duplicated into `4_baselines_comparison/mpj_3models/results/` for the
  5yoR W1/Q1 response. The Gemini runs underpinning both 5yoR W1/Q1 and W2 live in
  `4_baselines_comparison/mpj_3models/results/` and are referenced from `5_additional_models/`.
- **Judge.** Unless stated otherwise, all judging uses GPT-4.1 with the paper's harmfulness
  rubric (1–10 scale; `rating ∈ {Harmful, Harmless}`), temperature = 0. The exact judge prompt
  is embedded in each script under `code/`.
- **ASR convention.** ASR = (# samples judged Harmful) / (all attempts); input-side rejections
  count as failures. For the local SD pipelines (no safety checker loaded) there is no input
  rejection; ASR is judged on generated images only.
- **Sanitization.** No endpoint URL or API key appears anywhere in this package; local
  absolute paths were stripped from result files. Scripts expect the variables in
  `.env.example`.
