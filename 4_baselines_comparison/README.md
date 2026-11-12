# 4. Baseline Method Comparisons

**Answers:** 5yoR W1/Q1 (vs MPJ, ACL 2025), YUQu W1/W5 (vs DACA and TYPO).

| Sub-folder | Comparison | Setting |
|---|---|---|
| `mpj_3models/` | MPJ vs CollageAttack | 200 DGHS prompts × 3 target models (Doubao, GPT-Image-2, Gemini Nano Banana), plus the Llama-Guard-4 five-method input-filter table |
| `daca_typo_50/` | DACA and TYPO vs CollageAttack | same 50 DGHS prompts, GPT-Image-2, same judge |

Both sub-folders are self-contained (code + per-sample results). The Llama-Guard-4 experiment
itself lives in `../6_defense/`; its result CSV is duplicated in `mpj_3models/results/` for
convenience.
