# 6. Defense Evaluation — Independent Input-Side Guard (Llama-Guard-4-12B)


## This folder: offline input-side filtering experiment

- Filter: **Llama-Guard-4-12B** (multimodal safety guard), run locally, single greedy pass.
- Input: 200 attack prompts per method — Original, TYPO-Standard, MPJ (sign template), PGJ,
  CollageAttack (the same prompts submitted to the target models; `guard_prompt_pack.csv`).

| Method | Original | TYPO-Standard | MPJ | PGJ | **CollageAttack** |
|---|---|---|---|---|---|
| Rejection rate | 92.0% (184/200) | 100% (200/200) | 91.0% (182/200) | 37.0% (74/200) | **22.5% (45/200)** |

MPJ (91%) sits at the raw-sentence level (92%) — any semantic reviewer catches a contiguous
harmful sentence. CollageAttack is the lowest of the five; its rejected vs. passed prompts
show no significant difference in length (1500 vs. 1467 chars) or keyword presence — the
evasion comes from fragmentation + spatial distribution itself, not from obfuscated wording.

Stacked worst-case estimate: even with Llama-Guard-4 placed in front of GPT-Image-2's own
guard, CollageAttack's residual ASR upper bound is ≈ 57.5% × (1 − 22.5%) ≈ 44.6%, vs.
→ 0 for Original.
