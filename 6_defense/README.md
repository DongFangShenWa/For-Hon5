# 6. Defense Evaluation — Independent Input-Side Guard (Llama-Guard-4-12B)

**Answers:** YUQu W4 (can CollageAttack bypass existing safety-enhancement methods?).

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

## The other two defense families (results live elsewhere in this package)

2. **Deployed-model defenses** (cross-references): Doubao 1.0% rejected / ASR 86.0%;
   GPT-Image-2 17.5% / 57.5% (`../4_baselines_comparison/mpj_3models/`); Gemini input
   refusal 32% + output-side dissolving, ASR 25.0% (`../5_additional_models/gemini_nanobanana/`);
   Midjourney word-level blacklist 66.5% / 19.5% (`../5_additional_models/midjourney/`).
3. **Output-side detectors**: the GPT-4.1 judge used throughout (same rubric as the paper's
   ASR) *is* an output-side VLM detector — it does detect the harmful content in CollageAttack
   images, showing output-side VLM detection is a viable mitigation direction; Gemini's
   deployed output-side dissolving still lets 36.8% of harmful fragments through.

**Scope note (concept erasure / unlearning):** our threat model is black-box API access;
concept erasure requires white-box weight modification, and existing erased checkpoints cover
only the SD family — whose text rendering is too weak for this attack class anyway
(`../5_additional_models/sd_family/`). We discuss this scope in the revised Limitations.

## Files

- `results/guard_results.csv` — per-prompt guard decision for all 5 methods (raw output,
  label, categories). Also duplicated in `../4_baselines_comparison/mpj_3models/results/`.
- `results/guard_prompt_pack.csv` — the exact 5 × 200 prompts fed to the guard.
- `code/llama_guard4_prompt_check.py` — loads Llama-Guard-4 locally (HF transformers) and
  classifies the prompt pack.
