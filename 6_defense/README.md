# 6. Defense Evaluation — Independent Input-Side Guard (Llama-Guard-4-12B)

## This folder: offline input-side filtering experiment

- Filter: **Llama-Guard-4-12B** (multimodal safety guard), run locally with a single greedy pass.
- Input: attack prompts from Original, TYPO-Standard, MPJ (sign template), PGJ, and CollageAttack, using the prompts submitted to the target models (`guard_prompt_pack.csv`).

| Method | Original | TYPO-Standard | MPJ | PGJ | **CollageAttack** |
|---|---:|---:|---:|---:|---:|
| Rejection rate | 92.0% | 100% | 91.0% | 37.0% | **22.5%** |

MPJ (91%) is close to the raw-sentence rate (92%). CollageAttack has the lowest rejection rate among the five methods. Its rejected and accepted prompts show no meaningful difference in length or keyword presence; the result is consistent with fragmentation and spatial distribution reducing the likelihood of detection by this input-side guardrail.

These are standalone input-filter rates. The combined ASR of this guard and a target model's own safety system cannot be inferred from the marginal rejection rates alone; it would require per-prompt overlap between their decisions.
