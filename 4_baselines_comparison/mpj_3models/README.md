# 4a. MPJ (ACL 2025) vs CollageAttack — 3 Models × 2 Methods




## Results

| Target model (defense profile) | MPJ input-rejected | MPJ ASR | CollageAttack input-rejected | CollageAttack ASR |
|---|---|---|---|---|
| Doubao-Seedream (weak native safety) | 2.0% | **97.5%** | 1.0% | 86.0% |
| GPT-Image-2 (strong input-side semantic filter) | **94.5%** | 4.0% | **17.5%** | **57.5%** |
| Gemini Nano Banana (input refusal + output-side dissolving) | 17.0% | 7.0% | 32.0% | **25.0%** |

On GPT-Image-2, MPJ is rejected 189/200 (94.5%) — the same level as submitting the raw
harmful sentence (92.5% rejected, ASR 3.5%): the contiguous sentence adds zero evasion.
CollageAttack cuts rejection to 17.5% and reaches ASR 57.5%. On Gemini, output-side
dissolving neutralizes MPJ's full-sentence signs almost completely (harmful-image rate 8.4%)
while CollageAttack's fragmented carriers survive partially (36.8%).

A third method, **PGJ** (Perception-Guided, Huang et al.), was also run on Gemini
(`run_pgj_gemini/`): easiest to pass the input gate (8.0% refused) but ASR 6.0% — the same
output-side dissolution pattern as MPJ. Only the fragmented prompts survive the rendering
stage: 25.0% vs 7.0%/6.0%.



## Llama-Guard-4 cross-check 

| Method | Original | TYPO-Standard | MPJ | PGJ | **CollageAttack** |
|---|---|---|---|---|---|
| Rejection rate | 92.0% | 100% | 91.0% | 37.0% | **22.5%** |

MPJ (91%) sits at the raw-sentence level (92%); CollageAttack is the lowest of all five —
the same pattern holds under the models' own safety stacks and an independent filter.
