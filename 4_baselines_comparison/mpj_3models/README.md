# 4a. MPJ (ACL 2025) vs CollageAttack

## Results

| Target model (defense profile) | MPJ input-rejected | MPJ ASR | CollageAttack input-rejected | CollageAttack ASR |
|---|---:|---:|---:|---:|
| Doubao-Seedream (weak native safety) | 2.0% | **97.5%** | 1.0% | 86.0% |
| GPT-Image-2 (strong input-side semantic filter) | **94.5%** | 4.0% | **17.5%** | **57.5%** |
| Gemini Nano Banana (input refusal + output-side dissolving) | 17.0% | 7.0% | 32.0% | **25.0%** |

On GPT-Image-2, MPJ is rejected at a rate of 94.5%, close to the rate for submitting the raw harmful sentence (92.5%); its ASR is 4.0%, compared with 3.5% for the raw sentence. The contiguous sentence provides little evasion. CollageAttack lowers the rejection rate to 17.5% and reaches an ASR of 57.5%. On Gemini, output-side dissolving neutralizes MPJ's full-sentence signs more often (harmful-image rate 8.4%), while CollageAttack's fragmented carriers survive partially (36.8%).

A third method, **PGJ** (Perception-Guided, Huang et al.), was also run on Gemini (`run_pgj_gemini/`): it has an input refusal rate of 8.0% and an ASR of 6.0%, with the same output-side dissolution pattern as MPJ. CollageAttack's fragmented prompts survive the rendering stage more often: 36.8% vs. 8.4%/6.5% harmful-image rates, and 25.0% vs. 7.0%/6.0% ASR.

## Llama-Guard-4 cross-check

| Method | Original | TYPO-Standard | MPJ | PGJ | **CollageAttack** |
|---|---:|---:|---:|---:|---:|
| Rejection rate | 92.0% | 100% | 91.0% | 37.0% | **22.5%** |

MPJ (91%) is close to the raw-sentence rate (92%); CollageAttack is the lowest among the five methods. The same pattern appears under the target models' safety stacks and an independent filter.
