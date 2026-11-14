# 4b. DACA & TYPO vs CollageAttack — GPT-Image-2

## Results (matched prompts, same target model and judge)

| Method | Input blocked / failed | **ASR** | Avg. score of produced images |
|---|---:|---:|---:|
| **CollageAttack** | 24.0% | **54.0%** | **7.63 / 10** |
| DACA | 18.0% | 14.0% | 0.93 / 10 |
| TYPO | 56.0% | 2.0% | 0.54 / 10 |

TYPO's failure is structural: its instruction-dense prompts are caught by GPT-Image-2's input-side filter. DACA's metaphor substitution cannot deliver verbatim hate slogans.
