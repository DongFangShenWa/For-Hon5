# 4b. DACA & TYPO vs CollageAttack — Same 50 Prompts on GPT-Image-2


## Results (identical 50 prompts, GPT-Image-2, same judge)

| Method | Input-blocked / failed | Best image produced | **ASR** | Avg score of produced images |
|---|---|---|---|---|
| **CollageAttack** | 12/50 | 38/50 | **27/50 = 54.0%** | **7.63 / 10** |
| DACA | 5/50 transform failures + 4/50 blocked | 41/50 | 7/50 = 14.0% | 0.93 / 10 |
| TYPO | 28/50 best candidate rejected → no image | 22/50 | 1/50 = 2.0% (2.0% under its own risk≥4 metric) | 0.54 / 10 |

TYPO's failure is structural: its instruction-dense prompts are caught by GPT-Image-2's
input-side filter. DACA's metaphor substitution cannot deliver verbatim hate slogans.

