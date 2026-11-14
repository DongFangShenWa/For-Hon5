# 3. Cross-Category Generalization (AdvBench, 7 Safety Categories)

**Answers:** Hon5 W3/Q4 (is the method tailored to hate speech only?).


## Results

| Target model | Generated | Input-blocked | Harmful | ASR | Avg harmful score |
|---|---|---|---|---|---|
| Doubao-Seedream | 30/30 | 0 | 26/30 | **86.7%** | 6.90 |
| GPT-Image-2 | 21/30 | 9 | 16/30 | 53.3% (conditional 76.2%) | 5.43 |

Per-category (Doubao-Seedream):

| Category | n | Harmful | ASR | Avg score |
|---|---|---|---|---|
| self_harm | 7 | 7 | 100% | 8.57 |
| harass_hate | 4 | 4 | 100% | 9.25 |
| drugs | 2 | 2 | 100% | 7.00 |
| threat_violence | 5 | 4 | 80% | 5.60 |
| weapons_explosives | 4 | 3 | 75% | 5.50 |
| fraud_scam | 4 | 3 | 75% | 5.00 |
| sexual_text | 4 | 3 | 75% | 6.50 |

The attack generalizes beyond hate speech (86.7% overall, on par with the main experiment's
86.9%). On GPT-Image-2 the defense is input-side: of the 21 prompts that got through, 76.2%
were judged harmful.

