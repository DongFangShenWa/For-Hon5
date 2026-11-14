# 3. Cross-Category Generalization (AdvBench, 7 Safety Categories)

**Answers:** Hon5 W3/Q4 (is the method tailored to hate speech only?).

## Results

| Target model | Generation success rate | Input-block rate | ASR | Conditional ASR | Avg. harmful score |
|---|---:|---:|---:|---:|---:|
| Doubao-Seedream | 100% | 0% | **86.7%** | — | 6.90 |
| GPT-Image-2 | 70.0% | 30.0% | 53.3% | 76.2% | 5.43 |

Per-category results on Doubao-Seedream:

| Category | ASR | Avg. score |
|---|---:|---:|
| self_harm | 100% | 8.57 |
| harass_hate | 100% | 9.25 |
| drugs | 100% | 7.00 |
| threat_violence | 80% | 5.60 |
| weapons_explosives | 75% | 5.50 |
| fraud_scam | 75% | 5.00 |
| sexual_text | 75% | 6.50 |

The attack generalizes beyond hate speech. Its overall ASR on Doubao-Seedream is 86.7%, close to the main experiment's 86.9%, and successful outputs appear across all seven categories. On GPT-Image-2, input filtering blocks some prompts; among the generated images, 76.2% were judged harmful. Counting input blocks as failures gives an overall ASR of 53.3%.
