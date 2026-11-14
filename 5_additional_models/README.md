# 5. Additional Target Models — Gemini Nano Banana, Midjourney, SD Family

Across the commercial models, CollageAttack ASR decreases as defenses become stronger but remains non-zero and exceeds the baselines. SDXL is an exception because its text encoder truncates the long prompts:

| Model | Defense profile | CollageAttack ASR |
|---|---|---:|
| Doubao-Seedream | weak input guard (1.0% rejected) | 86.0% |
| GPT-Image-2 | input-side semantic review (17.5% rejected) | 57.5% |
| Gemini Nano Banana | input refusal (32%) + output-side dissolving | 25.0% |
| Midjourney | word-level blacklist + safety re-check (66.5% rejected) | 19.5% |

## 5a. `gemini_nanobanana/` — gemini-nano-banana-2.1

| Method | ASR | Harmful-image rate among generated images |
|---|---:|---:|
| MPJ | 7.0% | 8.4% |
| PGJ | 6.0% | 6.5% |
| **CollageAttack** | **25.0%** | **36.8%** |

Gemini uses **output-side rendering mitigation**: generated signs may be rewritten as harmless text (e.g. a hateful slogan rendered as “Bananas are for everyone!”). Full-sentence MPJ signs are often neutralized, while PGJ has a low ASR despite a relatively low input-refusal rate. CollageAttack's fragments survive partially, with a higher harmful-image rate and ASR than MPJ and PGJ. These runs are also referenced in `../4_baselines_comparison/mpj_3models/`.

## 5b. `midjourney/` — Midjourney

| Method | Submission-level rejection | ASR |
|---|---:|---:|
| MPJ | 88.0% | 4.0% |
| PGJ | 19.0% | 1.5% |
| **CollageAttack** | 66.5% | **19.5%** |

Midjourney applies a word-level blacklist at submission time, followed by a safety re-check. It catches many fragmented prompts, which limits the method's applicability: CollageAttack targets semantic safety mechanisms, while word-list filters are more resistant. Even here, CollageAttack has the highest ASR among the three methods.

## 5c. `sd_family/` — SDXL-base-1.0 and SD3.5-Large

Local diffusers pipelines were run without a safety checker. There is no input rejection; ASR is the fraction of generated images judged Harmful (GPT-4.1, paper rubric, rating == Harmful).

| Method | SDXL-base-1.0 | SD3.5-Large |
|---|---:|---:|
| PGJ | 2.0% | 12.5% |
| **MPJ** | 14.5% | 28.0% |
| **CollageAttack** | 1.0% | **30.0%** |

- **SDXL is architecturally constrained, not defended**: dual CLIP encoders, a 77-token cap, and no T5 mean the long CollageAttack prompt (and PGJ's long prompts) is truncated before it reaches the model in full; most outputs score 0. The SDXL ranking is therefore shaped by prompt length and text-encoder architecture, not by safety filtering.
- **SD3.5 handles the full prompt**: its T5-XXL encoder (`max_sequence_length=512`) accommodates the prompt and renders text reliably; CollageAttack reaches 30.0%, the highest ASR among the compared methods.
