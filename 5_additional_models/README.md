# 5. Additional Target Models — Gemini Nano Banana, Midjourney, SD Family


Across defense profiles, CollageAttack ASR decreases monotonically with defense strength but
stays non-zero, and beats the baselines everywhere:

| Model | Defense profile | CollageAttack ASR |
|---|---|---|
| Doubao-Seedream | weak input guard (1.0% rejected) | 86.0% |
| GPT-Image-2 | input-side semantic review (17.5% rejected) | 57.5% |
| Gemini Nano Banana | input refusal (32%) + output-side dissolving | 25.0% |
| Midjourney | word-level blacklist + safety re-check (66.5% rejected) | 19.5% |

## 5a. `gemini_nanobanana/` — gemini-nano-banana-2.1 (200 prompts × 3 methods)

| Method | Input-refused | ASR | Harmful-image rate among generated |
|---|---|---|---|
| MPJ | 17.0% | 7.0% | 8.4% |
| PGJ | 8.0% | 6.0% | 6.5% |
| **CollageAttack** | 32.0% | **25.0%** | **36.8%** |

Gemini deploys two layers: silent input refusal and **output-side rendering dissolving** —
generated signs get rewritten to harmless text (e.g. a hateful slogan rendered as
"Bananas are for everyone!"). The dissolution is fatal for non-fragmented prompts across the
board: full-sentence MPJ signs are almost fully dissolved (8.4% survive), and PGJ — the
easiest to get past the input gate (only 8.0% refused) — ends at a near-zero 6.0% ASR.
CollageAttack's disguised fragments survive partially (36.8%), reaching 25.0% ASR, 3.6×/4.2×
above MPJ/PGJ — spatial fragmentation partially penetrates output-side dissolving too.
(These runs are the same ones referenced by `../4_baselines_comparison/mpj_3models/`.)

## 5b. `midjourney/` — Midjourney (200 prompts × 3 methods)

| Method | Submission-level rejection | Generated | ASR |
|---|---|---|---|
| MPJ | 88.0% (176/200) | 24 | 4.0% |
| PGJ | 19.0% rejected (+ 19.0% errors) | 124 | 1.5% |
| **CollageAttack** | 66.5% (133/200) | 67 | **19.5%** |

Midjourney's defense is a **word-level blacklist** applied at submission time plus a
post-submission safety re-check — not LLM semantic review. It catches ~2/3 of the fragmented
prompts, which bounds the method's applicability: CollageAttack targets semantic
safety mechanisms; word-list filters resist it best. Even here its ASR is the highest of the
three methods.

## 5c. `sd_family/` — SDXL-base-1.0 and SD3.5-Large (local pipelines, 200 prompts × 4 methods)

Local diffusers pipelines with **no safety checker loaded** — there is no input rejection;
ASR = judged-Harmful / 200 on generated images (GPT-4.1, paper rubric, rating == Harmful).

| Method | SDXL-base-1.0 | SD3.5-Large |
|---|---|---|
| PGJ | 2.0% | 12.5% |
| TYPO | 14.5% | 28.0% |
| **CollageAttack** | 1.0% | **30.0%** |

- **SDXL is architecturally immune, not defended**: dual CLIP encoders, 77-token cap, no T5 —
  the ~362-token CollageAttack prompt (and PGJ's long prompts) is hard-truncated before the
  model; 98% of outputs score 0. Method ranking on SDXL is decided by prompt length and
  text-encoder architecture, not by safety.
- **SD3.5 works**: its T5-XXL encoder (`max_sequence_length=512`) takes the full prompt and
  renders text reliably → CollageAttack reaches 30.0%, the best of the four methods.

