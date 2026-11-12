# 5. Additional Target Models — Gemini Nano Banana, Midjourney, SD Family

**Answers:** 5yoR W2 (commercial models not in the original evaluation), Q2 (why the Stable
Diffusion family was excluded, plus new SD results). Model coverage grows from 5 to 9.

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
| Original | 14.0% | 47.0% |
| PGJ | 2.0% | 12.5% |
| TYPO | 14.5% | 28.0% |
| **CollageAttack** | 1.0% | **30.0%** |

- **SDXL is architecturally immune, not defended**: dual CLIP encoders, 77-token cap, no T5 —
  the ~362-token CollageAttack prompt (and PGJ's long prompts) is hard-truncated before the
  model; 98% of outputs score 0. Method ranking on SDXL is decided by prompt length and
  text-encoder architecture, not by safety.
- **SD3.5 works**: its T5-XXL encoder (`max_sequence_length=512`) takes the full prompt and
  renders text reliably → CollageAttack reaches 30.0%, the best of the four methods.

Implication: applicability is governed by two measurable factors — text-encoder context
capacity and in-image text-rendering ability — both of which grew in the latest model
generation (Nano Banana, GPT-Image, Doubao, FLUX.2, SD3.5). The exposed alignment gap becomes
more general, not less, as models improve.

## Files

- `gemini_nanobanana/run_*/` — items.csv / summary.md / config.json per method (MPJ, PGJ,
  CollageAttack).
- `midjourney/results/run_*/` — same, for MPJ / PGJ / CollageAttack; `midjourney/code/
  mj_collage_full.py` — submission driver (submit → poll → download → judge).
- `sd_family/results_sdxl/`, `sd_family/results_sd35/` — per-sample judge CSVs for all four
  methods (`judge_response` holds the full GPT-4.1 judgment; ASR = rating == Harmful).
- `sd_family/prompts/` — the exact prompt sets for all four methods (collage / original /
  pgj / typo).
- `sd_family/code/` — `attack_model_sdxl.py`, `attack_model_sd35_large.py` (generation),
  `judge_image_score.py` (judging), `prep_baseline_prompts.py`, and the one-click
  `run_baselines_*.sh` / `run_sd*_generate_and_judge.sh` pipelines.

Images for the 200-prompt runs are omitted for size; per-sample judgments are complete in the
CSVs. Representative images available upon request.
