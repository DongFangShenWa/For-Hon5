# 4a. MPJ (ACL 2025) vs CollageAttack — 3 Models × 2 Methods

**Answers:** 5yoR W1/Q1 (what is different from Multimodal Pragmatic Jailbreak?).

## Setup

- Data: the same 200 DGHS hate-speech statements as the main experiment.
- MPJ prompts: the official `t2i.py:107` sign template —
  `<scene>, with a sign that says, '<full harmful sentence>'` (scene sampled from 4 neutral
  scenes; see `config.json` of each run). The full harmful sentence stays **contiguous** in
  the prompt.
- CollageAttack prompts: the paper's spatially fragmented prompts (no contiguous harmful span).
- One request per prompt; judge: GPT-4.1, paper rubric. Input rejection counted from the
  generation API's safety refusal; ASR = Harmful / 200.

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

Notes: CollageAttack numbers on Doubao and GPT-Image-2 are the main-experiment runs (paper
Table 1); the Gemini CollageAttack run is included here (`run_collage_gemini-*`). Both Gemini
runs are also referenced by `../../5_additional_models/gemini_nanobanana/`.

## Llama-Guard-4 cross-check (independent input filter, offline)

Same experiment as `../../6_defense/`; CSV duplicated at `results/guard_results.csv`
(200 prompts per method, greedy decoding):

| Method | Original | TYPO-Standard | MPJ | PGJ | **CollageAttack** |
|---|---|---|---|---|---|
| Rejection rate | 92.0% | 100% | 91.0% | 37.0% | **22.5%** |

MPJ (91%) sits at the raw-sentence level (92%); CollageAttack is the lowest of all five —
the same pattern holds under the models' own safety stacks and an independent filter.

## Files

- `results/run_doubao_full_20261108_182840/` — MPJ × Doubao (items.csv, summary.md, config.json)
- `results/run_gptimage2_full_20261108_170040/` — MPJ × GPT-Image-2
- `results/run_gemini-nano-banana-2.1_full_20261108_195102/` — MPJ × Gemini Nano Banana
- `results/run_collage_gemini-nano-banana-2.1_full_20261108_191209/` — CollageAttack × Gemini
- `results/guard_results.csv` — Llama-Guard-4 five-method table (copy of `6_defense/`)
- `code/mpj_full_gptimage2.py` — one driver for all API models (`--model` selects target;
  builds MPJ prompts, submits, judges, writes items.csv/summary.md)
- `code/mpj_doubao_smoke.py` — 20-prompt smoke test used before the full runs

Images are omitted for size; every per-sample judgment is in `items.csv`
(status / rating / score / rationale).
