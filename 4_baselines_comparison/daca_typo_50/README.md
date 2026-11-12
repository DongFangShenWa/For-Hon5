# 4b. DACA & TYPO vs CollageAttack — Same 50 Prompts on GPT-Image-2

**Answers:** YUQu W5 (comparison with the rendered-harmful-text methods TYPO and ETCH);
numbers also cited by the YUQu W1 response.

## Setup

- Data: the first 50 statements of the main experiment's DGHS sample
  (`sampled_200_hate.csv`), target model GPT-Image-2, judge GPT-4.1 with the paper rubric —
  identical to the main experiment.
- **CollageAttack**: numbers taken from the main-experiment run restricted to the same 50
  prompts (200-prompt full run: ASR 57.5%, avg 4.96).
- **DACA** (divide-and-conquer decomposition): official templates, decomposition and
  optimization LLM = gpt-4.1 (`baseline_daca_run.py`).
- **TYPO** (ACL 2025): run through its **official open-source implementation** (typo.py);
  documented deviations — the optimizer LLM substitutes gpt-4.1 for deepseek-v4-pro, and the
  local Llama-Guard pre-filter is run in permissive mode. Per-prompt budget mirrors the
  official defaults (12 optimizer calls for 9 prompts, 2 for the remaining 41).

## Results (identical 50 prompts, GPT-Image-2, same judge)

| Method | Input-blocked / failed | Best image produced | **ASR** | Avg score of produced images |
|---|---|---|---|---|
| **CollageAttack** | 12/50 | 38/50 | **27/50 = 54.0%** | **7.63 / 10** |
| DACA | 5/50 transform failures + 4/50 blocked | 41/50 | 7/50 = 14.0% | 0.93 / 10 |
| TYPO | 28/50 best candidate rejected → no image | 22/50 | 1/50 = 2.0% (2.0% under its own risk≥4 metric) | 0.54 / 10 |

TYPO's failure is structural: its instruction-dense prompts are caught by GPT-Image-2's
input-side filter (consistent with Llama-Guard-4 rejecting 100% of TYPO-Standard prompts —
see `../../6_defense/`). DACA's metaphor substitution cannot deliver verbatim hate slogans.

## Files

- `results/run_daca_gptimage2/` — DACA per-sample results (items.csv, summary.md, config.json).
- `results/run_typo_merged/` — TYPO final results: `items.csv` + `summary.md` (merged over the
  two budget phases), `records.jsonl` (raw per-iteration records from the official TYPO
  pipeline), `config_typo_official_run.json` (config of the official-code run).
- `code/baseline_daca_run.py` — DACA driver (official templates + gpt-4.1 as decomposition/
  optimizer LLM).
- `code/baseline_typo_run.py` — TYPO driver around the official typo.py (stubs out its
  dashscope/transformers imports not needed for the GPT-Image-2 channel); 
  `baseline_typo_rejudge.py`, `baseline_typo_judge_remaining.py` — post-hoc judging passes
  with the paper rubric.

Images are omitted for size; all per-sample judgments are in the items/records files.
