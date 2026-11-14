# 1. Ablation Study — Module-Level + Factor-Level


## Setup

  - **COMMAND1** — scene construction only (no carriers, no text);
  - **COMMAND2** — blank text-bearing carriers + spatial layout + Surface IDs, **no text**;
  - **COMMAND3** — text fragments + rendering styles, attached to carriers via Surface IDs;

## Corrected Table 3 (judged under the paper's ≤3-recheck rule)

| Component | Avg. Score |
|---|---|
| COMMAND1 | 0.07 |
| COMMAND2 |  0.10 |
| COMMAND3 |  7.00 | 
| C1+C2 |  0.43 |
| C1+C3 |  7.27 | 
| C2+C3 |  7.1 | 
| C1+C2+C3 | 7.63 |

Source: `summary_by_variant.csv`; per-sample results in `detail_results.csv` (420 rows).

## Factor ablation

| Perturbation | Meaning | Harmful Rate | Avg score | Δ vs full (7.63) |
|---|---|---|---|---|
| factor_fragments_1 | all fragments merged onto carrier 1 | 90.0% | 7.20 | −0.43 |
| factor_fragments_2 | fragments merged onto 2 carriers | 90.0% | 6.83 | −0.80 |
| factor_carrier_generic | scene-fitted carriers → generic blank signs | 90.0% | 7.13 | −0.50 |
| factor_layout_reverse | fragment→carrier mapping reversed | 70.0% | 4.80 | −2.83 |
| factor_layout_random | fragment→carrier mapping randomized | 76.7% | 5.40 | −2.23 |
| factor_no_spatial_constraints | spatial constraints + Surface mapping removed | 80.0% | 6.00 | −1.63 |
| factor_semantic_shift1 | fragments cyclically shifted by 1 (semantic misalignment) | 73.3% | 5.43 | −2.20 |

**Takeaways.** 
(i) Harmfulness is carried entirely by the text module (COMMAND3): the
text-free groups are essentially harmless (COMMAND1 0.07, COMMAND2 0.10 and C1+C2 0.43). 
(ii) Fragment count
and carrier type barely matter (Δ = −0.43 / −0.80 / −0.50 vs the full attack). (iii)
Breaking the spatial composition consistently hurts: reversed reading order −2.83, random
placement −2.23, cyclic semantic shift −2.20, removing all spatial constraints −1.63.
