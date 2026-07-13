---
name: scaling_laws
description: "Species-universe downsampling benchmarks for suppressor-prediction data efficiency"
paths:
  service: GLV_ML/scaling_laws.py
  outputs: GLV_ML/outputs/benchmarks/scaling_laws_real/
exports:
  - scaling_law_datasets.csv
  - scaling_law_metrics.csv
  - scaling_law_summary.csv
  - scaling_law_required_rows.csv
  - scaling_law_precision_required_fraction.png
  - scaling_law_precision_required_rows.png
  - scaling_law_auprc_required_fraction.png
  - scaling_law_auprc_required_rows.png
  - scaling_law_recall_required_fraction.png
  - scaling_law_recall_required_rows.png
  - scaling_law_suppressor_precision_by_model.png
  - scaling_law_suppressor_precision_by_model_rows.png
  - scaling_law_suppressor_precision_by_species.png
  - scaling_law_suppressor_precision_by_species_rows.png
  - scaling_law_suppressor_auprc_by_model.png
  - scaling_law_suppressor_auprc_by_model_rows.png
  - scaling_law_suppressor_auprc_by_species.png
  - scaling_law_suppressor_auprc_by_species_rows.png
  - scaling_law_suppressor_recall_by_model.png
  - scaling_law_suppressor_recall_by_model_rows.png
  - scaling_law_suppressor_recall_by_species.png
  - scaling_law_suppressor_recall_by_species_rows.png
consumes:
  - real_world_data.rw_summary.csv
  - ml target-biomass benchmark conventions
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/scaling_laws.py"
  smoke: ".venv/bin/python GLV_ML/scaling_laws.py GLV_ML/outputs/real_world/log/rw_summary.csv --target-species pathogen --output-dir GLV_ML/outputs/benchmarks/scaling_laws_real_smoke --species-counts 5,6 --universes-per-size 1 --repeat-count 1 --models ridge_pairwise"
---

# Scaling-Laws Domain

Owns benchmarks that ask how target-suppressor prediction changes when the available
partner-species universe is artificially reduced.

The full real-world screen has 11 partner species. A universe-size `k` dataset is made
by choosing `k` retained partners and keeping every community whose partners are all in
that retained set. Communities containing any excluded partner are dropped. The target
pathogen remains implicit and present in every row.

This is not the same as filtering by community partner count. For universe size `k`, keep
all partner-count combinations from 1 through `k` that exist inside that retained species
set. The question is whether the fraction of measured rows needed for suppressor-class
performance saturation stays roughly stable as the possible design space shrinks from
11 partners to 10, 9, 8, 7, 6, and so on. The default sweep covers 5 through 11
retained partner species.

## Outputs

`scaling_law_datasets.csv` records each generated universe:

- `seed`
- `species_count`
- `retained_species`
- `rows`
- `suppressor_count`
- `suppressor_rate`

`scaling_law_metrics.csv` records raw model metrics by universe, model, train rows, and
repeat seed.

`scaling_law_summary.csv` aggregates those metrics.

`scaling_law_required_rows.csv` brackets, per universe/model, the budget reaching each
threshold. A universe that never reaches it is right-censored, not missing, so each
metric prefix writes `<p>_lower_rows`, `<p>_required_rows` (NaN when censored),
`<p>_required_fraction`, and `<p>_censored`, alongside `max_budget`, `test_rows`, and
`true_suppressor_count`. The plots that read `<p>_required_rows` are descriptive and
drop censored universes; `scaling_law_fits` is the inferential consumer and uses both
bounds.

The crossing is taken on an isotonic fit of the learning curve, which is nondecreasing
in expectation but noisy at each budget.

Undefined suppressor metrics stay NaN. A suppressor metric needs at least one true
suppressor among the held-out communities, and filling 0.0 makes an unmeasurable
universe indistinguishable from a model that measured everything and failed.

Budgets default to a geometric row grid (`--train-grid log`), so the bracket around
`n_tau` is a fixed ratio wide at every universe size. A fraction grid spaces checkpoints
by `rows`, which at k=11 puts the first checkpoint at 102 measurements, already past the
requirement it is meant to locate. `--train-grid fraction` restores the old
`--train-fractions` behaviour and `--train-sizes` overrides both.

The default sweep is `5,6,7,8,9,10,11`. The suppressor cutoff sits 2-fold below each
universe's own median, and universes smaller than that hold almost no such community
(prevalence 2.9% at k=3, 4.0% at k=4), so every suppressor metric there is undefined at
every budget. Even at k=5-7 the 25% holdout leaves 0-2 true suppressors, so AUPRC is a
near-degenerate statistic; only k >= 8 holds enough positives to estimate it. See
`scaling_law_fits` for what that costs the fitted exponent.

Plots are emitted in paired views where useful: fraction-based x-axes show scaling
relative to each universe size, while `_rows` plots show the same metrics against
absolute measured row counts for interpretability.

## Downstream

`scaling_law_fits` consumes `scaling_law_metrics.csv` and fits `n_tau(k)` against
`2^k - 1`, `k`, and `k + C(k,2)`. One convention here still matters to it: `split_dataset`
holds out a fixed 25% of each universe and draws training subsets from the remaining 75%,
so the largest budget is `0.75 * rows` and any universe that never reaches a threshold is
right-censored there. That ceiling, and the small held-out splits it implies at low `k`,
are design choices rather than facts about the data. Evaluating on the complement of the
training subset instead would remove the ceiling and roughly triple the held-out
suppressors at k=5-7; it would also change what the fitted exponent means, so it is not
done silently.

## Implementation Notes

Reuse `GLV_ML/ml_benchmark.py` loading/model conventions where possible, but do not grow
that script. This domain is an orchestration layer that creates derived datasets and runs
the existing target-biomass benchmark logic on each one.

Species-universe sampling must be deterministic by seed. The first implementation should
sample several retained species sets per universe size, then aggregate, because one
specific omitted species can dominate performance.

Suppressor cutoff scale is explicit. Use `--suppressor-target-scale log` for the log1p
summary and `--suppressor-target-scale raw` for the raw background-corrected summary.
