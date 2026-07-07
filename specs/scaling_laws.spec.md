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
  smoke: ".venv/bin/python GLV_ML/scaling_laws.py GLV_ML/outputs/real_world/log/rw_summary.csv --target-species pathogen --output-dir GLV_ML/outputs/benchmarks/scaling_laws_real_smoke --species-counts 3,4 --universes-per-size 1 --repeat-count 1 --models ridge_pairwise --train-fractions 0.1,0.5,1.0"
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
11 partners to 10, 9, 8, 7, 6, and so on. The default sweep covers 3 through 11
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

`scaling_law_required_rows.csv` reports, per universe/model, the smallest measured row
count reaching configurable thresholds such as suppressor precision, AUPRC, Spearman,
or a fraction of full-data performance.

Budgets default to measured fractions (`--train-fractions`) instead of fixed row counts,
so each derived universe is sampled at comparable fractions of its own design space.
`--train-sizes` remains available when exact row counts are needed and overrides the
fraction grid.

Plots are emitted in paired views where useful: fraction-based x-axes show scaling
relative to each universe size, while `_rows` plots show the same metrics against
absolute measured row counts for interpretability.

## Implementation Notes

Reuse `GLV_ML/ml_benchmark.py` loading/model conventions where possible, but do not grow
that script. This domain is an orchestration layer that creates derived datasets and runs
the existing target-biomass benchmark logic on each one.

Species-universe sampling must be deterministic by seed. The first implementation should
sample several retained species sets per universe size, then aggregate, because one
specific omitted species can dominate performance.

Suppressor cutoff scale is explicit. Use `--suppressor-target-scale log` for the log1p
summary and `--suppressor-target-scale raw` for the raw background-corrected summary.
