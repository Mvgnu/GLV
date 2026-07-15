---
name: output_layout
description: "Canonical GLV_ML output tree for generated inputs, simulations, calibration, real-world adapters, and benchmarks"
paths:
  root: GLV_ML/outputs/
exports:
  - GLV_ML/outputs/inputs/
  - GLV_ML/outputs/simulation/
  - GLV_ML/outputs/calibration/
  - GLV_ML/outputs/real_world/
  - GLV_ML/outputs/benchmarks/
consumes: []
verification:
  inspect: "find GLV_ML/outputs -maxdepth 3 -type d | sort"
---

# Output Layout

`GLV_ML/outputs/` is the canonical root for generated artifacts. Scripts should not
default to loose repo-root output folders.

## Tree

- `inputs/`
  Generated or imported input material that downstream scripts can reuse without
  redownloading or regenerating raw data.
- `inputs/real_world/`
  Matched plate-reader source files.
- `simulation/`
  GLV simulator outputs.
- `simulation/exhaustive/`
  Exhaustive or near-exhaustive community trajectory/summary runs.
- `simulation/sampled/`
  Future explicit-community simulation batches.
- `calibration/`
  Assay-noise fits, empirical priors, and suppressor-rate calibration artifacts.
- `calibration/assay_noise/`
  Noise model, matched-context effect priors, and ridge coefficient diagnostics.
- `calibration/suppressor_rates/`
  Suppressor/non-suppressor rate calibration sweeps.
- `real_world/`
  Processed real-world summaries and real-world-only comparisons.
- `real_world/log/`
  Default log1p pathogen-signal benchmark summary.
- `real_world/raw/`
  Raw-scale sensitivity summary.
- `real_world/comparisons/`
  Comparisons between real-world target transforms or QC views.
- `benchmarks/`
  ML, active-learning, selection, and scaling benchmark outputs.
- `benchmarks/ml/`
  Standalone model benchmark reports.
- `benchmarks/active_learning/`
  Model-dependent acquisition reports.
- `benchmarks/selection_baselines/`
  Model-independent measurement-order reports.
- `benchmarks/selection_comparison/`
  Combined active-learning vs selection-baseline plots.
- `benchmarks/scaling_laws_real/`
  Real-world species-universe downsampling benchmarks.
- `benchmarks/scaling_laws_simulated/`
  Sampled simulated-landscape scaling benchmarks.

Smoke or exploratory outputs may use sibling names ending in `_smoke`, but canonical
reruns should use the stable directories above.
