---
name: real_world_data
description: "Adapter from matched plate-reader files to benchmark-ready pathogen-signal summaries"
paths:
  service: GLV_ML/rw_prepare.py
  raw: GLV_ML/outputs/inputs/real_world/13022025_data/
  outputs: GLV_ML/outputs/real_world/
exports:
  - rw_replicates.csv
  - rw_summary.csv
  - rw_controls.csv
  - rw_qc.json
  - rw_suppressor_noise.png
  - raw-target variants when `--target-transform raw` is used
consumes:
  - matched biolum/*.csv and OD/*.csv plate files
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/rw_prepare.py"
  smoke: ".venv/bin/python GLV_ML/rw_prepare.py --output-dir GLV_ML/outputs/real_world/log"
  raw_smoke: ".venv/bin/python GLV_ML/rw_prepare.py --output-dir GLV_ML/outputs/real_world/raw --target-transform raw"
---

# Real-World Data Domain

Owns conversion of matched real-world plate-reader files into the summary schema
used by `GLV_ML/ml_benchmark.py`.

Bioluminescence is the primary pathogen-abundance target. OD600 is retained as
QC and a possible secondary endpoint, but it is not used as an input feature for
pathogen prediction because it is measured at the same endpoint.

`Combination` lists partner species only; the pathogen target is implicit.
Control rows are encoded as `[]`. The adapter uses same-design controls for
background correction when available, with the global control median as fallback.

`rw_summary.csv` contains one row per non-control community, aggregating replicate
wells and exposing `community`, `partner_count`, `target_species`, and
`final_target_biomass` for direct benchmark reuse.

By default, `final_target_biomass` is on a natural-log scale (`log1p` of background-corrected
biolum, replicate-averaged); `pathogen_signal_std` is the replicate SD on that same
log scale. The log transform is deliberate: raw biolum variance grows with the mean
(heteroscedastic), so on the raw scale a few bright wells would dominate regression
loss and the dim suppressor wells would look artificially precise. Logging stabilises
the variance and turns differences into fold-changes — but it also reveals that the
suppressor tail is the noisiest part of the data (replicate-SD vs biomass Spearman
-0.53). `rw_suppressor_noise.png` is the diagnostic for this: it shows the strongest
suppressors' +-1 SE bars overlapping (no resolvable single best, even with the
conservative standard-error bar) and the SD-vs-mean trend flipping sign between the
log and raw scales. This is why the benchmark grades a
suppressor *class*, not a within-suppressor ranking (see ml spec).

Use `--target-transform raw` to emit the same schema with unlogged background-corrected
biolum as `final_target_biomass`. This exists for sensitivity/illustration only: it
answers whether benchmark conclusions depend on the log transform, not whether raw scale
is the preferred modelling target.
