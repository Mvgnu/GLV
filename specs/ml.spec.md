---
name: ml
description: "Model benchmarks over simulated GLV community summary outputs"
paths:
  service: GLV_ML/ml_benchmark.py
  tests: GLV_ML/tests/test_simulated_landscape_scaling.py
  transform_comparison: GLV_ML/compare_target_transforms.py
  reports: GLV_ML/outputs/benchmarks/ml/
exports:
  - dataset_from_summary
  - target_biomass_metrics.csv
  - target_biomass_split_metrics.csv
  - target_biomass_skill_vs_baseline.csv
  - target_suppression_by_size.csv
  - target_suppression_by_size_split.csv
  - target_suppressor_classification.csv
  - target_suppressor_classification_split.csv
  - target_biomass_coefficients.csv
  - target_biomass_predictions.csv
  - target biomass, suppression, and suppressor-classification learning-curve / by-size PNG plots
  - target_cutoff_sweep_precision.png / target_cutoff_sweep_recall.png (cutoff-strictness diagnostic)
  - target_percentile_overlap.png (predicted-vs-true bottom-X% placement, gap-free rank view)
  - target_transform_comparison.csv / target_transform_comparison_*.png
consumes:
  - simulation.summary_stats.csv
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/ml_benchmark.py GLV_ML/compare_target_transforms.py"
  tests: ".venv/bin/python -m unittest discover -s GLV_ML/tests -v"
  smoke: ".venv/bin/python GLV_ML/ml_benchmark.py GLV_ML/outputs/simulation/exhaustive/outputs_target_12_species_bounded/all_summary_stats.csv --target-species sp_012 --output-dir GLV_ML/outputs/benchmarks/ml_smoke"
---

# ML Domain

Owns benchmarks that consume simulation summaries and compare models for:

1. Target-species final biomass prediction from partner strain identifiers.
2. Data-efficiency estimation for target-species fitness-landscape prediction.

Input features are multi-hot encoded partner strain identifiers from the `community`
column, excluding the target species because it is present in every sample. Pairwise
models add partner-by-partner interaction features.

The primary regression target is `final_target_biomass`, falling back to
`final_<target_species>` for older summaries.

The kept metrics are:

- `spearman` — does the model order communities correctly. Scale-free, low variance,
  pinned at 0 for a constant predictor; robust because it is a global statistic, not a
  tail-boundary one. Reported overall and per partner count (`target_spearman_by_size.png`).
- `r2` / `rmse` / `mae` — point-prediction quality.
- Suppressor classification (next section) — the "does it predict suppressors?" headline.

`target_biomass_skill_vs_baseline.csv` expresses each model relative to `mean_baseline`
(`spearman_skill`, `rmse_skill`) to make "better than predicting the average?" explicit.

Note on the target modality: `final_target_biomass` is background-subtracted,
log-transformed, replicate-averaged bioluminescence (see real_world_data spec) and is
NOT normalised to cell density. Bioluminescence conflates pathogen cell count with
per-cell light output, so low signal could be true suppression or metabolic quenching;
this cannot be resolved without orthogonal abundance data (CFU/qPCR). In the current
plate data the floor-clamp never bites, replicate noise is ~20% of the between-community
signal, and biolum is negatively correlated with total OD600 (r=-0.49) — strong
suppressors have higher total biomass, which argues against a "nothing grew" artifact.

All CSV exports round float columns to four decimals for readability; in-memory
values used for plots keep full precision.

## Suppressor classification (the noise-honest tail metric)

`target_suppressor_classification.csv` is the headline answer to "can we identify
suppressive communities, data-efficiently?" — the question that originally motivated
the benchmark.

**The one suppressor definition used:** a community is suppressive when its target
biomass is at least `--suppressor-fold` (default 2.0) fold below the **global median**
— i.e. `cutoff = median − ln(2) ≈ 0.69 ln` below the median, a fixed effect-size offset
on the log target, anchored to the whole dataset so the labels do not drift across
splits. (There is no bottom-decile / percentile cutoff; that lived only in the deleted
rank metrics.)

We grade a binary suppressive-vs-not call, NOT a ranking, because of the noise structure:

- The 2× cutoff sits ~0.69 ln below the median, which is ~7× the per-community
  replicate SE (~0.10 ln). So distinguishing a suppressor from a non-suppressor is a
  confident call; only communities within ~2 SE (≈0.19 ln) of the cutoff are genuinely
  borderline (reported as `ambiguous_call_fraction`, ~42% of calls here).
- Fine ranking *within* the suppressor set is mostly noise: ordering two communities
  needs their biomass to differ by ≥~0.27 ln (≈2 SE on the difference), and only ~38% of
  tail community pairs clear that bar. Coarse position is resolvable (the tail spans ~14×
  the SE), but "which community is THE best" is not, so we never grade fine rank.

Every call counts, scored the way a real screen would experience it. We do NOT exclude
communities near the boundary: prospectively you cannot see which calls are borderline,
so you validate them and pay for the misses; dropping them would inflate the reported
precision (an earlier version did exactly that and overstated deployment precision by
~8-14 points, which is why it was removed).
- `suppressor_precision` is operational precision (= 1 − FDR): of all communities the
  model calls suppressive (predicted below the cutoff), the share truly below it. This is
  the deployment number — the fraction of validations that pan out.
- `ambiguous_call_fraction` reports the share of the model's calls within `--buffer-z`
  (default 1.96) SEs of the cutoff. It is a **transparency flag** for how fragile the
  precision is — it does NOT remove any call from precision/recall/AUPRC. On the plate
  data ~42% of calls are borderline.
- `suppressor_auprc` sweeps the threshold over all test communities (precision-recall
  area, robust to the rare positive class). A constant baseline makes no calls
  (precision NaN, recall 0) and sits at the base-rate AUPRC.

**Coarse-ranking ability** (`resolvable_pair_concordance`): of the community pairs whose
true biomass differs by more than `--concordance-z` (default 1.96) SEs — i.e. pairs the
data can actually order — the share the model orders correctly (random 0.5, perfect 1.0;
prediction ties score 0.5, so a constant predictor sits at exactly 0.5). It conditions on
ground-truth resolvability, not on the model's calls, so it is not the precision buffer in
disguise. Reported two ways:

- `concordance_overall` — high (~0.93–0.96 at full data), scored on the **68%** of all
  community pairs that are resolvable, because the model cleanly separates suppressors
  from non-suppressors.
- `concordance_within_suppressors` — near chance (~0.57–0.68), and only **17%** of
  suppressor pairs are even resolvable. So global Spearman/concordance must NOT be read as
  "ranks suppressors well"; almost all the ordering skill is the class split, not
  within-class ranking. (`*_resolvable_fraction` columns report these 68% / 17%.)

Per training size: `target_suppressor_precision_learning_curve.png`,
`target_suppressor_auprc_learning_curve.png`,
`target_concordance_overall_learning_curve.png`,
`target_concordance_within_suppressors_learning_curve.png`.

What this implies (the intended two-stage strategy): the full 11-partner combinatorial
space (2^11-1 = 2047 communities) is already measured, so within it the best suppressors
are known by sorting, not prediction. The ML contribution is **data-efficient coarse
separation** — from ~10-25% of measurements, flag the suppressor subset and tell strong
suppressors from non-suppressors. Fine ranking *within* that subset is not model-
addressable with this data (it is replicate noise), so the natural next stage is
**targeted experimental optimisation inside the ML-identified subset** — sequential
design / hill-climbing / added replicates over a small candidate set, which is where
those search methods legitimately apply (not over the full, noise-limited space).

Learning-curve outputs train on increasing numbers of measured communities to
estimate how many combinations are needed before the target fitness landscape is
predicted accurately enough to prioritize high-biomass combinations.

`ridge_pairwise` is the primary interpretable baseline. Its coefficients are
exported for main-effect and pairwise feature inspection.

`gnn` is a target-aware identity graph baseline. Each community is represented as
a graph with one target node and one node per present partner species; node
features are species identity plus a target flag. It is useful for testing whether
message passing over community membership improves suppressor discovery.

Use `--models` to run a comma-separated model subset when an initial benchmark
should avoid slower neural baselines.

## Target-transform comparison

`GLV_ML/compare_target_transforms.py` compares benchmark reports generated from the same
plate data under different target transforms, currently `log1p` versus raw
background-corrected biolum. It is ingest-only: it does not train models. The intended
use is illustrative sensitivity analysis, showing whether suppressor-class conclusions
change when the target is not log transformed.
