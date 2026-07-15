---
name: simulated_landscape_scaling
description: "Sampled simulated-landscape benchmarks for surrogate collapse as species count and community size grow"
paths:
  service: GLV_ML/simulated_landscape_scaling.py
  tests: GLV_ML/tests/test_simulated_landscape_scaling.py
  outputs: GLV_ML/outputs/benchmarks/scaling_laws_simulated/
exports:
  - run_config.json
  - simulated_scaling_runs.csv
  - simulated_scaling_metrics.csv
  - simulated_scaling_summary.csv
  - phase2_optimizer_metrics.csv
  - phase2_optimizer_summary.csv
  - best_model_by_band.csv
  - best_strategy_by_band.csv
  - strategy_model_performance.csv
  - per-run interaction tables
  - per-seed max-universe interaction tables
  - per-strategy measured summaries and hidden audit summaries
  - suppressor_auprc_by_species_count.png
  - suppressor_auprc_by_partner_count_band_species_<N>.png
  - suppressor_precision_by_species_count.png
  - suppressor_precision_by_partner_count_band_species_<N>.png
  - suppressor_class_recall_by_species_count.png
  - suppressor_class_recall_by_partner_count_band_species_<N>.png
  - spearman_by_species_count.png
  - spearman_by_partner_count_band_species_<N>.png
  - rmse_by_species_count.png
  - rmse_by_partner_count_band_species_<N>.png
  - best_measured_biomass_by_species_count.png
  - best_measured_biomass_by_partner_count_band_species_<N>.png
  - best_model_by_partner_count_band_<metric>.png
  - best_strategy_by_partner_count_band_<metric>.png
  - strategy_model_performance_<metric>.png
  - phase2_best_validated_biomass_species_<N>.png
consumes:
  - simulation.generate_interaction_data
  - simulation_assay_noise.add_assay_observations
  - simulation_assay_noise.calibrated effect priors
  - ml.TargetBiomassDataset loading and model conventions
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/simulated_landscape_scaling.py"
  tests: ".venv/bin/python -m unittest discover -s GLV_ML/tests -v"
  smoke: ".venv/bin/python GLV_ML/simulated_landscape_scaling.py --species-counts 12 --partner-counts 3-5 --partner-count-bands small:3-5 --proposal-candidate-size 80 --audit-size 40 --audit-fraction 0.25 --budgets 20,40 --batch-size 10 --seeds 1 --models ridge_pairwise --strategies random,size_balanced,max_diversity,bayesian_optimization --phase2-optimizers predicted_best,greedy_forward,simulated_annealing,genetic_algorithm --phase2-top-k 3 --interaction-generator hierarchical --carrying-capacity-min 0.5 --carrying-capacity-max 2.0 --hierarchy-strength 0.15 --hierarchy-noise 0.02 --interaction-response saturating --saturation-pressure 1.0 --target-scale-mapping latent --assay-noise-scale 0 --output-dir GLV_ML/outputs/benchmarks/scaling_laws_simulated_smoke"
---

# Simulated Landscape Scaling Domain

Owns benchmarks that ask whether target-biomass surrogate models keep working as the
simulated species universe grows beyond the exhaustive regime.

The script does **not** enumerate every possible community by default. For each
`species_count x seed`, it:

1. Generates one calibrated max-species GLV interaction table per seed.
2. Takes each requested species count as a nested prefix of that same max-species
   universe, so species-count effects are not confounded with unrelated landscapes.
3. Builds pooled independent max-species calibration landscapes for target-scale mapping,
   so one unusually suppressive interaction table cannot set the scale.
4. Samples a disjoint hidden audit pool uniformly from the requested community space.
5. Lets each exploration strategy propose communities from the remaining combinatorial
   space.
6. Simulates only communities that the strategy actually proposes or probes.
7. Applies the calibrated assay-noise layer when a real-world summary is available.
8. Trains requested surrogate models on increasing measured prefixes from that strategy.
9. Evaluates each model on the fixed audit pool.
10. Runs Phase 2 optimizers against the trained surrogate and validates their top-k
   recommendations with the simulator.
11. Runs the same Phase 2 optimizers against budget-matched noisy simulator measurements,
    then validates their recommendations against noiseless simulator values.

This is separate from `scaling_laws.py`, which downsamples real-world species universes.
Here the axis is simulated design-space scale: more partner species, larger possible
community counts, and fixed measurement budgets.

`--proposal-candidate-size` controls candidate pools for max-diversity, Bayesian
optimization, and Phase 2 `predicted_best`. Walk optimizers evaluate only their own paths.
`--audit-size` is an absolute audit cap; `--audit-fraction` caps the audit to a fraction
of the finite combinatorial space for small species pools.

`run_config.json` records the first invocation's input paths and settings for provenance.
The output directory is the run identity: persisted interaction tables, pools, and
per-community deterministic noise seeds make an interrupted invocation resumable. The
manifest is informational and does not reject a resumed run through a configuration hash
gate; use a new output directory for a scientifically different run. Evaluation datasets
are assembled in memory, so the audit pool is not duplicated into one CSV per budget.
Per-budget metric and Phase 2 checkpoints remain on disk during the sweep and are loaded
only once for final aggregation. Resuming skips complete checkpoints without retaining
their rows, and fitted estimators are released between checkpoints to keep memory bounded
across multi-seed runs.

## Metrics

Measured and audit rows remain distinct. Model-fit metrics are evaluated on the audit
pool, while discovery metrics report the best measured community in the current strategy
prefix. `best_audit_gap` is the target-biomass difference between the best measured
community and the best audit community for the same band. The audit pool remains hidden
from strategy selection and model training.

Model metrics reuse the `ml_benchmark.py` definitions:

- regression: RMSE, MAE, R2, Spearman
- suppressor class: precision, recall, AUPRC
- discovery: lowest measured target biomass, plus gap to the audit best

The suppressor cutoff is anchored to the audit-pool median so it remains fixed for every
budget within one generated landscape.

Each metric is emitted for `partner_count_band = overall` plus configured community-size
bands. The default simulated scaling run uses `--partner-counts 3-18` and bands
`small:3-5`, `medium:6-10`, `large:11-15`, and `very_large:16-18`. This is the main view
for testing whether surrogate performance collapses specifically on larger communities
rather than only when the species universe grows.

Default measurement budgets run in 50-community steps. That granularity is used for all
strategies so model-performance and suppressor-discovery curves are directly comparable.

Model-performance plots use one output file per metric, with model rows and strategy
columns. Discovery plots such as `best_measured_biomass_by_species_count.png` are
strategy-only because the measured-set optimum is independent of which model is later
trained. Within each panel, lines compare species counts across measured-community
budgets. Gap columns remain in the CSV for diagnostics, but lowest measured biomass is
the primary search readout because audit-gap values depend on the sampled audit pool.

Winner summaries are emitted separately from the full metric grids:
`best_model_by_band.csv` averages model scores across measurement strategies and reports
the best model for each species count, partner-count band, measured budget, and metric.
`best_strategy_by_band.csv` averages strategy scores across downstream models and reports
the best measurement/search strategy. The associated winner plots annotate the terminal
winner per band so they remain readable when additional models such as GNN are added.

`strategy_model_performance.csv` answers a separate question: which exploration strategy
produces the best downstream ML model fastest. It averages audit model-performance
metrics across species counts and model classes for each measured budget and strategy.
The corresponding plots use metrics such as suppressor AUPRC, Spearman, and RMSE.

`phase2_optimizer_metrics.csv` is the exploitation readout. For each measured-row budget,
acquisition strategy, model, and optimizer, it records what the optimizer finds when it
walks the surrogate landscape, then validates the top-k recommendations with the
simulator. Rows with `search_source = simulator` are the direct noisy-measurement baseline
for the same optimizer. The key columns are `best_validated_biomass`,
`mean_validated_biomass`, `best_search_score`, and `optimizer_evaluated_count`. This is
not a measured-prefix statistic: it asks when a trained surrogate becomes good enough for
a search algorithm to find low-biomass communities.

Training rows retain configured assay noise. At each measured-row budget, direct simulator
search receives the same number of noisy community measurements: every distinct community
an optimizer scores consumes one measurement, including every neighbor considered by a
greedy step. Repeated queries use the cached observation and are free. The optimizer ranks
its recommendations by those noisy observations; only its final top-k recommendations are
validated with the assay-noise scale fixed to zero. `optimizer_evaluated_count` therefore
equals `measured_count` for direct rows, while surrogate rows report computational model
evaluations and are not measurement-budget limited.

The calibrated assay-noise layer can apply a higher-order partner-count correction, but
the simulated scaling benchmark defaults `--partner-count-effect-scale` to `0.0`. This
keeps lab-like replicate noise while avoiding a baked-in assumption that very large
communities are non-suppressive. Non-zero partner-count effects are calibration
experiments, not default scaling-law evidence.

`--assay-noise-scale` multiplies the fitted replicate-noise standard deviation after
latent GLV biomass is mapped onto the real target scale. The default `1.0` preserves the
calibrated lab-like noise. Use `0.0` for ideal deterministic labels when the question is
how well models recover the simulator without measurement noise.

Native `latent` targets must not be combined with real-assay noise. A run with a real
summary, `target_scale_mapping = latent`, and non-zero assay noise fails before generating
inputs because the real assay SD is not expressed in GLV biomass units.

`--target-scale-mapping quantile` is preferred for noisy assay-like scaling runs. It maps
latent GLV biomass ranks onto the empirical real-world pathogen-signal distribution
before replicate noise is applied, avoiding absolute noise-scale mismatch between GLV
units and assay units. The mapping reference pools five independent max-species
calibration landscapes and is shared across every species-count prefix and seed. The
audit distribution therefore remains hidden and species-count shifts are not separately
normalized away.
`latent` keeps simulated target biomass on the GLV scale and is preferred for noiseless
diagnostics of the simulator itself.
Values outside the calibration distribution are extrapolated rather than clamped to the
real-data sample extrema, avoiding an artificial shared best-biomass floor.
Latent targets use raw fold suppressor thresholds (`median / suppressor_fold`) for audit
metrics; real-scale targets use log-fold thresholds (`median - log(suppressor_fold)`).

`--interaction-generator hierarchical` uses the balanced trait-structured generator from
the simulation domain. Nested species-count prefixes retain broad trait coverage rather
than assigning dominance by species index. It should be evaluated alongside the legacy
generator because the hierarchy terms remain a simulator hypothesis.

`--interaction-response saturating` replaces the linear off-diagonal pressure term with
a bounded row-total response for every species equation. This is the preferred core
simulator mechanism for avoiding runaway additive interactions, because it applies to
the whole community model rather than target-specific post hoc dampening. The
`--saturation-pressure` value controls the total off-diagonal pressure scale where
effects begin to plateau.

`--target-interaction-scale` still exists as a diagnostic target-row rescaling knob, but
simulated scaling defaults it to `1.0`. Prefer global saturation over target-only
dampening for production-style scaling runs.

## Canonical Scientific Framing

This domain should be treated as a two-phase workflow.

Phase 1 is landscape learning. The question is how many measured communities are needed
before a surrogate model is good enough to trust. This phase should use lab-realistic
batch acquisition strategies only: random sampling, size-balanced sampling,
composition-diverse sampling, and model-guided active learning such as Bayesian
optimization or uncertainty/disagreement sampling. A strategy in this phase must answer:
given measured communities so far, which batch should be measured next?

`random` samples uniformly over all available communities, so partner counts appear in
proportion to their combinatorial abundance. `size_balanced` is the explicit alternative
that allocates approximately equal rows across requested partner counts.

Phase 2 is landscape exploitation. Once a surrogate reaches an empirical quality
threshold, optimizers can walk the surrogate-predicted landscape cheaply. The optimizer's
recommended final communities are then validated by the simulator and compared to the
same optimizer spending an equal budget on noisy simulator measurements directly. This
phase answers: which model plus optimizer combination gets close to direct search using
the same experimental budget?

The headline Phase 2 endpoint is validated recommendation quality, not only abstract
audit-set model fit. For each measured-row budget `n`, train the surrogate on the
available `n` rows, run each optimizer on the surrogate-predicted landscape, validate the
optimizer's top recommendation or top-k recommendations with the simulator, and compare
that validated biomass to the same optimizer's direct simulator-walk baseline. This
directly compares acquisition strategy, model class, optimizer, and measurement budget as
one actionable pipeline:

`n measured rows -> surrogate model -> optimizer recommendations -> simulator validation`

This answers the practical question: from what measured-row budget does a model become a
useful surrogate for landscape exploration? AUPRC, RMSE, and Spearman remain diagnostic
model-quality metrics, but they should not be the sole endpoint because they can be opaque
and threshold-dependent.

## Extrapolation and Confounding Guardrails

Calibration against the current real assay constrains an 11-partner system. Species added
for 16-28-species simulations have no real interaction or plate-layout calibration, so
those runs are synthetic sensitivity analyses, not validation that real larger microbial
communities are learnable. Apparent partner-count effects in the real data may also include
plate, dilution, or measurement-design effects. They must not be encoded as biological
laws and then cited as evidence that the resulting synthetic landscapes are predictable.

Overall model metrics can be inflated when target biomass differs mainly by partner count:
a model may learn community cardinality without resolving which species composition works
within a size. Therefore within-partner-count-band RMSE, Spearman, AUPRC, and validated
Phase 2 recommendations are the primary scaling evidence. Overall metrics are supporting
diagnostics. A calibrated regime and a generic/noise-free regime should be treated as
separate sensitivity scenarios; agreement across them is stronger evidence than either
scenario alone.

Sequential oracle optimizers such as greedy local search, simulated annealing, and genetic
algorithms are retained only as measurement-budgeted direct-search baselines. A neighbor,
offspring, or proposal must be measured before its score can guide the next move. They are
not treated as batch-realistic Phase 1 acquisition strategies.

## Threshold Policy

Do not hard-code universal model-quality thresholds yet. Thresholds must be calibrated
from the current simulator regime and audit distribution. Earlier hierarchical latent
runs before the saturating response produced weak model quality (overall Spearman around
0.45 and suppressor AUPRC around 0.25 at best), so those results should be treated as a
failed/diagnostic regime rather than a threshold source.

For the saturating hierarchical latent-noise regime, first run Phase 1 with only
lab-realistic acquisition strategies and inspect saturation curves. Candidate working
thresholds should then be chosen from empirical plateaus, for example:

- Spearman: use the plateau range across species counts and bands, not an arbitrary
  universal cutoff.
- RMSE: normalize against the target-biomass range or mean-baseline RMSE before treating it
  as an actionable threshold.
- Suppressor AUPRC: require that positives are not degenerate; report positive rate next
  to AUPRC so the threshold is interpretable.
- Best-suppressor recall/top-k overlap: use this as the most task-aligned gate when Phase
  2 is "find the best suppressive communities."

The default next run should restrict Phase 1 strategies to:

- `random`
- `size_balanced`
- `max_diversity`
- `bayesian_optimization`

Phase 2 runs as a separate optimizer-on-surrogate evaluation after each trained surrogate.
Sequential oracle optimizers stay out of the acquisition strategy list and are compared by
their simulator-validated recommendations instead.

## Exploration Strategies

The benchmark includes composition-only exploration strategies:

- `random`
- `size_balanced`
- `max_diversity`

It also includes model-based Bayesian optimization:

- `bayesian_optimization`

This strategy fits the shared Gaussian-process surrogate and selects each batch by
Expected Improvement for minimization. It refits on every accumulated measured batch;
it is not a lowest-predicted-ridge shortcut.

The following optimizers are used in Phase 2 to walk the trained surrogate landscape, then
validate their final top-k recommendations with the simulator:

- `predicted_best`
- `genetic_algorithm`
- `greedy_forward`
- `simulated_annealing`
