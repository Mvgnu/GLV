# GLV ML

Tools for generating generalized Lotka-Volterra (GLV) community landscapes, preparing real-world target-biomass data, benchmarking surrogate models, and testing model-guided landscape search.

The examples below assume the existing project virtualenv:

```bash
.venv/bin/python ...
```

## Workflow

1. Prepare real-world summaries:

```bash
.venv/bin/python GLV_ML/rw_prepare.py \
  --input-dir GLV_ML/rw_data \
  --output-dir GLV_ML/outputs/real_world/log \
  --target-transform log1p
```

Alternatively '--target-transform raw' to skip log transformation.

2. Fit assay-noise/effect priors from real data:

```bash
.venv/bin/python GLV_ML/simulation_assay_noise.py \
  --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/calibration/assay_noise
```

Used as a reference for fitting interaction-effect priors that mimic real-world assay-noise structure for the simulation. 

3. Run simulated landscape scaling:

```bash
.venv/bin/python GLV_ML/simulated_landscape_scaling.py \
  --species-counts 12,16,20,24,28 \
  --partner-counts 3-18 \
  --partner-count-bands small:3-5,medium:6-10,large:11-15,very_large:16-18 \
  --proposal-candidate-size 2000 \
  --audit-size 5000 \
  --audit-fraction 0.25 \
  --budgets 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 \
  --batch-size 100 \
  --seeds 3 \
  --models ridge_pairwise,random_forest,hist_gradient_boosting,gnn \
  --strategies random,size_balanced,max_diversity \
  --phase2-optimizers predicted_best,greedy_forward,simulated_annealing,genetic_algorithm \
  --phase2-top-k 5 \
  --interaction-generator hierarchical \
  --carrying-capacity-min 0.5 \
  --carrying-capacity-max 2.0 \
  --hierarchy-strength 0.15 \
  --hierarchy-noise 0.02 \
  --interaction-response saturating \
  --saturation-pressure 1.0 \
  --target-interaction-scale 1 \
  --target-scale-mapping quantile \
  --assay-noise-scale 1 \
  --output-dir GLV_ML/outputs/benchmarks/scaling_laws_simulated_phase2_hierarchical_saturating_quantile_noise
```

Parameter notes:

- `--species-counts`: species-pool sizes to test. The target species is included in each pool.
- `--partner-counts`: number of non-target partners in each community. Total community size is `partner_count + 1` because the target is always present.
- `--partner-count-bands`: reporting bands used to evaluate whether model performance changes across small, medium, and large communities within the same species pool.
- `--proposal-candidate-size`: candidate pool used by max-diversity, Bayesian acquisition, and Phase 2 `predicted_best`. Walk optimizers evaluate only their own paths.
- `--audit-size`: maximum number of hidden audit communities used to evaluate trained models.
- `--audit-fraction`: caps audit rows as a fraction of the finite combinatorial space, mainly to avoid oversized audit pools for smaller species counts.
- `--budgets`: measured-row checkpoints used to train/evaluate models and estimate how many measurements are needed.
- `--batch-size`: acquisition batch size for Bayesian optimization. It does not control the reporting checkpoints when `--budgets` is explicit.
- `--seeds`: number of independent landscape/acquisition repeats.
- `--models`: surrogate models trained at each measured-row budget. `gnn` is much slower than the other listed models.
- `--strategies`: Phase 1 measurement/acquisition strategies that generate the training rows: `random`, `size_balanced`, `max_diversity`, and `bayesian_optimization`.
- `--phase2-optimizers`: Phase 2 optimizers that walk either the trained surrogate landscape or the direct simulator baseline: `predicted_best`, `greedy_forward`, `simulated_annealing`, and `genetic_algorithm`.
- `--phase2-top-k`: number of optimizer recommendations validated and reported.
- `--interaction-generator`: generator used to create the GLV interaction table. `hierarchical` adds trait-like dominant species structure.
- `--carrying-capacity-min` / `--carrying-capacity-max`: range used to set species self-interaction strengths in the hierarchical generator.
- `--hierarchy-strength`: strength of species-level hierarchy effects in generated interactions.
- `--hierarchy-noise`: noise around the hierarchy effects.
- `--interaction-response`: simulated landscape scaling accepts `saturating`; off-diagonal pressure is bounded inside the integrated GLV equations.
- `--saturation-pressure`: pressure scale where saturating interactions begin to plateau.
- `--target-interaction-scale`: multiplier for target-row interactions; useful as a diagnostic target-effect rescaling knob.
- `--target-scale-mapping`: output mapping for target biomass. `quantile` maps latent biomass ranks to the real-world target distribution using an independent calibration landscape shared across species counts; `zscore` uses mean/std mapping; `latent` keeps GLV latent biomass.
- `--assay-noise-scale`: multiplier for fitted assay noise. Use `0` for deterministic labels and `1` for calibrated lab-like noise.
- `--output-dir`: output directory for run metadata, sampled pools, metrics, summaries, and plots.

`random` samples uniformly from all available communities; `size_balanced` deliberately
allocates similar row counts to each requested partner count. Phase 2 training and direct
search both use noisy observations. Direct search spends exactly the current training-row
budget, while final optimizer recommendations are validated against deterministic
zero-noise simulator values. `bayesian_optimization` refits a Gaussian process after each
measured batch and selects the next batch by Expected Improvement. Mapped values extrapolate
beyond the calibration tails instead of being clamped to the real-data extrema.


## Script Reference

### `rw_prepare.py`

Prepares real-world community measurements into one ML-ready summary table and plots diagnostics.

Default input path is `GLV_ML/rw_data`.

Log-transformed pathogen signal:

```bash
.venv/bin/python GLV_ML/rw_prepare.py \
  --input-dir GLV_ML/rw_data \
  --output-dir GLV_ML/outputs/real_world/log \
  --species-prefix sp \
  --target-species pathogen \
  --target-transform log1p
```

Raw pathogen signal:

```bash
.venv/bin/python GLV_ML/rw_prepare.py \
  --input-dir GLV_ML/rw_data \
  --output-dir GLV_ML/outputs/real_world/raw \
  --species-prefix sp \
  --target-species pathogen \
  --target-transform raw
```

Main output:

```text
GLV_ML/outputs/real_world/<log|raw>/rw_summary.csv
```

### `ml_benchmark.py`

Benchmarks regressors for target-species final biomass and suppressor classification.

Depends on:

- a summary CSV with `community`, `final_target_biomass`, and optionally `pathogen_signal_std` / `replicate_count`

Real-world log target:

```bash
.venv/bin/python GLV_ML/ml_benchmark.py \
  GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/ml/rw_log \
  --target-species pathogen \
  --models ridge_main,ridge_pairwise,random_forest,hist_gradient_boosting \
  --suppressor-target-scale log
```

Real-world raw target:

```bash
.venv/bin/python GLV_ML/ml_benchmark.py \
  GLV_ML/outputs/real_world/raw/rw_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/ml/rw_raw \
  --target-species pathogen \
  --models ridge_main,ridge_pairwise,random_forest,hist_gradient_boosting \
  --suppressor-target-scale raw
```

Simulated target:

```bash
.venv/bin/python GLV_ML/ml_benchmark.py \
  GLV_ML/outputs/simulation/exhaustive/all_summary_stats.csv \
  --output-dir GLV_ML/outputs/benchmarks/ml/simulated \
  --target-species sp_012 \
  --species-ids sp_001,sp_002,sp_003,sp_004,sp_005,sp_006,sp_007,sp_008,sp_009,sp_010,sp_011,sp_012 \
  --models ridge_main,ridge_pairwise,random_forest,hist_gradient_boosting
```

### `lotka_volterra.py`

Generates synthetic interaction tables and runs exhaustive GLV simulations over species combinations.

Depends on:

- optional `interaction_effect_prior.csv` from `simulation_assay_noise.py`

Generate a synthetic interaction table:

```bash
.venv/bin/python GLV_ML/lotka_volterra.py \
  --generate-csv GLV_ML/outputs/inputs/generated_12_species.csv \
  --species-count 12 \
  --target-species sp_012 \
  --interaction-generator hierarchical \
  --carrying-capacity-min 0.5 \
  --carrying-capacity-max 2.0 \
  --hierarchy-strength 0.15 \
  --hierarchy-noise 0.02 \
  --off-diagonal-min -0.5 \
  --off-diagonal-max 0.2 \
  --self-interaction -1.0 \
  --target-self-interaction -1.0 \
  --seed 42
```

Simulate all target-containing communities with 3 to 11 partners:

```bash
.venv/bin/python GLV_ML/lotka_volterra.py \
  GLV_ML/outputs/inputs/generated_12_species.csv \
  --target-species sp_012 \
  --min-community-size 3 \
  --max-community-size 11 \
  --initial-density 0.5 \
  --max-time 2000 \
  --time-step 0.1 \
  --output-dir GLV_ML/outputs/simulation/exhaustive
```

Use `--skip-community-plots` for faster non-visual runs.

### `simulation_assay_noise.py`

Fits real-world assay-noise structure and derives interaction-effect priors from a real-world summary. Optionally applies the fitted noise model to an existing simulation summary.

Depends on:

- `GLV_ML/outputs/real_world/log/rw_summary.csv`
- optionally a simulation summary via `--simulation-summary`

Fit assay noise and write priors:

```bash
.venv/bin/python GLV_ML/simulation_assay_noise.py \
  --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/calibration/assay_noise \
  --target-species pathogen
```

Apply fitted noise to a simulation summary:

```bash
.venv/bin/python GLV_ML/simulation_assay_noise.py \
  --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv \
  --simulation-summary GLV_ML/outputs/simulation/exhaustive/all_summary_stats.csv \
  --output-dir GLV_ML/outputs/calibration/assay_noise \
  --target-species pathogen \
  --simulation-target-species sp_012 \
  --species-ids sp_001,sp_002,sp_003,sp_004,sp_005,sp_006,sp_007,sp_008,sp_009,sp_010,sp_011,sp_012
```

Main output:

```text
GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv
```

### `calibrate_simulation_rates.py`

Sweeps GLV effect-scale parameters and compares simulated suppressor rates against real-world suppressor rates.

Calibration can use `--interaction-response linear` as a fast coexistence-equilibrium shortcut for tuning sweeps. Simulated landscape scaling uses the integrated saturating endpoint instead.

Depends on:

- `GLV_ML/outputs/real_world/log/rw_summary.csv`
- `GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv`

Run calibration:

```bash
.venv/bin/python GLV_ML/calibrate_simulation_rates.py \
  --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv \
  --effect-prior-csv GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv \
  --output-dir GLV_ML/outputs/calibration/suppressor_rates \
  --species-count 12 \
  --target-species sp_012 \
  --interaction-response saturating \
  --saturation-pressure 1.0 \
  --endpoint-initial-density 0.5 \
  --endpoint-max-time 500 \
  --seed 42
```

Main outputs:

```text
GLV_ML/outputs/calibration/suppressor_rates/best_calibrated_interactions.csv
GLV_ML/outputs/calibration/suppressor_rates/best_calibrated_summary.csv
GLV_ML/outputs/calibration/suppressor_rates/suppressor_rate_calibration.csv
```

### `simulated_landscape_scaling.py`

Generates sampled simulated landscapes, trains surrogates at increasing measurement budgets, evaluates audit performance, and compares free surrogate walks with direct optimizers spending the same number of noisy simulator measurements.

Depends on:

- `GLV_ML/outputs/real_world/log/rw_summary.csv` for `quantile` or `zscore` target mapping
- optionally `GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv`

Recommended first full run without Bayesian acquisition:

```bash
.venv/bin/python GLV_ML/simulated_landscape_scaling.py \
  --species-counts 12,16,20,24,28 \
  --partner-counts 3-18 \
  --partner-count-bands small:3-5,medium:6-10,large:11-15,very_large:16-18 \
  --proposal-candidate-size 2000 \
  --audit-size 5000 \
  --audit-fraction 0.25 \
  --budgets 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 \
  --batch-size 100 \
  --seeds 3 \
  --models ridge_pairwise,random_forest,hist_gradient_boosting \
  --strategies random,size_balanced,max_diversity \
  --phase2-optimizers predicted_best,greedy_forward,simulated_annealing,genetic_algorithm \
  --phase2-top-k 5 \
  --interaction-generator hierarchical \
  --carrying-capacity-min 0.5 \
  --carrying-capacity-max 2.0 \
  --hierarchy-strength 0.15 \
  --hierarchy-noise 0.02 \
  --interaction-response saturating \
  --saturation-pressure 1.0 \
  --target-interaction-scale 1 \
  --target-scale-mapping quantile \
  --assay-noise-scale 1 \
  --output-dir GLV_ML/outputs/benchmarks/scaling_laws_simulated_phase2_hierarchical_saturating_quantile_noise
```

Noiseless diagnostic run:

```bash
.venv/bin/python GLV_ML/simulated_landscape_scaling.py \
  --species-counts 12,16,20,24,28 \
  --partner-counts 3-18 \
  --partner-count-bands small:3-5,medium:6-10,large:11-15,very_large:16-18 \
  --proposal-candidate-size 2000 \
  --audit-size 5000 \
  --audit-fraction 0.25 \
  --budgets 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 \
  --batch-size 100 \
  --seeds 3 \
  --models ridge_pairwise,random_forest,hist_gradient_boosting \
  --strategies random,size_balanced,max_diversity \
  --phase2-optimizers predicted_best,greedy_forward,simulated_annealing,genetic_algorithm \
  --phase2-top-k 5 \
  --interaction-generator hierarchical \
  --carrying-capacity-min 0.5 \
  --carrying-capacity-max 2.0 \
  --hierarchy-strength 0.15 \
  --hierarchy-noise 0.02 \
  --interaction-response saturating \
  --saturation-pressure 1.0 \
  --target-interaction-scale 1 \
  --target-scale-mapping latent \
  --assay-noise-scale 0 \
  --output-dir GLV_ML/outputs/benchmarks/scaling_laws_simulated_phase2_hierarchical_saturating_latent_nonoise
```

Main outputs:

```text
simulated_scaling_metrics.csv
simulated_scaling_summary.csv
phase2_optimizer_metrics.csv
phase2_optimizer_summary.csv
run_config.json
```

`run_config.json` records the first invocation for provenance. Resume stability comes from
persisted interaction/pool CSVs and deterministic per-community noise seeds. The output
directory is the run identity, so use a new directory for a scientifically different run.

### `active_learning.py`

Replays model-dependent acquisition strategies over a fully materialized summary table. Use this for active-learning comparisons where the full oracle table already exists.

Depends on:

- a materialized summary CSV

Run model-dependent acquisition strategies:

```bash
.venv/bin/python GLV_ML/active_learning.py \
  GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/active_learning/rw_log \
  --target-species pathogen \
  --models ridge_pairwise,random_forest,hist_gradient_boosting \
  --strategies predicted_best,diverse_predicted_best,size_balanced_predicted_best,ensemble_uncertainty,bayesian_optimization \
  --initial-size 50 \
  --batch-size 25 \
  --seeds 5
```

### `selection_baselines.py`

Replays model-independent search and selection methods over a fully materialized summary table. Models are trained only to evaluate downstream landscape learning, not to select rows.

Depends on:

- a materialized summary CSV

Run model-independent baselines:

```bash
.venv/bin/python GLV_ML/selection_baselines.py \
  GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/selection_baselines/rw_log \
  --target-species pathogen \
  --models ridge_pairwise,random_forest,hist_gradient_boosting \
  --methods random,greedy_forward,simulated_annealing,genetic_algorithm,max_diversity,size_balanced \
  --batch-size 25 \
  --seeds 5
```

### `compare_selection_runs.py`

Compares active-learning and selection-baseline summaries.

Depends on:

- `active_learning.py` summary output
- `selection_baselines.py` summary output

Run comparison:

```bash
.venv/bin/python GLV_ML/compare_selection_runs.py \
  --active-summary GLV_ML/outputs/benchmarks/active_learning/rw_log/active_learning_summary.csv \
  --selection-summary GLV_ML/outputs/benchmarks/selection_baselines/rw_log/selection_baseline_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/selection_comparison/rw_log
```

### `compare_target_transforms.py`

Compares raw versus log target benchmark reports.

Depends on:

- `ml_benchmark.py` output for log target
- `ml_benchmark.py` output for raw target

Run comparison:

```bash
.venv/bin/python GLV_ML/compare_target_transforms.py \
  --log-report-dir GLV_ML/outputs/benchmarks/ml/rw_log \
  --raw-report-dir GLV_ML/outputs/benchmarks/ml/rw_raw \
  --output-dir GLV_ML/outputs/real_world/comparisons/target_transform
```

### `scaling_laws.py`

Runs real-data species-universe scaling-law benchmarks by pretending fewer partner species are available.

Depends on:

- a materialized real-world summary CSV

Run real-world scaling laws:

```bash
.venv/bin/python GLV_ML/scaling_laws.py \
  GLV_ML/outputs/real_world/log/rw_summary.csv \
  --output-dir GLV_ML/outputs/benchmarks/scaling_laws_real/log \
  --target-species pathogen \
  --species-counts 5,6,7,8,9,10,11 \
  --universes-per-size 5 \
  --train-grid log \
  --models ridge_pairwise,random_forest,hist_gradient_boosting
```

Parameter notes:

- `--species-counts`: retained partner-species universe sizes. k=3 and k=4 are omitted by default: the suppressor cutoff sits 2-fold below each universe's own median, and universes that small hold almost no such community (prevalence 2.9% and 4.0%), so every suppressor metric there is undefined.
- `--train-grid`: `log` (default) uses geometric row budgets, so the bracket around the requirement is a fixed ratio wide at every `k`. `fraction` restores the old `--train-fractions` grid, whose first checkpoint at k=11 is already 102 measurements.

`scaling_law_required_rows.csv` now brackets each crossing (`<p>_lower_rows`, `<p>_required_rows`, `<p>_censored`) and carries `test_rows` and `true_suppressor_count`, because a universe that never crosses is right-censored rather than missing. Undefined suppressor metrics stay NaN instead of being filled with `0.0`.

### `scaling_law_fits.py`

Fits and compares competing scaling hypotheses for the measurements required, `n_tau(k)`, against the structural sizes of the problem: the full landscape `2^k - 1`, main effects `k`, and the pairwise representation size `k + C(k,2)`.

Depends on:

- `scaling_laws.py` output: `scaling_law_metrics.csv`

Fit and compare hypotheses:

```bash
.venv/bin/python GLV_ML/scaling_law_fits.py \
  GLV_ML/outputs/benchmarks/scaling_laws_real/scaling_law_metrics.csv \
  --output-dir GLV_ML/outputs/benchmarks/scaling_laws_real/fits \
  --primary-metric suppressor_auprc \
  --identifiability-draws 200
```

Four hypotheses are fitted: `constant`, `linear` (`a*k + b`), `power_pairwise` (`a*d(k)^gamma`), and `power_landscape` (`a*(2^k - 1)^beta`). Each exponent tests a reference size directly: `beta = 1` is a constant fraction of the landscape, `gamma = 1` is a constant number of measurements per pairwise coefficient.

Three things about this fit are not optional:

- **Censoring.** The largest budget is `0.75*(2^k - 1)`, so a universe that never reaches the threshold gives `n_tau(k) > max_budget`, not a missing value. Censoring concentrates at small `k`, and dropping those universes steepens `beta` by roughly `+0.2`. Every universe is entered as a censored observation and fitted by maximum likelihood (Tobit).
- **Undefined metrics.** A suppressor metric needs a true suppressor in the held-out split. `--min-test-suppressors` (default 1) drops universes where it cannot exist, into `scaling_law_excluded_universes.csv`. That does not make it *reliable*: at k=5 the split holds a median of 0.67 positives, so raise the threshold to about 5 when the question is what the exponent is.
- **Identifiability.** `--identifiability-draws` simulates from each hypothesis through the real budget grid and reports how often AIC recovers the family that produced it. Run it before believing a ranking.

Parameter notes:

- `--metrics` / `--thresholds`: metric columns and the threshold `tau` defining `n_tau`. One threshold, or one per metric.
- `--min-test-suppressors`: drop universes holding out fewer true suppressors than this.
- `--extrapolate-to`: project the fitted laws out to this universe size.

Main outputs:

```text
scaling_law_fit_comparison.csv
scaling_law_reference_comparison.csv
scaling_law_nested_tests.csv
scaling_law_identifiability.csv
scaling_law_beta_recovery.csv
```

## Notes

- `--batch-size` controls acquisition batch size in some scripts. It does not automatically set reporting checkpoints unless the script says so; use explicit `--budgets` or `--train-sizes` when you need exact reporting points.
- `--assay-noise-scale 0` gives deterministic/noiseless observed labels. `--assay-noise-scale 1` uses the fitted assay-noise scale.
- `target-scale-mapping quantile` is the preferred noisy assay mode. It fits one mapping from an independent max-species calibration landscape for the run and reuses it across measured rows, audit rows, phase-2 validation, species-count prefixes, and seeds. `latent` keeps simulated target biomass on the GLV latent scale for noiseless mechanics checks. `zscore` uses the same independent calibration distribution with mean/std mapping.
- The GNN model is available as `gnn`, but it is slower than the other listed models.
