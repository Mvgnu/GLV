---
name: simulation_assay_noise
description: "Calibrate lab-like assay noise from real-world replicates and export empirical pairwise suppressor effects"
paths:
  service: GLV_ML/simulation_assay_noise.py
  calibration: GLV_ML/calibrate_simulation_rates.py
  tests: GLV_ML/tests/test_simulated_landscape_scaling.py
  outputs: GLV_ML/outputs/calibration/
exports:
  - assay_noise_model.json
  - assay_noise_fit.csv
  - assay_noise_diagnostic.png
  - simulated_noisy_summary.csv
  - simulated_noisy_replicates.csv
  - rw_ridge_pairwise_effects.csv
  - rw_pairwise_effect_matrix.csv
  - rw_main_effects.csv
  - interaction_effect_prior.csv
  - suppressor_rate_calibration.csv
  - suppressor_rate_targets.csv
  - suppressor_rate_by_scale.csv
  - best_suppressor_rates.csv
  - best_calibrated_interactions.csv
  - best_calibrated_summary.csv
  - landscape_structure_metrics.csv
  - landscape_species_effects.csv
consumes:
  - real_world_data.rw_summary.csv
  - simulation.all_summary_stats.csv
  - ml_benchmark.TargetBiomassDataset
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/simulation_assay_noise.py"
  tests: ".venv/bin/python -m unittest discover -s GLV_ML/tests -v"
  smoke: ".venv/bin/python GLV_ML/simulation_assay_noise.py --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv --simulation-summary GLV_ML/outputs/simulation/exhaustive/outputs_target_12_species_bounded/all_summary_stats.csv --target-species pathogen --simulation-target-species sp_012 --output-dir GLV_ML/outputs/calibration/assay_noise_smoke"
  calibration_smoke: ".venv/bin/python GLV_ML/calibrate_simulation_rates.py --rw-summary GLV_ML/outputs/real_world/log/rw_summary.csv --effect-prior-csv GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv --output-dir GLV_ML/outputs/calibration/suppressor_rates_smoke --species-count 12 --target-species sp_012 --target-effect-scales 0,0.25 --pair-effect-scales 0,0.25 --partner-count-effect-scales 0,0.5 --partner-count-effect-centers 6.0 --partner-count-effect-widths 2.5"
---

# Simulation Assay-Noise Domain

Owns the bridge between clean GLV latent biomass and lab-like plate-reader summaries.
The GLV simulator remains the latent biology source; this domain wraps its terminal
target biomass with a calibrated assay layer.

The assay layer fits the empirical mean-to-replicate-SD relationship from real-world
summary rows. For current log1p pathogen signal, lower target biomass can have higher
relative uncertainty, so the fitted noise model must be exported and plotted rather than
hidden inside downstream benchmarks.

When a simulation summary is supplied, latent `final_target_biomass` is z-score mapped
onto the empirical real-world target scale, then replicate observations are sampled from
the fitted assay noise model. The output summary preserves the benchmark schema:
`community`, `partner_count`, `target_species`, `final_target_biomass`,
`pathogen_signal_std`, and `replicate_count`.

Callers may pass an assay-noise scale to multiply the fitted replicate standard
deviation. Scale `1.0` is calibrated lab-like noise; scale `0.0` keeps the real-scale
mean mapping but emits deterministic, zero-SD observations for idealized model-capacity
checks.

Callers may also choose the target-scale mapping. `quantile` is the preferred noisy
assay mode: it preserves the simulator ranking while mapping labels onto the empirical
real-world pathogen-signal distribution before noise is applied. Scaling runs fit this
mapping on pooled independent calibration landscapes and reuse it for measured rows, audit
rows, and phase-2 validation. `zscore` uses the same fixed calibration reference with a
mean/std mapping. `latent` leaves GLV biomass on its native nonnegative scale and is best
used for noiseless simulator-mechanics checks.

Quantile mapping interpolates within the calibration distribution and linearly
extrapolates beyond its tails. Assay values remain nonnegative, but they are not clipped
to the empirical minimum or maximum of the real dataset because those sample extrema are
not physical assay bounds.

This domain keeps full-data ridge-pairwise coefficients as model diagnostics, but simulator
priors use matched-community contrasts from the exhaustive real screen. Main effects are
the mean change when one species is added to the same context; pair effects are mean
epistasis across contexts. This preserves the observed mix of suppressive and facilitative
species that correlated regression coefficients can obscure. Pair effects remain output
epistasis, not direct GLV `A_ij` parameters.

Calibration compares simulated and real suppressor/non-suppressor splits plus landscape
structure. Structure diagnostics use matched community contexts to report each species'
marginal target effect and negative-effect rate, then report the context variation of
those effects and pair epistasis. This checks whether a simulator matches only the class
balance or also the heterogeneous signal a model must learn.

`calibrate_simulation_rates.py` owns the fast suppressor-rate calibration loop. It uses
equilibrium-style GLV endpoint estimates rather than full ODE trajectories so scale
sweeps are cheap enough to iterate. The output ranks target-effect, pair-effect, and
partner-count-effect settings by mismatch to real overall and per-partner-count
suppressor rates. The best candidate also exports `landscape_structure_metrics.csv` and
`landscape_species_effects.csv`, with real and simulated rows on standardized target
scales. These are diagnostics, not direct GLV parameter estimates.

The current no-partner-count-correction calibration uses target-effect scale `0.3` and
pair-effect scale `2.5`. Positive matched-context pair epistasis maps to stronger partner
competition, which weakens accumulated suppression in larger communities. These settings
improve the real rate profile but do not reproduce its abrupt eight-partner transition.

The calibration loop also exposes a higher-order partner-count axis:
`--partner-count-effect-scales`, `--partner-count-effect-centers`, and
`--partner-count-effect-widths`. This is not encoded in the pairwise GLV matrix; it is a
phenomenological size-response correction applied before assay noise. Positive scale
makes communities near the center more suppressive while pushing very small and very
large communities away from suppression. It exists specifically to test whether the real
mid-size suppressor band can be reproduced without pretending pairwise coefficients
alone explain it. The adjustment is an absolute function of partner count and therefore
does not change when rows are evaluated individually rather than in one batch.
