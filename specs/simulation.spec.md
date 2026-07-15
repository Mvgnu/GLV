---
name: simulation
description: "GLV dataset generation, community simulation, stop classification, and plots"
paths:
  service: GLV_ML/lotka_volterra.py
  tests: GLV_ML/tests/test_simulated_landscape_scaling.py
  examples: GLV_ML/species_interactions_example.csv
  outputs: GLV_ML/outputs/simulation/
exports:
  - summary_stats.csv
  - eigenvalues.csv
  - all_summary_stats.csv
  - all_eigenvalues.csv
  - community trajectory CSVs
consumes: []
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/lotka_volterra.py"
  tests: ".venv/bin/python -m unittest discover -s GLV_ML/tests -v"
---

# Simulation Domain

Owns the GLV input CSV format, synthetic species generation, community enumeration,
ODE integration, stopping conditions, equilibrium/eigenvalue summaries, and plot
generation.

`summary_stats.csv` is the stable table consumed by downstream analysis. It must
continue to expose `community`, `status`, `stable_at_stop`, `extinct_species`,
`stop_time`, and `final_<species_id>` columns.

Fixed-point stopping uses RMS derivative,
`sqrt(mean(density_derivative_i^2))`, so the threshold is comparable across
community sizes. `max_abs_derivative` remains a diagnostic for single-species
movement at the stopping time.

Generated GLV inputs may use explicit off-diagonal bounds. `--off-diagonal-min`
and `--off-diagonal-max` apply to every interspecies interaction `A_ij` where
`i != j`, while `--self-interaction` controls the diagonal in legacy mode.
`--target-self-interaction` can override the target species diagonal in legacy mode
when target-specific damping is needed.

Generated GLV inputs support two generator modes. `--interaction-generator legacy`
keeps the historical uniform off-diagonal draw and fixed diagonal. `hierarchical`
adds species-level traits from a seeded low-discrepancy sequence. Every nested species
prefix therefore spans suppressive and facilitative trait values without making earlier
species systematically dominant. Trait values set carrying capacities, diagonal terms
as `Aii = -growth_rate / carrying_capacity_i`, and coherent broad off-diagonal effects.
This is intended for testing structured community landscapes rather than independent
random interaction tables.

`--target-interaction-scale` multiplies the target row's off-diagonal interactions
after empirical priors are applied. Values below `1.0` damp direct partner pressure on
the target species in the interaction table itself, which is the core simulator control
for avoiding trivial landscapes where many communities drive the target to exact
extinction.

Calibration sweeps may use `linear` as a fast GLV coexistence-equilibrium approximation.
Production simulated-landscape runs use the integrated `saturating` endpoint, where each
species receives a signed, row-total off-diagonal pressure passed through
`saturation_pressure * tanh(raw_pressure / saturation_pressure)`. This applies to all
species, not only the target, and is intended to create graded suppressor landscapes
without arbitrary target-only rescue terms. The endpoint stops when RMS derivative falls
below `1e-6`, rather than integrating already-settled communities to the configured
maximum time. Diagonal and off-diagonal matrices are prepared once per integration.

Generated GLV inputs may also use empirical target-effect priors from
`simulation_assay_noise.interaction_effect_prior.csv`. For covered species,
`--effect-prior-csv` sets target-row interactions from matched-context main effects,
scaled by `--target-effect-scale`. Matched-context pair effects are target-output epistasis rather
than direct GLV parameters, so `--pair-effect-scale` maps them heuristically onto
partner-partner interactions: positive pair coefficients become stronger partner
competition, which can weaken combined suppression. This is a simulator axis for
calibration experiments, not a claim that empirical output effects are mechanistic `A_ij`
values.

The current real-data calibration uses `target_effect_scale = 0.3` and
`pair_effect_scale = 2.5` without a partner-count correction.

Target-species simulations set `--target-species`. In that mode the target species
is included in every simulated community, `--community-size` / size ranges refer to
partner count, and outputs include `target_species`, `partner_count`, and
`final_target_biomass`. Extinction of a non-target partner must not stop the ODE
integration because the downstream measurement is the target species' terminal mass.
