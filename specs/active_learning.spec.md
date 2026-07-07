---
name: active_learning
description: "Oracle-replay active-learning benchmarks for target-pathogen suppression screens"
paths:
  active_learning: GLV_ML/active_learning.py
  outputs: GLV_ML/outputs/benchmarks/active_learning/
exports:
  - active_learning_rounds.csv
  - active_learning_acquisitions.csv
  - active_learning_summary.csv
  - best_suppressor_gap_model_dependent_by_model.png
  - model_rmse_by_model.png
  - model_rmse_by_strategy.png
  - suppressor_precision_by_model.png
  - suppressor_precision_by_strategy.png
  - suppressor_recall_by_model.png
  - suppressor_recall_by_strategy.png
  - suppressor_auprc_by_model.png
  - suppressor_auprc_by_strategy.png
  - uncertainty and Bayesian acquisition rows when requested
consumes:
  - real_world_data.rw_summary.csv
  - simulation.all_summary_stats.csv
  - ml.TargetBiomassDataset loading and model conventions
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/active_learning.py"
  smoke: ".venv/bin/python GLV_ML/active_learning.py GLV_ML/outputs/real_world/log/rw_summary.csv --target-species pathogen --output-dir GLV_ML/outputs/benchmarks/active_learning_smoke --initial-size 50 --batch-size 25 --rounds 2 --seeds 1 --models ridge_pairwise --strategies predicted_best,diverse_predicted_best"
  smoke_uncertainty: ".venv/bin/python GLV_ML/active_learning.py GLV_ML/outputs/real_world/log/rw_summary.csv --target-species pathogen --output-dir GLV_ML/outputs/benchmarks/active_learning_smoke --initial-size 50 --batch-size 25 --rounds 2 --seeds 1 --models ridge_pairwise --strategies ridge_posterior_uncertainty,ridge_posterior_ucb,committee_disagreement"
---

# Active-Learning Domain

Owns sequential measurement-design benchmarks for target-pathogen biomass prediction
and suppressor discovery. The benchmark uses an oracle replay setup: the full measured
or simulated dataset is already available, but each strategy only sees a small measured
subset at first. New "measurements" are acquired by selecting candidate communities and
looking up their true target biomass in the hidden table.

This domain must not grow `GLV_ML/ml_benchmark.py`. Shared loading/model behavior may be
imported from that module, but model-dependent active-learning logic and acquisition
bookkeeping live in `GLV_ML/active_learning.py`.

Model-independent selection baselines (random, diversity, size-balanced) and the
cross-family comparison live in their own domain — see `selection_baselines.spec.md`. That
sibling imports this module's `round_metrics` / `summarize_rows` so its rounds CSV shares
this schema and the two union directly.

## Script Split

`GLV_ML/active_learning.py` runs model-dependent retraining oracle replay:

1. Start from `--initial-size 50` measured communities.
2. Train the selected surrogate model(s).
3. Score the unmeasured acquisition pool.
4. Select `--batch-size 25` new communities with each acquisition strategy.
5. Reveal their true target biomass from the oracle table.
6. Retrain and repeat for `--rounds`, or until the acquisition pool is exhausted.

## Core Questions

The active-learning benchmark tracks two separate questions:

- **Suppressor discovery:** how quickly does a strategy find communities with low
  target biomass?
- **Landscape learning:** how quickly does the surrogate become accurate over the
  broader candidate space?

These must be reported separately. A strategy can find good suppressors early while
learning a poor global model, and a more exploratory strategy can learn the landscape
well while initially missing the strongest suppressors.

## Data Contract

Inputs follow the `ml` domain target-biomass schema:

- `community`: semicolon-delimited partner identifiers.
- `target_species`: optional if provided by CLI.
- `partner_count`: preferred, otherwise inferred from encoded partners.
- `final_target_biomass`, or `final_<target_species>` fallback.
- optional `pathogen_signal_std` and `replicate_count` for noise-aware diagnostics.

Lower target biomass means stronger suppression. Acquisition and optimizer outputs must
therefore minimize predicted target biomass unless the CLI explicitly asks for a
different objective.

## Candidate Pools

Use a deterministic fixed audit set for model-performance metrics. The audit set is
never acquired. The remaining rows are the acquisition pool. This avoids leakage in
RMSE/Spearman learning curves while still letting suppressor discovery operate over a
realistic unmeasured candidate set.

Initial measured rows should be stratified by `partner_count` where possible so the
first 50 measurements do not collapse into one community size. Acquisition outputs must
record `partner_count` so later comparisons can show whether a strategy only succeeds by
over-concentrating on one size class.

## Initial Strategies

Implement these model-dependent strategies first in `active_learning.py`:

- `predicted_best`: lowest predicted target biomass among unmeasured communities.
- `diverse_predicted_best`: low predicted biomass with Hamming-distance diversity.
- `size_balanced_predicted_best`: low predicted biomass while spreading acquisitions
  across partner counts.
- `ensemble_uncertainty`: highest prediction disagreement across seeded ensemble models.
- `ucb_suppression`: low predicted biomass minus an uncertainty bonus.
- `ridge_posterior_uncertainty`: highest analytic Bayesian-ridge predictive variance.
- `ridge_posterior_ucb`: low predicted biomass minus an analytic-variance bonus.
- `committee_disagreement`: highest prediction variance across distinct model families.
- `bayesian_optimization`: Gaussian-process surrogate scored by Expected Improvement,
  balancing predicted suppression against model uncertainty. Selecting the top-EI batch each
  round makes this batch Bayesian optimization / active learning when `--batch-size > 1`.

`ensemble_uncertainty` and `ucb_suppression` must use actual repeated surrogate fits
to estimate prediction standard deviation. They should not fabricate uncertainty from a
single point regressor.

## Uncertainty Estimation Methods

The acquisition strategies above draw their uncertainty from three distinct mechanisms.
They capture different things and have different failure modes, so the benchmark keeps
them separate rather than collapsing to one "uncertainty" column. **Honesty rule:** every
method may only use the currently measured rows. None may peek at unmeasured candidate
labels, and none may bake in hyperparameters tuned with full-data knowledge.

- **Bootstrap ensemble** (`ensemble_uncertainty`, `ucb_suppression`). Refit the *same*
  model on `ensemble_size` with-replacement resamples of the measured rows; the per-row
  std across fits is the uncertainty. This is access-honest (it only reweights measured
  points, it never invents a measurement), but for a regularised linear model the spread
  is small, fairly flat across the input space, and blind to bias: if the functional form
  is wrong, every member agrees confidently and wrong. It measures variance, not
  misspecification, and does not grow off-manifold.

- **Analytic ridge posterior** (`ridge_posterior_uncertainty`, `ridge_posterior_ucb`).
  Ridge is the MAP of Bayesian linear regression, so the weight posterior is Gaussian and
  the predictive variance `sigma^2 (1 + x^T (X^T X + alpha I)^-1 x)` has a closed form.
  The leverage term grows for candidates poorly spanned by the measured rows, giving a
  real exploration signal off the data manifold — the thing the bootstrap only crudely
  approximates — at one extra matrix solve and no resampling. Same `alpha=1.0` and
  StandardScaler-on-train pairwise features as the loop's ridge model, so it is the exact
  posterior of the model being benchmarked.

- **Query-by-committee** (`committee_disagreement`). Train a committee of distinct model
  *families* (ridge, random forest, hist gradient boosting) on the same measured rows and
  use the per-row variance across families as the uncertainty. Unlike a bootstrap of one
  model class, this captures *structural* uncertainty — regions where linear and tree
  inductive biases diverge — which is often where the truth is genuinely ambiguous. It is
  only as good as the committee's diversity and individual competence at small budgets.

For the current 11-partner real-world screen, exhaustive scoring of all unmeasured
candidates is preferred. Sampling/search approximations are only justified once the
candidate universe is too large to score directly.

## Outputs

`active_learning_rounds.csv` contains one row per seed/model/strategy/round:

- `seed`
- `model`
- `strategy`
- `round`
- `measured_count`
- `new_measurements`
- `audit_rows`
- `pool_rows_remaining`
- `best_measured_biomass`
- `global_best_biomass`
- `global_best_gap`
- `global_best_gap_fraction`
- `suppressor_precision`
- `suppressor_class_recall`
- `suppressor_auprc`
- `audit_rmse`
- `audit_mae`
- `audit_r2`
- `audit_spearman`

`active_learning_acquisitions.csv` contains one row per acquired community:

- `seed`
- `model`
- `strategy`
- `round`
- `acquisition_rank`
- `row_index`
- `community`
- `partner_count`
- `predicted_target_biomass`
- `true_target_biomass`
- optional `acquisition_score`

`active_learning_summary.csv` aggregates repeated seeds by model, strategy, and
measured count. It is the main table for comparing measurement efficiency.

## Model Scope

The first implementation should reuse the stable sklearn models from `ml_benchmark.py`:

- `ridge_pairwise`
- `random_forest`
- `hist_gradient_boosting`

`mean_baseline` may be included as a sanity check, but it is not a useful acquisition
model. `gnn` can be added after the tabular active-learning loop is stable because it is
slower and its uncertainty estimates require extra care.

All preprocessing, feature scaling, feature selection, and model fitting must happen
inside each active-learning round using only the currently measured training rows.

## Plotting

Plot suppressor-discovery and model-learning curves separately:

- best measured suppressor gap vs measured count.
- audit RMSE / Spearman vs measured count.
- suppressor precision / recall / AUPRC vs measured count.
- strategy comparison at fixed budgets, especially 50, 75, 100, 150, 200, 500.

Do not make rank-position the headline metric. Ranking inside the suppressor tail is
noise-limited in the real-world data; use effect-size gaps and suppressor-class metrics.
