---
name: selection_baselines
description: "Model-independent search/selection methods for community discovery: how fast does each find the best suppressor per measurement, and how well does its measured set train a model?"
paths:
  selection_baselines: GLV_ML/selection_baselines.py
  comparison: GLV_ML/compare_selection_runs.py
  outputs: GLV_ML/outputs/benchmarks/selection_baselines/
exports:
  - selection_baselines_rounds.csv
  - selection_baselines_selections.csv
  - selection_baselines_summary.csv
  - best_suppressor_gap_by_method.png
  - model_rmse_by_method.png
  - model_spearman_by_method.png
  - suppressor_auprc_by_method.png
  - selection_comparison.csv
  - selection_comparison_gap.png
  - selection_comparison_exploitation_gap.png
  - selection_comparison_rmse.png
consumes:
  - real_world_data.rw_summary.csv
  - simulation.all_summary_stats.csv
  - ml.TargetBiomassDataset loading and model conventions
  - active_learning.round_metrics / summarize_rows / train_model (shared harness, imported)
  - active_learning.active_learning_rounds.csv (for joint model-dependent vs independent plots)
verification:
  syntax: ".venv/bin/python -m py_compile GLV_ML/selection_baselines.py GLV_ML/compare_selection_runs.py"
  smoke: ".venv/bin/python GLV_ML/selection_baselines.py GLV_ML/outputs/real_world/log/rw_summary.csv --target-species pathogen --output-dir GLV_ML/outputs/benchmarks/selection_baselines_smoke --batch-size 25 --rounds 4 --seeds 1 --models ridge_pairwise --methods greedy_forward,hill_climb,simulated_annealing"
---

# Model-Independent Selection / Search Benchmark

This domain owns `GLV_ML/selection_baselines.py`. It evaluates the classic combinatorial
**objective-guided search heuristics** used in bioinformatics for community/feature
discovery — greedy forward selection, greedy backward elimination, hill climbing,
simulated annealing, and genetic algorithms. These are **model-independent**:
the search is guided by the **true measured biomass** of the communities it has probed, not
by any surrogate's prediction. No ML is involved in deciding what to measure.

It must not grow `GLV_ML/ml_benchmark.py`. Dataset loading, feature construction, and model
building come from `ml_benchmark.py`; the oracle-replay round metrics and summary
aggregation are imported from `active_learning.py` so the rounds schema is identical by
construction. The search heuristics and the trajectory/reporting loop live here.

## Oracle Replay: search by measuring, not by predicting

The full table of communities and their true target biomass exists, but a method only
"knows" a community's biomass once it **measures** (queries) it. A search heuristic explores
the discrete community space; every candidate it evaluates is one measurement (an oracle
lookup). Greedy forward selection grows a community partner-by-partner, at each step
measuring the candidate extensions and keeping the one with the lowest true biomass; hill
climbing measures a community's neighbors and moves downhill; simulated annealing measures
proposed neighbors and accepts them stochastically. The decision rule uses only true,
already-measured values — there is no surrogate in the loop.

This is the correction to the previous `optimizer_search.py`, whose searches scored
candidates with `model.predict` over a *fixed surrogate landscape* and then reported how the
model's single best-predicted pick ranked against the truth. That was circular (the model
both chose and was graded) and ML-dependent. The fix is not to drop the algorithms — it is
to swap their objective from the surrogate's prediction to the oracle measurement. See
**Retired Design**.

## Core Questions

For each method, reported separately (a method can win one and lose the other):

- **Suppressor discovery (primary):** how quickly does the best **measured** community
  approach the true global-best suppressor, as a function of the number of communities
  measured? This needs no model — it is the method's pure discovery performance.
- **Landscape learning (secondary):** if you trained a model on the communities this method
  measured, how accurate is it on a held-out audit set, as a function of measurements? This
  is where a model appears — purely as the *thing being evaluated*, never to guide
  selection. Greedy/hill-climb concentrate measurements near optima, which may train a
  biased (worse) global model than space-filling search — exactly the trade-off to surface.

## Selection / Search Methods

All methods choose what to measure using only community composition and the true biomass of
already-measured communities. None consult a surrogate.

- `greedy_forward`: forward selection — from a small community, repeatedly measure the
  single-partner additions and move to the lowest-biomass one; random restarts.
- `greedy_backward`: backward elimination — from a large community, repeatedly measure the
  single-partner removals and move to the lowest-biomass one; random restarts.
- `hill_climb`: from a random community, measure add/remove/swap neighbors and move to the
  best improving one until no neighbor improves; random restarts.
- `simulated_annealing`: from a random community, measure a proposed neighbor each iteration
  and accept downhill moves always, uphill moves with probability `exp(-Δ/T)`; cooling
  schedule `--start-temperature` × `--cooling`; random restarts.
- `size_balanced`: round-robin across partner-count classes — coverage baseline.

A method that needs predicted biomass to decide is a model-dependent strategy and belongs in
`active_learning.py`, not here.

Uniform random and max-diversity sampling also belong in `active_learning.py`: they are
composition-only acquisition controls for constructing surrogate training sets, not
objective-guided searches that walk already measured biomass values.

## Evaluation Loop (oracle replay)

For each `seed × method`:

1. Build a deterministic fixed audit set (never measured) and a discoverable pool from the
   remaining rows. Searches may only measure communities in the discoverable pool, so
   audit-set model metrics stay leakage-free.
2. Run the method; it produces an ordered list of distinct measured communities (its oracle
   queries) up to a budget (`--rounds × --batch-size`, or the whole pool). Re-querying a
   measured community is free and does not re-count.
3. At a measured-count grid (multiples of `--batch-size`), take the first `k` measured
   communities and record round metrics: the true suppressor gap (model-independent) and the
   audit metrics of each requested model trained on those `k` rows.

The measured-set trajectory is built once per `(seed, method)` and is model-independent;
every requested model trains on the same prefixes, so learning curves stay comparable.

## Selection Methods are Model-Independent

The model never influences which communities a method measures. Models are trained only to
produce the landscape-learning curve. Selection uses the oracle (true measured biomass)
alone — this is what makes "no ML required" literally true for the methods themselves.

## Data Contract & Candidate Pools

Inherited from the `ml` target-biomass schema: `community` (semicolon-delimited partners),
optional `target_species`, `partner_count`, `final_target_biomass` (lower = stronger
suppression). Fixed deterministic audit set, never measured. Selections record
`partner_count` so comparisons can show whether a method only wins by concentrating on one
community size.

## Outputs

`selection_baselines_rounds.csv` — one row per `seed/model/strategy/round`, **identical
schema to `active_learning_rounds.csv`** (the `strategy` column holds the method name), so
the two union directly. `measured_count` is the grid point; metrics come from the shared
`round_metrics` helper (gap + audit regression + suppressor-class columns).

`selection_baselines_selections.csv` — one row per measured community, in measurement order.
**True values only, no `model` column**, because selection is model-independent: `seed`,
`strategy`, `round`, `selection_rank`, `row_index`, `community`, `partner_count`,
`true_target_biomass`.

`selection_baselines_summary.csv` — repeated seeds aggregated by `model`, `strategy`,
`measured_count` (mean and std). The main measurement-efficiency table.

`compare_selection_runs.py` keeps acquisition and exploitation separate. Acquisition
plots compare `global_best_gap_fraction` from measured prefixes. Exploitation plots compare
the active learner's validated top-k surrogate recommendation against the direct search
method's best measured community at the same measurement budget. Matched reports must use
the same dataset, split seed series, measured-count grid, and model list. Both exploitation
gaps use the complete finite screen's best biomass as their common reference; the direct
search still cannot query held-out audit rows.

## Plotting

x-axis = measured communities, discovery and learning plotted separately:

- `best_suppressor_gap_by_method.png` (headline): relative gap vs measured count, one line
  per method (gap is model-independent, so models collapse).
- `model_rmse_by_method.png`, `model_spearman_by_method.png`, `suppressor_auprc_by_method.png`:
  audit metrics vs measured count, one panel per evaluated model, one line per method.

Do not use rank-position as a headline metric. Bounded [0,1] metrics keep a fixed axis;
RMSE autoscales.

## Model Scope

Reuses `ridge_pairwise`, `random_forest`, `hist_gradient_boosting` from `ml_benchmark.py`
for the landscape-learning axis only. All fitting happens on the measured prefix; the model
never influences selection.

## Retired Design

Removed — do not reintroduce:

- The `optimizer_search.py` / `compare_optimizer_runs.py` scripts and their
  `optimizer_search_results.csv`, `optimizer_search_trajectories.csv`, `optimizer_comparison.csv`.
- **Surrogate-scored search**: scoring candidates with `model.predict` over a fixed
  predicted landscape. The search heuristics stay, but their objective is the oracle
  measurement, not a prediction.
- `selected_true_rank_fraction` and any "how well did the model's pick rank" metric
  (circular: grades the model with the model).
- `global_best_gap` defined as the gap of a single best-_predicted_ pick, and
  prediction-ranked "trajectory" rows.
- The `optimizer_search_gap.png` / `optimizer_comparison_gap.png` scatter plots.
