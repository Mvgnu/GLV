#!/usr/bin/env python3
"""Model-independent search/selection methods for oracle-replay community discovery.

These are the classic objective-guided combinatorial selection heuristics (greedy
forward/backward, hill climbing, simulated annealing, and genetic algorithms). They are
MODEL-INDEPENDENT:
each candidate a search evaluates is a real measurement (an oracle lookup of true target
biomass), and the search navigates community space using only those true measured values.
No surrogate is involved in deciding what to measure -- that was the circular sin of the old
optimizer_search.py, which scored candidates with model.predict over a fixed surrogate
landscape. Here the objective is the oracle measurement.

Each method produces an ordered list of measured communities. We then report, at a
measured-count grid, two separate axes:
  - suppressor discovery: gap of the best measured community to the true global best (no ML),
  - landscape learning: audit metrics of a model trained on the measured prefix (the model
    is the thing evaluated, never the selector).

The rounds CSV shares ``active_learning_rounds.csv``'s schema (imported round_metrics /
summarize_rows), so ``compare_selection_runs.py`` unions the two. See
``.agent/specs/selection_baselines.spec.md``.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from active_learning import round_metrics, summarize_rows, train_model
from ml_benchmark import (
    load_dataset,
    model_configs,
    parse_model_names,
    parse_species_ids,
    split_dataset,
    write_csv,
)


@dataclass(frozen=True)
class SelectionConfig:
    batch_size: int
    rounds: int | None
    test_size: float
    seed: int
    seeds: int
    suppressor_fold: float
    max_partners: int | None
    iterations: int
    start_temperature: float
    cooling: float
    ga_population: int = 24
    ga_mutation: float = 0.1
    restart_stall_limit: int = 3


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("--seeds must be at least 1")
    return parsed


def method_offset(method: str) -> int:
    """Deterministic per-method RNG offset (hash() is salted, so roll our own)."""
    offset = 0
    for char in method:
        offset = (offset * 131 + ord(char)) % 1_000_003
    return offset


# --- Community-space bookkeeping (presence vectors <-> dataset rows). ---


def vector_key(presence_row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in presence_row.tolist())


def build_vector_index(dataset) -> dict[tuple[int, ...], int]:
    return {vector_key(dataset.presence[index]): index for index in range(len(dataset.presence))}


def candidate_keys(
    dataset,
    candidate_indices: np.ndarray,
    vector_index: dict[tuple[int, ...], int],
    max_partners: int | None,
) -> list[tuple[int, ...]]:
    keys = []
    for index in candidate_indices:
        key = vector_key(dataset.presence[index])
        if key not in vector_index:
            continue
        if max_partners is not None and sum(key) > max_partners:
            continue
        keys.append(key)
    return keys


def add_neighbors(key: tuple[int, ...], valid_key_set: set) -> list[tuple[int, ...]]:
    out = []
    for index in range(len(key)):
        if key[index] == 0:
            moved = list(key)
            moved[index] = 1
            moved_key = tuple(moved)
            if moved_key in valid_key_set:
                out.append(moved_key)
    return out


def remove_neighbors(key: tuple[int, ...], valid_key_set: set) -> list[tuple[int, ...]]:
    out = []
    for index in range(len(key)):
        if key[index] == 1:
            moved = list(key)
            moved[index] = 0
            moved_key = tuple(moved)
            if any(moved_key) and moved_key in valid_key_set:
                out.append(moved_key)
    return out


def valid_neighbors(key: tuple[int, ...], valid_key_set: set) -> list[tuple[int, ...]]:
    """Add / remove / swap one partner, restricted to measurable non-empty communities."""
    neighbors = add_neighbors(key, valid_key_set) + remove_neighbors(key, valid_key_set)
    present = [index for index in range(len(key)) if key[index] == 1]
    absent = [index for index in range(len(key)) if key[index] == 0]
    for remove_index in present:
        for add_index in absent:
            moved = list(key)
            moved[remove_index] = 0
            moved[add_index] = 1
            moved_key = tuple(moved)
            if moved_key in valid_key_set:
                neighbors.append(moved_key)
    return neighbors


# --- The oracle: measuring records the (distinct) order and returns the true value. ---


class Oracle:
    def __init__(self, dataset, vector_index, valid_keys, budget: int):
        self.target_biomass = dataset.target_biomass
        self.vector_index = vector_index
        self.valid_keys = valid_keys
        self.valid_key_set = set(valid_keys)
        self.budget = budget
        self.order: list[int] = []
        self._seen: set[int] = set()

    def measure(self, key: tuple[int, ...]) -> float:
        index = self.vector_index[key]
        if index not in self._seen:
            self._seen.add(index)
            self.order.append(index)
        return float(self.target_biomass[index])

    @property
    def done(self) -> bool:
        return len(self.order) >= self.budget or len(self._seen) >= len(self.valid_keys)

    def random_key(self, rng) -> tuple[int, ...]:
        return self.valid_keys[int(rng.integers(len(self.valid_keys)))]


# --- Search heuristics: each drives the oracle and stops at the budget. ---


def run_greedy(oracle: Oracle, rng, neighbor_fn, stall_limit: int) -> None:
    stalls = 0
    while not oracle.done and stalls < stall_limit:
        before = len(oracle.order)
        current = oracle.random_key(rng)
        oracle.measure(current)
        while not oracle.done:
            candidates = neighbor_fn(current, oracle.valid_key_set)
            if not candidates:
                break
            scores = [oracle.measure(candidate) for candidate in candidates]
            best = candidates[int(np.argmin(scores))]
            if oracle.measure(best) >= oracle.measure(current):
                break
            current = best
        stalls = 0 if len(oracle.order) > before else stalls + 1


def run_hill_climb(oracle: Oracle, rng, stall_limit: int) -> None:
    stalls = 0
    while not oracle.done and stalls < stall_limit:
        before = len(oracle.order)
        current = oracle.random_key(rng)
        current_score = oracle.measure(current)
        while not oracle.done:
            neighbors = valid_neighbors(current, oracle.valid_key_set)
            if not neighbors:
                break
            scores = [oracle.measure(neighbor) for neighbor in neighbors]
            best_position = int(np.argmin(scores))
            if scores[best_position] >= current_score:
                break
            current = neighbors[best_position]
            current_score = scores[best_position]
        stalls = 0 if len(oracle.order) > before else stalls + 1


def run_simulated_annealing(oracle: Oracle, rng, config: SelectionConfig) -> None:
    stalls = 0
    while not oracle.done and stalls < config.restart_stall_limit:
        before = len(oracle.order)
        current = oracle.random_key(rng)
        current_score = oracle.measure(current)
        temperature = config.start_temperature
        for _iteration in range(config.iterations):
            if oracle.done:
                break
            neighbors = valid_neighbors(current, oracle.valid_key_set)
            if not neighbors:
                break
            proposal = neighbors[int(rng.integers(len(neighbors)))]
            proposal_score = oracle.measure(proposal)
            delta = proposal_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                current = proposal
                current_score = proposal_score
            temperature *= config.cooling
        stalls = 0 if len(oracle.order) > before else stalls + 1


def _tournament_select(population, fitness, rng, size: int = 3):
    positions = rng.integers(0, len(population), size=size)
    best = min((int(p) for p in positions), key=lambda p: fitness[p])
    return population[best]


def _crossover(parent_a, parent_b, rng) -> np.ndarray:
    mask = rng.integers(0, 2, size=len(parent_a)).astype(bool)
    return np.where(mask, np.array(parent_a), np.array(parent_b))


def _mutate(vector: np.ndarray, rate: float, rng) -> np.ndarray:
    flips = rng.random(len(vector)) < rate
    mutated = vector.copy()
    mutated[flips] = 1 - mutated[flips]
    return mutated


def _snap_to_valid(vector, valid_presence, valid_keys, rng) -> tuple[int, ...]:
    distances = np.abs(valid_presence - vector).sum(axis=1)
    best = np.flatnonzero(distances == distances.min())
    return valid_keys[int(best[rng.integers(len(best))])]


def run_genetic_algorithm(oracle: Oracle, rng, valid_presence: np.ndarray, config: SelectionConfig) -> None:
    """Evolve binary community vectors; fitness is the oracle measurement (lower = better).

    Offspring that fall outside the measurable set are snapped to the nearest valid
    community, so the GA explores only communities the oracle can actually measure.
    """
    n_partners = valid_presence.shape[1]
    population = [oracle.random_key(rng) for _ in range(config.ga_population)]
    fitness = [oracle.measure(key) for key in population]
    stalls = 0
    while not oracle.done and stalls < config.restart_stall_limit:
        before = len(oracle.order)
        elite = population[int(np.argmin(fitness))]
        offspring = [elite]
        while len(offspring) < config.ga_population:
            parent_a = _tournament_select(population, fitness, rng)
            parent_b = _tournament_select(population, fitness, rng)
            child = _mutate(_crossover(parent_a, parent_b, rng), config.ga_mutation, rng)
            if not child.any():
                child[int(rng.integers(n_partners))] = 1
            child_key = tuple(int(value) for value in child)
            if child_key not in oracle.valid_key_set:
                child_key = _snap_to_valid(child, valid_presence, oracle.valid_keys, rng)
            offspring.append(child_key)
        population = offspring
        fitness = [oracle.measure(key) for key in population]
        stalls = 0 if len(oracle.order) > before else stalls + 1


# --- Partner-count coverage control: composition-only order, no oracle decisions. ---


def size_balanced_positions(partner_counts: np.ndarray, rng) -> list[int]:
    groups: dict[int, list[int]] = {}
    for position, partner_count in enumerate(partner_counts):
        groups.setdefault(int(partner_count), []).append(position)
    for partner_count in groups:
        rng.shuffle(groups[partner_count])
    order: list[int] = []
    classes = sorted(groups)
    while any(groups.values()):
        for partner_count in classes:
            if groups[partner_count]:
                order.append(groups[partner_count].pop())
    return order


def measured_order(
    method: str,
    dataset,
    valid_keys: list[tuple[int, ...]],
    valid_indices: np.ndarray,
    vector_index: dict[tuple[int, ...], int],
    budget: int,
    rng: np.random.Generator,
    config: SelectionConfig,
) -> list[int]:
    """Return the order in which a method measures distinct communities (row indices)."""
    oracle = Oracle(dataset, vector_index, valid_keys, budget)

    if method == "size_balanced":
        positions = size_balanced_positions(dataset.partner_counts[valid_indices], rng)
        for position in positions:
            if oracle.done:
                break
            oracle.measure(valid_keys[position])
        return oracle.order

    if method == "greedy_forward":
        run_greedy(oracle, rng, add_neighbors, config.restart_stall_limit)
    elif method == "greedy_backward":
        run_greedy(oracle, rng, remove_neighbors, config.restart_stall_limit)
    elif method == "hill_climb":
        run_hill_climb(oracle, rng, config.restart_stall_limit)
    elif method == "simulated_annealing":
        run_simulated_annealing(oracle, rng, config)
    elif method == "genetic_algorithm":
        run_genetic_algorithm(oracle, rng, dataset.presence[valid_indices], config)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    return oracle.order


def selection_rows(
    dataset,
    method: str,
    seed: int,
    order: list[int],
) -> list[dict[str, object]]:
    """One row per measured community, in measurement order -- true values only."""
    rows = []
    for rank, row_index in enumerate(order, start=1):
        rows.append({
            "seed": seed,
            "strategy": method,
            "selection_rank": rank,
            "row_index": int(row_index),
            "community": dataset.communities[row_index],
            "partner_count": int(dataset.partner_counts[row_index]),
            "true_target_biomass": float(dataset.target_biomass[row_index]),
        })
    return rows


def measured_count_grid(batch_size: int, n_measured: int) -> list[int]:
    # Clean multiples of batch_size so methods share an x-grid; a method whose trajectory
    # ends earlier simply has a shorter line (no ragged endpoint points).
    grid = list(range(batch_size, n_measured + 1, batch_size))
    if not grid and n_measured > 0:
        grid = [n_measured]
    return grid


# --- Plotting: discovery (headline) and landscape-learning curves, separately. ---


def plot_gap_by_method(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    grouped = (
        summary.groupby(["strategy", "measured_count"])["global_best_gap_fraction"]
        .mean()
        .reset_index()
    )
    for method, group in grouped.groupby("strategy"):
        group = group.sort_values("measured_count")
        ax.plot(group["measured_count"], group["global_best_gap_fraction"], marker="o", linewidth=2, label=method)
    ax.set_title("Suppressor Discovery (model-independent search)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Measured communities")
    ax.set_ylabel("Relative gap to best suppressor")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_method(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = (0, 1.02),
) -> None:
    models = sorted(summary["model"].unique())
    columns = min(3, len(models))
    rows = int((len(models) + columns - 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows), dpi=120, squeeze=False)
    panels = axes.ravel()
    for index, model_name in enumerate(models):
        ax = panels[index]
        model_rows = summary[summary["model"] == model_name]
        for method, group in model_rows.groupby("strategy"):
            group = group.sort_values("measured_count")
            ax.plot(group["measured_count"], group[metric], marker="o", linewidth=1.8, label=method)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Measured communities")
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    for empty_index in range(len(models), len(panels)):
        panels[empty_index].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def write_selection_plots(summary: pd.DataFrame, output_path: Path) -> None:
    plot_gap_by_method(summary, output_path / "best_suppressor_gap_by_method.png")
    plot_metric_by_method(summary, "rmse", output_path / "model_rmse_by_method.png",
                          "Model RMSE by Selection Method", "RMSE", y_limits=None)
    plot_metric_by_method(summary, "spearman", output_path / "model_spearman_by_method.png",
                          "Model Spearman by Selection Method", "Spearman")
    plot_metric_by_method(summary, "suppressor_auprc", output_path / "suppressor_auprc_by_method.png",
                          "Suppressor AUPRC by Selection Method", "Suppressor AUPRC")


def run_selection_baselines(
    summary_path: str,
    output_dir: str,
    species_ids: list[str] | None,
    target_species: str | None,
    selected_models: list[str] | None,
    methods: list[str],
    config: SelectionConfig,
) -> tuple[Path, Path, Path]:
    dataset = load_dataset(summary_path, target_species, species_ids)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    median_biomass = float(np.median(dataset.target_biomass))
    vector_index = build_vector_index(dataset)

    round_output_rows: list[dict[str, object]] = []
    selection_output_rows: list[dict[str, object]] = []

    for run_index in range(config.seeds):
        split_seed = config.seed + run_index
        split = split_dataset(dataset, config.test_size, split_seed)
        discoverable_indices = np.array(split.train_indices, dtype=int)
        audit_indices = np.array(split.test_indices, dtype=int)
        valid_keys = candidate_keys(dataset, discoverable_indices, vector_index, config.max_partners)
        if not valid_keys:
            continue
        valid_indices = np.array([vector_index[key] for key in valid_keys], dtype=int)
        budget = config.rounds * config.batch_size if config.rounds else len(valid_keys)
        budget = min(budget, len(valid_keys))

        for method in methods:
            # The measured-set trajectory is model-independent: built once, then every
            # requested model trains on the same prefixes for comparable learning curves.
            order = measured_order(
                method,
                dataset,
                valid_keys,
                valid_indices,
                vector_index,
                budget,
                np.random.default_rng(split_seed + method_offset(method)),
                config,
            )
            selection_output_rows.extend(selection_rows(dataset, method, split_seed, order))

            grid = measured_count_grid(config.batch_size, len(order))
            for model_name, feature_set in model_configs(selected_models):
                previous_k = 0
                for round_index, measured_k in enumerate(grid, start=1):
                    measured_set = np.array(order[:measured_k], dtype=int)
                    model, features = train_model(
                        dataset, model_name, feature_set, measured_set, split_seed + round_index
                    )
                    metrics = round_metrics(
                        dataset, model, features, audit_indices, measured_set,
                        discoverable_indices, median_biomass, config.suppressor_fold,
                    )
                    round_output_rows.append({
                        "seed": split_seed,
                        "model": model_name,
                        "feature_set": feature_set,
                        "strategy": method,
                        "round": round_index,
                        "measured_count": int(measured_k),
                        "new_measurements": int(measured_k - previous_k),
                        "pool_rows_remaining": int(len(discoverable_indices) - measured_k),
                        **metrics,
                    })
                    previous_k = measured_k

    rounds_path = output_path / "selection_baselines_rounds.csv"
    selections_path = output_path / "selection_baselines_selections.csv"
    summary_path_out = output_path / "selection_baselines_summary.csv"
    write_csv(pd.DataFrame(round_output_rows), rounds_path)
    write_csv(pd.DataFrame(selection_output_rows), selections_path)
    summary = summarize_rows(round_output_rows)
    write_csv(summary, summary_path_out)
    write_selection_plots(summary, output_path)

    return rounds_path, selections_path, summary_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model-independent search/selection methods for oracle-replay community discovery."
    )
    parser.add_argument("summary_path")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/selection_baselines")
    parser.add_argument("--target-species")
    parser.add_argument("--species-ids")
    parser.add_argument(
        "--models",
        default="ridge_pairwise,random_forest,hist_gradient_boosting",
        help="Models trained on the measured prefix to score landscape learning (never used to select).",
    )
    parser.add_argument(
        "--methods",
        default="greedy_forward,greedy_backward,hill_climb,simulated_annealing,genetic_algorithm,size_balanced",
        help="Model-independent search/selection methods to compare.",
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Measured-count grid step.")
    parser.add_argument("--rounds", type=int, help="Budget = rounds x batch-size; default measures the whole pool.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=parse_seed_count, default=5)
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument("--max-partners", type=int)
    parser.add_argument("--iterations", type=int, default=250, help="Simulated-annealing steps per restart.")
    parser.add_argument("--start-temperature", type=float, default=0.5)
    parser.add_argument("--cooling", type=float, default=0.98)
    parser.add_argument("--ga-population", type=int, default=24, help="Genetic-algorithm population size.")
    parser.add_argument("--ga-mutation", type=float, default=0.1, help="Genetic-algorithm per-bit mutation rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SelectionConfig(
        batch_size=args.batch_size,
        rounds=args.rounds,
        test_size=args.test_size,
        seed=args.seed,
        seeds=args.seeds,
        suppressor_fold=args.suppressor_fold,
        max_partners=args.max_partners,
        iterations=args.iterations,
        start_temperature=args.start_temperature,
        cooling=args.cooling,
        ga_population=args.ga_population,
        ga_mutation=args.ga_mutation,
    )
    rounds_path, selections_path, summary_path = run_selection_baselines(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        species_ids=parse_species_ids(args.species_ids),
        target_species=args.target_species,
        selected_models=parse_model_names(args.models),
        methods=parse_csv_list(args.methods),
        config=config,
    )
    print(f"Wrote selection-baseline rounds to {rounds_path}")
    print(f"Wrote selection records to {selections_path}")
    print(f"Wrote selection-baseline summary to {summary_path}")


if __name__ == "__main__":
    main()
