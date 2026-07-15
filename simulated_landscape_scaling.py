#!/usr/bin/env python3
"""Sampled simulated-landscape scaling benchmarks for target suppression models."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_simulation_rates import (
    add_assay_observations,
    validate_assay_mapping,
)
from active_learning import (
    bayesian_optimization_statistics,
    train_model as train_active_model,
)
from lotka_volterra import generate_interaction_data, saturating_endpoint
from ml_benchmark import (
    add_pairwise_features,
    dataset_from_summary,
    model_configs,
    parse_model_names,
    regression_metrics,
    suppressor_classification_metrics,
    write_csv,
)
from simulation_assay_noise import AssayNoiseModel, fit_assay_noise_model


MAPPING_CALIBRATION_SIZE = 500
MAPPING_CALIBRATION_LANDSCAPES = 5


@dataclass(frozen=True)
class PartnerCountBand:
    label: str
    min_count: int
    max_count: int

    @property
    def display_label(self) -> str:
        if self.min_count == self.max_count:
            return f"{self.label} ({self.min_count})"
        return f"{self.label} ({self.min_count}-{self.max_count})"


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_partner_counts(value: str) -> list[int]:
    counts: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        # for ranges
        if "-" in item:
            start, end = item.split("-", maxsplit=1)
            counts.update(range(int(start), int(end) + 1))
        else:
            counts.add(int(item))
    return sorted(counts)


def parse_partner_count_bands(value: str) -> list[PartnerCountBand]:
    bands = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        label, raw_range = item.split(":", maxsplit=1)
        if "-" in raw_range:
            start, end = raw_range.split("-", maxsplit=1)
        else:
            start = end = raw_range
        bands.append(PartnerCountBand(label.strip(), int(start), int(end)))
    return bands


def sample_partner_groups(
    partners: list[str],
    partner_count_request: dict[int, int] | list[int],
    rng: np.random.Generator,
    excluded: set[tuple[str, ...]] | None = None,
    count: int | None = None,
) -> list[tuple[str, ...]]:
    # exclude audit set communities to avoid test/train overlap for model training/sampling, empty for exploration/phase 2
    excluded = excluded or set()
    groups: list[tuple[str, ...]] = []
    seen = set(excluded)
    # Dict path requests exact rows per size. List path samples uniformly from all
    # remaining communities, so sizes are weighted by their combination counts.
    if isinstance(partner_count_request, dict):
        rows_by_partner_count = partner_count_request
    else:
        rows_by_partner_count = {partner_count: 0 for partner_count in partner_count_request}
        available_by_partner_count = {
            partner_count: (
                math.comb(len(partners), partner_count)
                - sum(1 for group in seen if len(group) == partner_count)
            )
            for partner_count in partner_count_request
        }
        take = min(count, sum(available_by_partner_count.values()))
        while sum(rows_by_partner_count.values()) < take:
            available_counts = [
                partner_count
                for partner_count, available in available_by_partner_count.items()
                if rows_by_partner_count[partner_count] < available
            ]
            remaining = np.array([
                available_by_partner_count[partner_count] - rows_by_partner_count[partner_count]
                for partner_count in available_counts
            ], dtype=float)
            partner_count = int(rng.choice(available_counts, p=remaining / remaining.sum()))
            rows_by_partner_count[partner_count] += 1

    for partner_count, requested_rows in sorted(rows_by_partner_count.items()):
        possible_count = math.comb(len(partners), partner_count)
        seen_at_size = sum(1 for group in seen if len(group) == partner_count)
        take = min(requested_rows, possible_count - seen_at_size)
        if take <= 0:
            continue

        # if we need at least 25% of the remaining space, just enumerate otherwise sample randomly
        if possible_count - seen_at_size <= take * 4:
            candidates = [
                candidate
                for candidate in itertools.combinations(partners, partner_count)
                if candidate not in seen
            ]
            rng.shuffle(candidates)
            for group in candidates[:take]:
                seen.add(group)
                groups.append(group)
        else:
            selected_count = 0
            while selected_count < take:
                candidate = tuple(sorted(rng.choice(partners, size=partner_count, replace=False)))
                if candidate in seen:
                    continue
                seen.add(candidate)
                groups.append(candidate)
                selected_count += 1

    return groups


def simulate_communities(
    interaction_data: pd.DataFrame,
    communities: list[tuple[str, ...]],
    target_species: str,
    extinction_threshold: float,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
) -> pd.DataFrame:
    species_ids = sorted(interaction_data["species_id"].astype(str).tolist())
    data = interaction_data.set_index("species_id").loc[species_ids]
    growth_rates = data["growth_rate"].to_numpy(dtype=float)
    interaction_matrix = data.loc[species_ids, species_ids].to_numpy(dtype=float)
    index_by_species = {species: index for index, species in enumerate(species_ids)}
    rows = []

    for community in communities:
        indices = [index_by_species[species] for species in community]
        local_target_index = list(community).index(target_species)
        if interaction_response != "saturating":
            raise ValueError("landscape scaling requires --interaction-response saturating")
        final = saturating_endpoint(
            growth_rates[indices],
            interaction_matrix[np.ix_(indices, indices)],
            endpoint_initial_density,
            endpoint_max_time,
            saturation_pressure,
        )
        # Treat numerically tiny terminal densities as extinct after integration.
        final[final < extinction_threshold] = 0.0
        rows.append({
            "community": ";".join(community),
            "community_size": len(community),
            "partner_count": len(community) - 1,
            "target_species": target_species,
            "latent_target_biomass": float(final[local_target_index]),
        })

    return pd.DataFrame(rows)


def add_observed_targets(
    latent_summary: pd.DataFrame,
    real_summary: pd.DataFrame | None,
    noise_model: AssayNoiseModel | None,
    suppressor_fold: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    assay_noise_scale: float,
    target_scale_mapping: str,
    seed: int,
    mapping_reference_values: np.ndarray | None = None,
) -> pd.DataFrame:
    if real_summary is None:
        summary = latent_summary.copy()
        summary["final_target_biomass"] = summary["latent_target_biomass"]
        summary["pathogen_signal_std"] = 0.0
        summary["replicate_count"] = 1
        summary["target_transform"] = "latent"
        return summary

    return add_assay_observations(
        latent_summary,
        real_summary,
        suppressor_fold,
        partner_count_effect_scale,
        partner_count_effect_center,
        partner_count_effect_width,
        seed,
        assay_noise_scale,
        target_scale_mapping,
        mapping_reference_values,
        noise_model,
    )


def presence_from_groups(
    groups: list[tuple[str, ...]],
    partners: list[str],
) -> np.ndarray:
    index_by_partner = {partner: index for index, partner in enumerate(partners)}
    presence = np.zeros((len(groups), len(partners)), dtype=int)
    for row_index, group in enumerate(groups):
        for partner in group:
            presence[row_index, index_by_partner[partner]] = 1
    return presence


def size_balanced_explore_groups(
    partners: list[str],
    partner_counts: list[int],
    rng: np.random.Generator,
    budget: int,
    excluded: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    groups = []
    seen = set(excluded)
    while len(groups) < budget:
        progressed = False
        for partner_count in partner_counts:
            selected = sample_partner_groups(partners, {partner_count: 1}, rng, seen)
            if selected:
                group = selected[0]
                seen.add(group)
                groups.append(group)
                progressed = True
            if len(groups) >= budget:
                break
        if not progressed:
            break
    return groups


def max_diversity_explore_groups(
    partners: list[str],
    partner_counts: list[int],
    rng: np.random.Generator,
    budget: int,
    excluded: set[tuple[str, ...]],
    proposal_candidate_size: int,
) -> list[tuple[str, ...]]:
    candidates = sample_partner_groups(
        partners,
        partner_counts,
        rng,
        excluded,
        count=max(proposal_candidate_size, budget),
    )
    if not candidates:
        return []
    presence = presence_from_groups(candidates, partners)
    first = int(rng.integers(len(candidates)))
    order = [first]
    min_distances = np.abs(presence - presence[first]).sum(axis=1).astype(float)
    min_distances[first] = -1.0
    while len(order) < min(budget, len(candidates)):
        next_position = int(np.argmax(min_distances))
        order.append(next_position)
        # Only the newly selected row can lower each candidate's nearest distance.
        distances = np.abs(presence - presence[next_position]).sum(axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[np.array(order, dtype=int)] = -1.0
    return [candidates[position] for position in order[:budget]]


def valid_neighbor_groups(
    group: tuple[str, ...],
    partners: list[str],
    partner_counts: list[int],
) -> list[tuple[str, ...]]:
    valid_sizes = set(partner_counts)
    group_set = set(group)
    neighbors = []
    # Add one partner.
    for partner in partners:
        if partner not in group_set:
            candidate = tuple(sorted((*group, partner)))
            if len(candidate) in valid_sizes:
                neighbors.append(candidate)
    # Remove one partner.
    for partner in group:
        candidate = tuple(item for item in group if item != partner)
        if len(candidate) in valid_sizes:
            neighbors.append(candidate)
    # Swap one present partner for one absent partner.
    for old_partner in group:
        for new_partner in partners:
            if new_partner in group_set:
                continue
            candidate = tuple(sorted((*(item for item in group if item != old_partner), new_partner)))
            if len(candidate) in valid_sizes:
                neighbors.append(candidate)
    return sorted(set(neighbors))


def phase1_measurement_groups(
    method: str,
    partners: list[str],
    partner_counts: list[int],
    rng: np.random.Generator,
    budget: int,
    excluded: set[tuple[str, ...]],
    proposal_candidate_size: int,
) -> list[tuple[str, ...]]:
    if method == "random":
        return sample_partner_groups(partners, partner_counts, rng, excluded, count=budget)
    if method == "size_balanced":
        return size_balanced_explore_groups(partners, partner_counts, rng, budget, excluded)
    if method == "max_diversity":
        return max_diversity_explore_groups(
            partners,
            partner_counts,
            rng,
            budget,
            excluded,
            proposal_candidate_size,
        )
    raise ValueError(f"Unknown Phase 1 measurement strategy: {method}")


def simulate_partner_groups(
    interaction_data: pd.DataFrame,
    partner_groups: list[tuple[str, ...]],
    target_species: str,
    extinction_threshold: float,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
    real_summary: pd.DataFrame | None,
    noise_model: AssayNoiseModel | None,
    suppressor_fold: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    assay_noise_scale: float,
    target_scale_mapping: str,
    seed: int,
    mapping_reference_values: np.ndarray | None = None,
) -> pd.DataFrame:
    communities = [tuple(sorted((*group, target_species))) for group in partner_groups]
    latent = simulate_communities(
        interaction_data,
        communities,
        target_species,
        extinction_threshold,
        interaction_response,
        saturation_pressure,
        endpoint_initial_density,
        endpoint_max_time,
    )
    return add_observed_targets(
        latent,
        real_summary,
        noise_model,
        suppressor_fold,
        partner_count_effect_scale,
        partner_count_effect_center,
        partner_count_effect_width,
        assay_noise_scale,
        target_scale_mapping,
        seed,
        mapping_reference_values,
    )


def partner_groups_from_summary(summary: pd.DataFrame, target_species: str) -> list[tuple[str, ...]]:
    groups = []
    for community in summary["community"].astype(str):
        groups.append(tuple(
            species
            for species in community.split(";")
            if species and species != target_species
        ))
    return groups


def stable_group_seed(base_seed: int, group: tuple[str, ...]) -> int:
    offset = 0
    for item in group:
        for character in item:
            offset = (offset * 131 + ord(character)) % 1_000_003
    return base_seed + offset


class LazyCommunityEvaluator:
    def __init__(
        self,
        interaction_data: pd.DataFrame,
        target_species: str,
        simulation_kwargs: dict[str, object],
        seed: int,
    ) -> None:
        self.interaction_data = interaction_data
        self.target_species = target_species
        self.simulation_kwargs = simulation_kwargs
        self.seed = seed
        self.cache: dict[tuple[str, ...], dict[str, object]] = {}

    def measure(self, group: tuple[str, ...]) -> float:
        if group not in self.cache:
            summary = simulate_partner_groups(
                self.interaction_data,
                [group],
                seed=stable_group_seed(self.seed, group),
                **self.simulation_kwargs,
            )
            self.cache[group] = summary.iloc[0].to_dict()
        return float(self.cache[group]["final_target_biomass"])

    def summary_for(self, groups: list[tuple[str, ...]]) -> pd.DataFrame:
        for group in groups:
            self.measure(group)
        return pd.DataFrame([self.cache[group] for group in groups])


def bayesian_iterative_groups(
    partners: list[str],
    partner_counts: list[int],
    evaluator: LazyCommunityEvaluator,
    target_species: str,
    species_ids: list[str],
    seed: int,
    initial_size: int,
    batch_size: int,
    budget: int,
    excluded: set[tuple[str, ...]],
    proposal_candidate_size: int,
) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    blocked = set(excluded)
    initial_size = min(initial_size, budget)
    measured_groups = size_balanced_explore_groups(
        partners,
        partner_counts,
        rng,
        initial_size,
        blocked,
    )
    measured_set = set(measured_groups)

    while len(measured_groups) < budget:
        measured_summary = evaluator.summary_for(measured_groups)
        candidate_groups = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            blocked | measured_set,
            count=proposal_candidate_size,
        )
        if not candidate_groups:
            break

        candidate_summary = pd.DataFrame({
            "community": [
                ";".join(sorted((*group, target_species)))
                for group in candidate_groups
            ],
            "partner_count": [len(group) for group in candidate_groups],
            "target_species": target_species,
            "final_target_biomass": 0.0,
        })
        acquisition_summary = pd.concat(
            [measured_summary, candidate_summary],
            ignore_index=True,
        )
        dataset = dataset_from_summary(acquisition_summary, target_species, species_ids)
        measured_indices = np.arange(len(measured_summary), dtype=int)
        _mean, _uncertainty, acquisition_scores = bayesian_optimization_statistics(
            dataset,
            measured_indices,
            seed + len(measured_groups),
        )
        take = min(batch_size, budget - len(measured_groups), len(candidate_groups))
        order = np.argsort(acquisition_scores[len(measured_summary):])[:take]
        for position in order:
            group = candidate_groups[int(position)]
            measured_groups.append(group)
            measured_set.add(group)

    if len(measured_groups) < budget:
        measured_groups.extend(
            sample_partner_groups(
                partners,
                partner_counts,
                rng,
                blocked | measured_set,
                count=budget - len(measured_groups),
            )
        )
    return measured_groups[:budget]


def run_phase2_optimizer(
    optimizer: str,
    score_groups,
    partners: list[str],
    partner_counts: list[int],
    seed: int,
    top_k: int,
    proposal_candidate_size: int,
    measurement_budget: int | None = None,
) -> tuple[list[tuple[str, ...]], np.ndarray, int]:
    """Walk a simulator or surrogate landscape and return the top recommendations.

    A direct simulator walk may set ``measurement_budget``. Each distinct community
    scored then consumes one measurement; cached repeats are free. Surrogate walks leave
    it unset because evaluating another model prediction has no laboratory cost.
    """
    rng = np.random.default_rng(seed)
    scores_by_group: dict[tuple[str, ...], float] = {}

    def budget_exhausted() -> bool:
        return measurement_budget is not None and len(scores_by_group) >= measurement_budget

    def evaluate(groups: list[tuple[str, ...]]) -> np.ndarray:
        missing = list(dict.fromkeys(
            group for group in groups if group not in scores_by_group
        ))
        if measurement_budget is not None:
            remaining = measurement_budget - len(scores_by_group)
            missing = missing[:remaining]
        if missing:
            scores = score_groups(missing)
            for group, score in zip(missing, scores, strict=True):
                scores_by_group[group] = float(score)
        return np.array([scores_by_group.get(group, np.inf) for group in groups], dtype=float)

    if optimizer == "predicted_best":
        candidates = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            set(),
            count=max(proposal_candidate_size, top_k),
        )
        evaluate(candidates)
    elif optimizer == "greedy_forward":
        # Start at the smallest requested size, then only add partners while improving.
        minimum_size = min(partner_counts)
        start_count = 32 if measurement_budget is None else measurement_budget
        starts = sample_partner_groups(
            partners,
            {minimum_size: min(start_count, math.comb(len(partners), minimum_size))},
            rng,
            set(),
        )
        for start in starts:
            if budget_exhausted():
                break
            current = start
            current_score = float(evaluate([current])[0])
            while len(current) < max(partner_counts) and not budget_exhausted():
                neighbors = [
                    tuple(sorted((*current, partner)))
                    for partner in partners
                    if partner not in current and len(current) + 1 in partner_counts
                ]
                if not neighbors:
                    break
                rng.shuffle(neighbors)
                scores = evaluate(neighbors)
                best_index = int(np.argmin(scores))
                best_score = float(scores[best_index])
                if best_score >= current_score:
                    break
                current = neighbors[best_index]
                current_score = best_score
    elif optimizer == "simulated_annealing":
        # The neighbor set is complete; annealing samples one neighbor per bounded step.
        start_count = 12
        if measurement_budget is not None:
            start_count = max(12, math.ceil(measurement_budget / 120))
        starts = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            set(),
            count=start_count,
        )
        for start in starts:
            if budget_exhausted():
                break
            current = start
            current_score = float(evaluate([current])[0])
            temperature = max(abs(current_score) * 0.1, 1e-3)
            for _step in range(120):
                if budget_exhausted():
                    break
                neighbors = valid_neighbor_groups(current, partners, partner_counts)
                if not neighbors:
                    break
                proposal = neighbors[int(rng.integers(len(neighbors)))]
                proposal_score = float(evaluate([proposal])[0])
                delta = proposal_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                    current = proposal
                    current_score = proposal_score
                temperature *= 0.98
    elif optimizer == "genetic_algorithm":
        # Population and generation counts cap compute, not the per-community neighbor set.
        population = sample_partner_groups(partners, partner_counts, rng, set(), count=48)
        population_size = len(population)
        generation_count = 60
        if measurement_budget is not None:
            generation_count = max(60, 2 * math.ceil(measurement_budget / population_size))
        for _generation in range(generation_count):
            if budget_exhausted():
                break
            scores = evaluate(population)
            elite_indices = np.argsort(scores)[:min(8, population_size)]
            offspring = [population[int(index)] for index in elite_indices]
            while len(offspring) < population_size:
                parent_a = population[int(rng.integers(len(population)))]
                parent_b = population[int(rng.integers(len(population)))]
                child = [partner for partner in sorted(set(parent_a) | set(parent_b)) if rng.random() < 0.5]
                for partner in partners:
                    if rng.random() < 0.03:
                        if partner in child:
                            child.remove(partner)
                        else:
                            child.append(partner)
                target_size = int(rng.choice(partner_counts))
                if len(child) > target_size:
                    child = list(rng.choice(child, size=target_size, replace=False))
                elif len(child) < target_size:
                    additions = [partner for partner in partners if partner not in child]
                    child.extend(rng.choice(additions, size=target_size - len(child), replace=False))
                group = tuple(sorted(child))
                if group in offspring:
                    replacement = sample_partner_groups(partners, partner_counts, rng, set(offspring), count=1)
                    if replacement:
                        group = replacement[0]
                offspring.append(group)
            population = offspring
        evaluate(population)
    else:
        raise ValueError(f"Unknown phase2 optimizer: {optimizer}")

    # Random restarts spend any budget left after a walk stalls or exhausts its local moves.
    if measurement_budget is not None and not budget_exhausted():
        evaluate(sample_partner_groups(
            partners,
            partner_counts,
            rng,
            set(scores_by_group),
            count=measurement_budget - len(scores_by_group),
        ))

    evaluated_groups = list(scores_by_group)
    scores = np.array([scores_by_group[group] for group in evaluated_groups], dtype=float)
    order = np.argsort(scores)[:top_k]
    recommendations = [evaluated_groups[int(index)] for index in order]
    return recommendations, scores[order], len(evaluated_groups)


def bounded_metric_limits(metric: str) -> tuple[float, float] | None:
    if metric in {"suppressor_precision", "suppressor_class_recall", "suppressor_auprc"}:
        return (0.0, 1.02)
    return None


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plot_data = summary[summary["metric"].eq(metric)].copy()
    if "partner_count_band" in plot_data.columns:
        plot_data = plot_data[plot_data["partner_count_band"].eq("overall")]
    if plot_data.empty:
        return

    models = sorted(plot_data["model"].unique())
    strategies = sorted(plot_data["strategy"].unique())
    fig, axes = plt.subplots(
        len(models),
        len(strategies),
        figsize=(4.6 * len(strategies), max(3.2, 2.9 * len(models))),
        dpi=130,
        sharex=True,
        sharey=True,
    )
    if len(models) == 1:
        axes = np.array([axes])
    if len(strategies) == 1:
        axes = axes.reshape(len(models), 1)

    for row_index, model_name in enumerate(models):
        for column_index, strategy in enumerate(strategies):
            ax = axes[row_index, column_index]
            panel_data = plot_data[
                plot_data["model"].eq(model_name)
                & plot_data["strategy"].eq(strategy)
            ]
            grouped = panel_data.groupby(
                ["species_count", "measured_count"],
                as_index=False,
            )["mean"].mean()
            for species_count, species_group in grouped.groupby("species_count", sort=True):
                ax.plot(
                    species_group["measured_count"],
                    species_group["mean"],
                    marker="o",
                    linewidth=1.5,
                    label=f"{species_count} species",
                )

            if row_index == 0:
                ax.set_title(strategy)
            if column_index == 0:
                ax.set_ylabel(f"{model_name}\n{metric}")
            if row_index == len(models) - 1:
                ax.set_xlabel("measured communities")
            limits = bounded_metric_limits(metric)
            if limits:
                ax.set_ylim(*limits)
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def plot_discovery_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plot_data = summary[summary["metric"].eq(metric)].copy()
    if "partner_count_band" in plot_data.columns:
        plot_data = plot_data[plot_data["partner_count_band"].eq("overall")]
    if plot_data.empty:
        return

    strategies = sorted(plot_data["strategy"].unique())
    fig, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(4.6 * len(strategies), 3.8),
        dpi=130,
        sharex=True,
        sharey=True,
    )
    if len(strategies) == 1:
        axes = np.array([axes])

    for ax, strategy in zip(axes, strategies, strict=True):
        panel_data = plot_data[plot_data["strategy"].eq(strategy)]
        grouped = panel_data.groupby(
            ["species_count", "measured_count"],
            as_index=False,
        )["mean"].mean()
        for species_count, species_group in grouped.groupby("species_count", sort=True):
            ax.plot(
                species_group["measured_count"],
                species_group["mean"],
                marker="o",
                linewidth=1.5,
                label=f"{species_count} species",
            )
        ax.set_title(strategy)
        ax.set_xlabel("measured communities")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_partner_band(
    summary: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    plot_data = summary[
        summary["metric"].eq(metric)
        & ~summary["partner_count_band"].eq("overall")
    ].copy()
    if plot_data.empty:
        return

    for species_count in sorted(plot_data["species_count"].unique()):
        species_data = plot_data[plot_data["species_count"].eq(species_count)]
        models = sorted(species_data["model"].unique())
        strategies = sorted(species_data["strategy"].unique())
        fig, axes = plt.subplots(
            len(models),
            len(strategies),
            figsize=(4.6 * len(strategies), max(3.2, 2.9 * len(models))),
            dpi=130,
            sharex=True,
            sharey=True,
        )
        if len(models) == 1:
            axes = np.array([axes])
        if len(strategies) == 1:
            axes = axes.reshape(len(models), 1)

        for row_index, model_name in enumerate(models):
            for column_index, strategy in enumerate(strategies):
                ax = axes[row_index, column_index]
                panel_data = species_data[
                    species_data["model"].eq(model_name)
                    & species_data["strategy"].eq(strategy)
                ]
                grouped = panel_data.groupby(
                    ["partner_count_band", "measured_count"],
                    as_index=False,
                )["mean"].mean()
                for band, band_group in grouped.groupby("partner_count_band", sort=True):
                    ax.plot(
                        band_group["measured_count"],
                        band_group["mean"],
                        marker="o",
                        linewidth=1.5,
                        label=str(band),
                    )
                if row_index == 0:
                    ax.set_title(strategy)
                if column_index == 0:
                    ax.set_ylabel(f"{model_name}\n{metric}")
                if row_index == len(models) - 1:
                    ax.set_xlabel("measured communities")
                limits = bounded_metric_limits(metric)
                if limits:
                    ax.set_ylim(*limits)
                ax.grid(alpha=0.25)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output_dir / f"{metric}_by_partner_count_band_species_{species_count}.png")
        plt.close(fig)


def plot_discovery_by_partner_band(
    summary: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    plot_data = summary[
        summary["metric"].eq(metric)
        & ~summary["partner_count_band"].eq("overall")
    ].copy()
    if plot_data.empty:
        return

    for species_count in sorted(plot_data["species_count"].unique()):
        species_data = plot_data[plot_data["species_count"].eq(species_count)]
        strategies = sorted(species_data["strategy"].unique())
        fig, axes = plt.subplots(
            1,
            len(strategies),
            figsize=(4.6 * len(strategies), 3.8),
            dpi=130,
            sharex=True,
            sharey=True,
        )
        if len(strategies) == 1:
            axes = np.array([axes])

        for ax, strategy in zip(axes, strategies, strict=True):
            panel_data = species_data[species_data["strategy"].eq(strategy)]
            grouped = panel_data.groupby(
                ["partner_count_band", "measured_count"],
                as_index=False,
            )["mean"].mean()
            for band, band_group in grouped.groupby("partner_count_band", sort=True):
                ax.plot(
                    band_group["measured_count"],
                    band_group["mean"],
                    marker="o",
                    linewidth=1.5,
                    label=str(band),
                )
            ax.set_title(strategy)
            ax.set_xlabel("measured communities")
            ax.grid(alpha=0.25)

        axes[0].set_ylabel(metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.savefig(output_dir / f"{metric}_by_partner_count_band_species_{species_count}.png")
        plt.close(fig)


def metric_higher_is_better(metric: str) -> bool:
    return metric not in {
        "rmse",
        "mae",
        "best_audit_biomass",
        "best_measured_biomass",
        "best_audit_gap",
        "best_audit_gap_fraction",
        "best_validated_biomass",
    }


def best_model_by_band(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        metric_rows = summary[summary["metric"].eq(metric)].copy()
        if metric_rows.empty:
            continue
        grouped = metric_rows.groupby(
            ["species_count", "partner_count_band", "measured_count", "model"],
            as_index=False,
        )["mean"].mean()
        ascending = not metric_higher_is_better(metric)
        for group_key, group in grouped.groupby(
            ["species_count", "partner_count_band", "measured_count"],
            sort=True,
        ):
            finite = group[np.isfinite(group["mean"])]
            if finite.empty:
                continue
            winner = finite.sort_values("mean", ascending=ascending).iloc[0]
            rows.append({
                "species_count": group_key[0],
                "partner_count_band": group_key[1],
                "measured_count": group_key[2],
                "metric": metric,
                "best_model": winner["model"],
                "best_score": float(winner["mean"]),
            })
    return pd.DataFrame(rows)


def best_strategy_by_band(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        metric_rows = summary[summary["metric"].eq(metric)].copy()
        if metric_rows.empty:
            continue
        grouped = metric_rows.groupby(
            ["species_count", "partner_count_band", "measured_count", "strategy"],
            as_index=False,
        )["mean"].mean()
        ascending = not metric_higher_is_better(metric)
        for group_key, group in grouped.groupby(
            ["species_count", "partner_count_band", "measured_count"],
            sort=True,
        ):
            finite = group[np.isfinite(group["mean"])]
            if finite.empty:
                continue
            winner = finite.sort_values("mean", ascending=ascending).iloc[0]
            rows.append({
                "species_count": group_key[0],
                "partner_count_band": group_key[1],
                "measured_count": group_key[2],
                "metric": metric,
                "best_strategy": winner["strategy"],
                "best_score": float(winner["mean"]),
            })
    return pd.DataFrame(rows)


def plot_best_model_by_band(best_models: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_data = best_models[
        best_models["metric"].eq(metric)
        & ~best_models["partner_count_band"].eq("overall")
    ].copy()
    if plot_data.empty:
        return

    species_counts = sorted(plot_data["species_count"].unique())
    fig, axes = plt.subplots(
        1,
        len(species_counts),
        figsize=(4.8 * len(species_counts), 3.8),
        dpi=130,
        sharex=True,
        sharey=True,
    )
    if len(species_counts) == 1:
        axes = np.array([axes])

    for ax, species_count in zip(axes, species_counts, strict=True):
        species_data = plot_data[plot_data["species_count"].eq(species_count)]
        for band, band_group in species_data.groupby("partner_count_band", sort=True):
            ax.plot(
                band_group["measured_count"],
                band_group["best_score"],
                marker="o",
                linewidth=1.5,
                label=str(band),
            )
            last = band_group.sort_values("measured_count").iloc[-1]
            ax.annotate(
                str(last["best_model"]),
                (last["measured_count"], last["best_score"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
        ax.set_title(f"{species_count} species")
        ax.set_xlabel("measured communities")
        ax.grid(alpha=0.25)
        limits = bounded_metric_limits(metric)
        if limits:
            ax.set_ylim(*limits)

    axes[0].set_ylabel(f"best model {metric}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def plot_best_strategy_by_band(best_strategies: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_data = best_strategies[
        best_strategies["metric"].eq(metric)
        & ~best_strategies["partner_count_band"].eq("overall")
    ].copy()
    if plot_data.empty:
        return

    species_counts = sorted(plot_data["species_count"].unique())
    fig, axes = plt.subplots(
        1,
        len(species_counts),
        figsize=(4.8 * len(species_counts), 3.8),
        dpi=130,
        sharex=True,
        sharey=True,
    )
    if len(species_counts) == 1:
        axes = np.array([axes])

    for ax, species_count in zip(axes, species_counts, strict=True):
        species_data = plot_data[plot_data["species_count"].eq(species_count)]
        for band, band_group in species_data.groupby("partner_count_band", sort=True):
            ax.plot(
                band_group["measured_count"],
                band_group["best_score"],
                marker="o",
                linewidth=1.5,
                label=str(band),
            )
            last = band_group.sort_values("measured_count").iloc[-1]
            ax.annotate(
                str(last["best_strategy"]),
                (last["measured_count"], last["best_score"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
        ax.set_title(f"{species_count} species")
        ax.set_xlabel("measured communities")
        ax.grid(alpha=0.25)
        limits = bounded_metric_limits(metric)
        if limits:
            ax.set_ylim(*limits)

    axes[0].set_ylabel(f"best strategy {metric}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def strategy_model_performance(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        metric_rows = summary[
            summary["metric"].eq(metric)
            & summary["partner_count_band"].eq("overall")
        ].copy()
        if metric_rows.empty:
            continue
        grouped = metric_rows.groupby(
            ["strategy", "measured_count", "metric"],
            as_index=False,
        )["mean"].mean()
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)


def plot_strategy_model_performance(
    strategy_performance: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plot_data = strategy_performance[strategy_performance["metric"].eq(metric)].copy()
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    for strategy, group in plot_data.groupby("strategy", sort=True):
        ordered = group.sort_values("measured_count")
        ordered = ordered[np.isfinite(ordered["mean"])]
        if ordered.empty:
            continue
        ax.plot(
            ordered["measured_count"],
            ordered["mean"],
            marker="o",
            linewidth=1.5,
            label=str(strategy),
        )
    limits = bounded_metric_limits(metric)
    if limits:
        ax.set_ylim(*limits)
    ax.set_xlabel("measured communities")
    ax.set_ylabel(metric)
    ax.set_title(f"Exploration strategy -> downstream model {metric}")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "rmse",
        "mae",
        "r2",
        "spearman",
        "suppressor_precision",
        "suppressor_class_recall",
        "suppressor_auprc",
        "best_audit_biomass",
        "best_measured_biomass",
        "best_audit_gap",
        "best_audit_gap_fraction",
    ]
    rows = []
    group_columns = ["species_count", "strategy", "model", "partner_count_band", "measured_count"]
    for group_key, group in metrics.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, group_key, strict=True))
        for value_column in value_columns:
            values = group[value_column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            rows.append({
                **base,
                "metric": value_column,
                "mean": float(np.mean(finite_values)) if len(finite_values) else float("nan"),
                "std": float(np.std(finite_values, ddof=0)) if len(finite_values) else float("nan"),
                "replicates": int(len(values)),
            })
    return pd.DataFrame(rows)


def summarize_phase2_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "best_search_score",
        "best_validated_biomass",
        "mean_validated_biomass",
    ]
    rows = []
    group_columns = ["species_count", "strategy", "search_source", "model", "optimizer", "measured_count"]
    for group_key, group in metrics.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, group_key, strict=True))
        for value_column in value_columns:
            values = group[value_column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            rows.append({
                **base,
                "metric": value_column,
                "mean": float(np.mean(finite_values)) if len(finite_values) else float("nan"),
                "std": float(np.std(finite_values, ddof=0)) if len(finite_values) else float("nan"),
                "replicates": int(len(values)),
            })
    return pd.DataFrame(rows)


def plot_phase2_optimizer_metric(
    summary: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    plot_data = summary[summary["metric"].eq(metric)].copy()
    if plot_data.empty:
        return

    for species_count in sorted(plot_data["species_count"].unique()):
        species_data = plot_data[plot_data["species_count"].eq(species_count)]
        surrogate_data = species_data[species_data["search_source"].eq("surrogate")]
        baseline_data = species_data[species_data["search_source"].eq("simulator")]
        models = sorted(surrogate_data["model"].unique())
        strategies = sorted(surrogate_data["strategy"].unique())
        fig, axes = plt.subplots(
            len(models),
            len(strategies),
            figsize=(4.6 * len(strategies), max(3.2, 2.9 * len(models))),
            dpi=130,
            sharex=True,
            sharey=True,
        )
        if len(models) == 1:
            axes = np.array([axes])
        if len(strategies) == 1:
            axes = axes.reshape(len(models), 1)

        for row_index, model_name in enumerate(models):
            for column_index, strategy in enumerate(strategies):
                ax = axes[row_index, column_index]
                panel_data = surrogate_data[
                    surrogate_data["model"].eq(model_name)
                    & surrogate_data["strategy"].eq(strategy)
                ]
                grouped = panel_data.groupby(
                    ["optimizer", "measured_count"],
                    as_index=False,
                )["mean"].mean()
                for optimizer, optimizer_group in grouped.groupby("optimizer", sort=True):
                    ax.plot(
                        optimizer_group["measured_count"],
                        optimizer_group["mean"],
                        marker="o",
                        linewidth=1.5,
                        label=str(optimizer),
                    )
                for optimizer, baseline_group in baseline_data.groupby("optimizer", sort=True):
                    baseline_group = baseline_group.sort_values("measured_count")
                    ax.plot(
                        baseline_group["measured_count"],
                        baseline_group["mean"],
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{optimizer} direct",
                    )
                if row_index == 0:
                    ax.set_title(strategy)
                if column_index == 0:
                    ax.set_ylabel(f"{model_name}\n{metric}")
                if row_index == len(models) - 1:
                    ax.set_xlabel("measured communities")
                ax.grid(alpha=0.25)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output_dir / f"phase2_{metric}_species_{species_count}.png")
        plt.close(fig)


def run_simulated_scaling(
    output_dir: str,
    species_counts: list[int],
    partner_counts: list[int],
    proposal_candidate_size: int,
    audit_size: int,
    audit_fraction: float,
    budgets: list[int],
    batch_size: int,
    partner_count_bands: list[PartnerCountBand],
    phase2_optimizers: list[str],
    phase2_top_k: int,
    seeds: int,
    base_seed: int,
    target_species: str | None,
    models: list[str] | None,
    strategies: list[str],
    real_summary_path: str | None,
    effect_prior_csv: str | None,
    interaction_range: float,
    off_diagonal_min: float,
    off_diagonal_max: float,
    growth_rate: float,
    self_interaction: float,
    interaction_generator: str,
    carrying_capacity_min: float,
    carrying_capacity_max: float,
    hierarchy_strength: float,
    hierarchy_noise: float,
    target_interaction_scale: float,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
    target_self_interaction: float,
    target_effect_scale: float,
    pair_effect_scale: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    assay_noise_scale: float,
    target_scale_mapping: str,
    suppressor_fold: float,
    buffer_z: float,
    extinction_threshold: float,
) -> tuple[Path, Path, Path]:
    validate_assay_mapping(
        real_summary_path is not None,
        target_scale_mapping,
        assay_noise_scale,
    )
    model_setup = model_configs(models)
    real_summary = pd.read_csv(real_summary_path) if real_summary_path else None
    noise_model = fit_assay_noise_model(real_summary)[0] if real_summary is not None else None

    output_path = Path(output_dir)
    inputs_path = output_path / "inputs"
    pools_path = output_path / "pools"
    output_path.mkdir(parents=True, exist_ok=True)

    input_paths = {}
    for label, path in {
        "real_summary": real_summary_path,
        "effect_prior": effect_prior_csv,
    }.items():
        if path:
            input_paths[label] = str(Path(path).resolve())

    run_config = {
        "species_counts": species_counts,
        "partner_counts": partner_counts,
        "proposal_candidate_size": proposal_candidate_size,
        "audit_size": audit_size,
        "audit_fraction": audit_fraction,
        "budgets": budgets,
        "batch_size": batch_size,
        "partner_count_bands": [
            {"label": band.label, "min_count": band.min_count, "max_count": band.max_count}
            for band in partner_count_bands
        ],
        "phase2_optimizers": phase2_optimizers,
        "phase2_top_k": phase2_top_k,
        "seeds": seeds,
        "base_seed": base_seed,
        "target_species": target_species,
        "models": models,
        "strategies": strategies,
        "input_paths": input_paths,
        "interaction_range": interaction_range,
        "off_diagonal_min": off_diagonal_min,
        "off_diagonal_max": off_diagonal_max,
        "growth_rate": growth_rate,
        "self_interaction": self_interaction,
        "interaction_generator": interaction_generator,
        "carrying_capacity_min": carrying_capacity_min,
        "carrying_capacity_max": carrying_capacity_max,
        "hierarchy_strength": hierarchy_strength,
        "hierarchy_noise": hierarchy_noise,
        "target_interaction_scale": target_interaction_scale,
        "interaction_response": interaction_response,
        "saturation_pressure": saturation_pressure,
        "endpoint_initial_density": endpoint_initial_density,
        "endpoint_max_time": endpoint_max_time,
        "target_self_interaction": target_self_interaction,
        "target_effect_scale": target_effect_scale,
        "pair_effect_scale": pair_effect_scale,
        "partner_count_effect_scale": partner_count_effect_scale,
        "partner_count_effect_center": partner_count_effect_center,
        "partner_count_effect_width": partner_count_effect_width,
        "assay_noise_scale": assay_noise_scale,
        "target_scale_mapping": target_scale_mapping,
        "suppressor_fold": suppressor_fold,
        "buffer_z": buffer_z,
        "extinction_threshold": extinction_threshold,
        "mapping_calibration_size": MAPPING_CALIBRATION_SIZE,
        "mapping_calibration_landscapes": MAPPING_CALIBRATION_LANDSCAPES,
    }
    manifest_path = output_path / "run_config.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    inputs_path.mkdir(parents=True, exist_ok=True)
    pools_path.mkdir(parents=True, exist_ok=True)

    run_rows = []
    metric_checkpoint_paths = []
    phase2_checkpoint_paths = []
    max_species_count = max(species_counts)
    common_target_species = target_species or f"sp_{min(species_counts):03d}"
    interaction_generation_kwargs = {
        "interaction_range": interaction_range,
        "off_diagonal_min": off_diagonal_min,
        "off_diagonal_max": off_diagonal_max,
        "growth_rate": growth_rate,
        "self_interaction": self_interaction,
        "target_species": common_target_species,
        "target_self_interaction": target_self_interaction,
        "effect_prior_csv": effect_prior_csv,
        "target_effect_scale": target_effect_scale,
        "pair_effect_scale": pair_effect_scale,
        "interaction_generator": interaction_generator,
        "carrying_capacity_min": carrying_capacity_min,
        "carrying_capacity_max": carrying_capacity_max,
        "hierarchy_strength": hierarchy_strength,
        "hierarchy_noise": hierarchy_noise,
        "target_interaction_scale": target_interaction_scale,
    }

    mapping_reference_values = None
    mapping_calibration_path = None
    if real_summary is not None and target_scale_mapping != "latent":
        calibration_species = [
            f"sp_{index + 1:03d}" for index in range(max_species_count)
        ]
        calibration_partners = [
            species for species in calibration_species if species != common_target_species
        ]
        calibration_partner_counts = [
            count for count in partner_counts if 0 < count <= len(calibration_partners)
        ]
        calibration_group_count = min(
            MAPPING_CALIBRATION_SIZE,
            audit_size,
            sum(
                math.comb(len(calibration_partners), count)
                for count in calibration_partner_counts
            ),
        )
        mapping_calibration_path = (
            inputs_path
            / (
                f"mapping_species_{max_species_count}_seed_{base_seed + 1_000_000}_"
                f"{MAPPING_CALIBRATION_LANDSCAPES}"
                "_landscapes_latent.csv"
            )
        )
        if mapping_calibration_path.exists():
            mapping_calibration = pd.read_csv(mapping_calibration_path)
        else:
            calibration_summaries = []
            for calibration_index in range(MAPPING_CALIBRATION_LANDSCAPES):
                calibration_seed = base_seed + 1_000_000 + calibration_index
                calibration_interaction_path = inputs_path / (
                    f"mapping_species_{max_species_count}_seed_{calibration_seed}"
                    "_interactions.csv"
                )
                if calibration_interaction_path.exists():
                    calibration_interaction_data = pd.read_csv(
                        calibration_interaction_path
                    )
                else:
                    calibration_interaction_data = generate_interaction_data(
                        species_count=max_species_count,
                        seed=calibration_seed,
                        **interaction_generation_kwargs,
                    )
                    calibration_interaction_data.to_csv(
                        calibration_interaction_path,
                        index=False,
                    )

                calibration_groups = sample_partner_groups(
                    calibration_partners,
                    calibration_partner_counts,
                    np.random.default_rng(calibration_seed),
                    count=calibration_group_count,
                )
                calibration_communities = [
                    tuple(sorted((*group, common_target_species)))
                    for group in calibration_groups
                ]
                calibration_summaries.append(simulate_communities(
                    calibration_interaction_data,
                    calibration_communities,
                    common_target_species,
                    extinction_threshold,
                    interaction_response,
                    saturation_pressure,
                    endpoint_initial_density,
                    endpoint_max_time,
                ))
            mapping_calibration = pd.concat(calibration_summaries, ignore_index=True)
            mapping_calibration.to_csv(mapping_calibration_path, index=False)
        mapping_reference_values = mapping_calibration[
            "latent_target_biomass"
        ].to_numpy(dtype=float)

    for seed_index in range(seeds):
        universe_seed = base_seed + seed_index
        universe_path = inputs_path / f"universe_species_{max_species_count}_seed_{universe_seed}_interactions.csv"
        if universe_path.exists():
            universe_interaction_data = pd.read_csv(universe_path)
        else:
            # One max-species universe keeps species-count comparisons nested.
            universe_interaction_data = generate_interaction_data(
                species_count=max_species_count,
                seed=universe_seed,
                **interaction_generation_kwargs,
            )
            universe_interaction_data.to_csv(universe_path, index=False)

        for species_count in species_counts:
            run_target_species = common_target_species
            species_ids = [f"sp_{index + 1:03d}" for index in range(species_count)]
            if run_target_species not in species_ids:
                raise ValueError(f"target species {run_target_species} is not in {species_count} species")
            partners = [species for species in species_ids if species != run_target_species]
            valid_partner_counts = [count for count in partner_counts if 0 < count <= len(partners)]
            if not valid_partner_counts:
                raise ValueError(f"No valid partner counts for {species_count} species")

            run_seed = base_seed + species_count * 1000 + seed_index
            rng = np.random.default_rng(run_seed)
            run_id = f"species_{species_count}_seed_{run_seed}"
            interaction_path = inputs_path / f"{run_id}_interactions.csv"
            if interaction_path.exists():
                interaction_data = pd.read_csv(interaction_path)
            else:
                # Subset the shared universe instead of regenerating a new landscape.
                interaction_data = universe_interaction_data[
                    universe_interaction_data["species_id"].isin(species_ids)
                ].copy()
                interaction_data.to_csv(interaction_path, index=False)

            possible_by_count = {
                count: math.comb(len(partners), count)
                for count in valid_partner_counts
            }
            total_groups = sum(possible_by_count.values())
            audit_target = min(audit_size, max(1, int(total_groups * audit_fraction)))
            audit_path = pools_path / f"{run_id}_audit_pool.csv"
            if audit_path.exists():
                audit_summary = pd.read_csv(audit_path)
                audit_groups = partner_groups_from_summary(audit_summary, run_target_species)
            else:
                audit_groups = sample_partner_groups(
                    partners,
                    valid_partner_counts,
                    rng,
                    count=audit_target,
                )
            audit_group_set = set(audit_groups)
            search_space_rows = max(0, total_groups - len(audit_group_set))
            max_budget = min(max(budgets), search_space_rows)
            if max_budget <= 0:
                raise ValueError(
                    f"No explorable communities remain for {run_id}; reduce --audit-size "
                    "or request more partner-count combinations."
                )
            initial_size = min(budgets)
            simulation_kwargs = {
                "target_species": run_target_species,
                "extinction_threshold": extinction_threshold,
                "interaction_response": interaction_response,
                "saturation_pressure": saturation_pressure,
                "endpoint_initial_density": endpoint_initial_density,
                "endpoint_max_time": endpoint_max_time,
                "real_summary": real_summary,
                "noise_model": noise_model,
                "suppressor_fold": suppressor_fold,
                "partner_count_effect_scale": partner_count_effect_scale,
                "partner_count_effect_center": partner_count_effect_center,
                "partner_count_effect_width": partner_count_effect_width,
                "assay_noise_scale": assay_noise_scale,
                "target_scale_mapping": target_scale_mapping,
                "mapping_reference_values": mapping_reference_values,
            }
            if not audit_path.exists():
                audit_communities = [tuple(sorted((*group, run_target_species))) for group in audit_groups]
                audit_latent = simulate_communities(
                    interaction_data,
                    audit_communities,
                    run_target_species,
                    extinction_threshold,
                    interaction_response,
                    saturation_pressure,
                    endpoint_initial_density,
                    endpoint_max_time,
                )
                audit_summary = add_observed_targets(
                    audit_latent,
                    real_summary,
                    noise_model,
                    suppressor_fold,
                    partner_count_effect_scale,
                    partner_count_effect_center,
                    partner_count_effect_width,
                    assay_noise_scale,
                    target_scale_mapping,
                    run_seed + 29,
                    mapping_reference_values,
                )
                audit_summary["pool"] = "audit"
                audit_summary.to_csv(audit_path, index=False)
            suppressor_target_scale = "raw"
            if noise_model is not None and target_scale_mapping != "latent":
                suppressor_target_scale = "log" if "log" in noise_model.target_scale else "raw"

            run_rows.append({
                "run_id": run_id,
                "species_count": species_count,
                "target_species": run_target_species,
                "seed": run_seed,
                "universe_seed": universe_seed,
                "universe_interaction_path": str(universe_path),
                "interaction_generator": interaction_generator,
                "carrying_capacity_min": float(carrying_capacity_min),
                "carrying_capacity_max": float(carrying_capacity_max),
                "hierarchy_strength": float(hierarchy_strength),
                "hierarchy_noise": float(hierarchy_noise),
                "target_interaction_scale": float(target_interaction_scale),
                "interaction_response": interaction_response,
                "saturation_pressure": float(saturation_pressure),
                "endpoint_initial_density": float(endpoint_initial_density),
                "endpoint_max_time": float(endpoint_max_time),
                "assay_noise_scale": float(assay_noise_scale),
                "target_scale_mapping": target_scale_mapping,
                "partner_counts": ",".join(str(count) for count in valid_partner_counts),
                "total_candidate_rows": int(total_groups),
                "requested_audit_rows": int(audit_size),
                "audit_fraction": float(audit_fraction),
                "candidate_space_rows": int(search_space_rows),
                "proposal_candidate_size": int(proposal_candidate_size),
                "audit_rows": int(len(audit_summary)),
                "interaction_path": str(interaction_path),
                "audit_pool_path": str(audit_path),
                "mapping_calibration_path": (
                    str(mapping_calibration_path) if mapping_calibration_path else ""
                ),
            })

            observed_evaluator = LazyCommunityEvaluator(
                interaction_data,
                run_target_species,
                simulation_kwargs,
                run_seed + 500_000,
            )
            truth_evaluator = LazyCommunityEvaluator(
                interaction_data,
                run_target_species,
                {**simulation_kwargs, "assay_noise_scale": 0.0},
                run_seed + 700_000,
            )

            def score_direct_groups(groups: list[tuple[str, ...]]) -> np.ndarray:
                return np.array([observed_evaluator.measure(group) for group in groups], dtype=float)

            direct_budgets = [budget for budget in budgets if budget <= max_budget]
            if not direct_budgets:
                direct_budgets = [max_budget]
            for measured_count in direct_budgets:
                direct_phase2_path = pools_path / f"{run_id}_direct_phase2_{measured_count}.csv"
                if direct_phase2_path.exists():
                    direct_optimizer_results = pd.read_csv(direct_phase2_path).to_dict("records")
                else:
                    direct_optimizer_results = []
                    for optimizer in phase2_optimizers:
                        recommendation_groups, search_scores, evaluated_count = run_phase2_optimizer(
                            optimizer,
                            score_direct_groups,
                            partners,
                            valid_partner_counts,
                            run_seed + 600_000 + sum(ord(char) for char in optimizer),
                            phase2_top_k,
                            proposal_candidate_size,
                            measurement_budget=measured_count,
                        )
                        if evaluated_count != measured_count:
                            raise RuntimeError(
                                f"{optimizer} used {evaluated_count} of {measured_count} measurements"
                            )
                        validated_summary = truth_evaluator.summary_for(recommendation_groups)
                        validated_values = validated_summary["final_target_biomass"].to_numpy(dtype=float)
                        best_position = int(np.argmin(validated_values))
                        direct_optimizer_results.append({
                            "optimizer": optimizer,
                            "recommended_count": int(len(recommendation_groups)),
                            "optimizer_evaluated_count": int(evaluated_count),
                            "best_search_score": float(np.min(search_scores)),
                            "best_validated_biomass": float(validated_values[best_position]),
                            "mean_validated_biomass": float(np.mean(validated_values)),
                            "best_recommended_community": validated_summary["community"].iloc[best_position],
                            "best_recommended_partner_count": int(
                                validated_summary["partner_count"].iloc[best_position]
                            ),
                        })
                direct_rows = [
                    {
                        "run_id": run_id,
                        "species_count": species_count,
                        "seed": run_seed,
                        "strategy": "direct",
                        "search_source": "simulator",
                        "model": "simulator_baseline",
                        "feature_set": "truth",
                        "measured_count": int(measured_count),
                        **result,
                    }
                    for result in direct_optimizer_results
                ]
                # Complete older result-only checkpoints once, then leave them untouched.
                if "run_id" not in direct_optimizer_results[0]:
                    pd.DataFrame(direct_rows).to_csv(direct_phase2_path, index=False)
                phase2_checkpoint_paths.append(direct_phase2_path)

            for strategy in strategies:
                strategy_seed = run_seed + sum(ord(char) for char in strategy)
                strategy_rng = np.random.default_rng(strategy_seed)
                evaluator = observed_evaluator
                final_measured_path = pools_path / f"{run_id}_{strategy}_measured.csv"
                if final_measured_path.exists():
                    final_measured_summary = pd.read_csv(final_measured_path)
                    ordered_groups = partner_groups_from_summary(final_measured_summary, run_target_species)
                elif strategy == "bayesian_optimization":
                    ordered_groups = bayesian_iterative_groups(
                        partners,
                        valid_partner_counts,
                        evaluator,
                        run_target_species,
                        species_ids,
                        strategy_seed,
                        initial_size,
                        batch_size,
                        max_budget,
                        audit_group_set,
                        proposal_candidate_size,
                    )
                    final_measured_summary = evaluator.summary_for(ordered_groups)
                else:
                    ordered_groups = phase1_measurement_groups(
                        strategy,
                        partners,
                        valid_partner_counts,
                        strategy_rng,
                        max_budget,
                        audit_group_set,
                        proposal_candidate_size,
                    )
                    final_measured_summary = evaluator.summary_for(ordered_groups)
                final_measured_summary["pool"] = "measured"
                if not final_measured_path.exists():
                    final_measured_summary.to_csv(final_measured_path, index=False)
                valid_budgets = [budget for budget in budgets if budget <= len(ordered_groups)]
                if len(ordered_groups) and not valid_budgets:
                    valid_budgets = [len(ordered_groups)]

                for measured_count in valid_budgets:
                    metrics_checkpoint_path = (
                        pools_path / f"{run_id}_{strategy}_{measured_count}_metrics.csv"
                    )
                    phase2_checkpoint_path = (
                        pools_path / f"{run_id}_{strategy}_{measured_count}_phase2.csv"
                    )
                    metric_checkpoint_paths.append(metrics_checkpoint_path)
                    phase2_checkpoint_paths.append(phase2_checkpoint_path)
                    if metrics_checkpoint_path.exists() and phase2_checkpoint_path.exists():
                        continue

                    budget_metric_rows = []
                    budget_phase2_rows = []
                    measured_summary = final_measured_summary.iloc[:measured_count].copy()
                    evaluation_summary = pd.concat(
                        [measured_summary, audit_summary],
                        ignore_index=True,
                    )
                    dataset = dataset_from_summary(
                        evaluation_summary,
                        run_target_species,
                        species_ids,
                    )
                    measured_indices = np.arange(len(measured_summary), dtype=int)
                    audit_indices = np.arange(
                        len(measured_summary),
                        len(evaluation_summary),
                        dtype=int,
                    )
                    train_indices = measured_indices
                    audit_median = float(np.median(dataset.target_biomass[audit_indices]))
                    audit_partner_counts = dataset.partner_counts[audit_indices]
                    train_partner_counts = dataset.partner_counts[train_indices]
                    evaluation_slices = [(
                        "overall",
                        audit_indices,
                        np.arange(len(audit_indices), dtype=int),
                        float(np.min(dataset.target_biomass[audit_indices])),
                        float(np.min(dataset.target_biomass[train_indices])),
                    )]
                    for band in partner_count_bands:
                        audit_mask = (
                            (audit_partner_counts >= band.min_count)
                            & (audit_partner_counts <= band.max_count)
                        )
                        if not np.any(audit_mask):
                            continue
                        train_mask = (
                            (train_partner_counts >= band.min_count)
                            & (train_partner_counts <= band.max_count)
                        )
                        best_measured = float("nan")
                        if np.any(train_mask):
                            best_measured = float(
                                np.min(dataset.target_biomass[train_indices[train_mask]])
                            )
                        evaluation_slices.append((
                            band.display_label,
                            audit_indices[audit_mask],
                            np.flatnonzero(audit_mask),
                            float(np.min(dataset.target_biomass[audit_indices[audit_mask]])),
                            best_measured,
                        ))

                    for model_name, feature_set in model_setup:
                        model, features = train_active_model(
                            dataset,
                            model_name,
                            feature_set,
                            train_indices,
                            run_seed + measured_count,
                        )
                        full_predictions = model.predict(features[audit_indices])
                        row_base = {
                            "run_id": run_id,
                            "species_count": species_count,
                            "seed": run_seed,
                            "strategy": strategy,
                            "model": model_name,
                            "feature_set": feature_set,
                            "measured_count": int(measured_count),
                        }
                        for (
                            band_label,
                            band_audit_indices,
                            prediction_positions,
                            best_audit,
                            best_measured,
                        ) in evaluation_slices:
                            best_gap = float("nan")
                            if np.isfinite(best_measured):
                                best_gap = max(0.0, best_measured - best_audit)
                            predictions = full_predictions[prediction_positions]
                            row = {
                                **row_base,
                                "partner_count_band": band_label,
                                "audit_rows": int(len(band_audit_indices)),
                                "best_audit_biomass": best_audit,
                                "best_measured_biomass": best_measured,
                                "best_audit_gap": best_gap,
                                "best_audit_gap_fraction": best_gap / max(abs(best_audit), 1e-12),
                            }
                            row.update(regression_metrics(dataset.target_biomass[band_audit_indices], predictions))
                            row.update(suppressor_classification_metrics(
                                dataset.target_biomass[band_audit_indices],
                                predictions,
                                dataset.target_se[band_audit_indices],
                                audit_median,
                                suppressor_fold,
                                buffer_z,
                                suppressor_target_scale,
                            ))
                            budget_metric_rows.append(row)

                        surrogate_scores: dict[tuple[str, ...], float] = {}

                        def score_surrogate_groups(groups: list[tuple[str, ...]]) -> np.ndarray:
                            missing = [group for group in groups if group not in surrogate_scores]
                            if missing:
                                group_features = presence_from_groups(missing, partners).astype(float)
                                if feature_set == "pairwise":
                                    group_features = add_pairwise_features(
                                        group_features,
                                        partners,
                                    )[0]
                                predictions = model.predict(group_features)
                                for group, prediction in zip(missing, predictions, strict=True):
                                    surrogate_scores[group] = float(prediction)
                            return np.array([surrogate_scores[group] for group in groups], dtype=float)

                        for optimizer in phase2_optimizers:
                            recommendation_groups, search_scores, evaluated_count = run_phase2_optimizer(
                                optimizer,
                                score_surrogate_groups,
                                partners,
                                valid_partner_counts,
                                run_seed + 600_000 + sum(ord(char) for char in optimizer),
                                phase2_top_k,
                                proposal_candidate_size,
                            )
                            validated_summary = truth_evaluator.summary_for(recommendation_groups)
                            validated_values = validated_summary["final_target_biomass"].to_numpy(dtype=float)
                            best_position = int(np.argmin(validated_values))
                            best_validated = float(validated_values[best_position])
                            budget_phase2_rows.append({
                                **row_base,
                                "search_source": "surrogate",
                                "optimizer": optimizer,
                                "recommended_count": int(len(recommendation_groups)),
                                "optimizer_evaluated_count": int(evaluated_count),
                                "best_search_score": float(np.min(search_scores)),
                                "best_validated_biomass": best_validated,
                                "mean_validated_biomass": float(np.mean(validated_values)),
                                "best_recommended_community": validated_summary["community"].iloc[best_position],
                                "best_recommended_partner_count": int(validated_summary["partner_count"].iloc[best_position]),
                            })
                        del model, features, full_predictions, score_surrogate_groups, surrogate_scores
                    pd.DataFrame(budget_metric_rows).to_csv(metrics_checkpoint_path, index=False)
                    pd.DataFrame(budget_phase2_rows).to_csv(phase2_checkpoint_path, index=False)
                    del (
                        budget_metric_rows,
                        budget_phase2_rows,
                        dataset,
                        measured_summary,
                        evaluation_summary,
                    )
                    gc.collect()

            del observed_evaluator, truth_evaluator, evaluator, score_direct_groups
            gc.collect()

    runs = pd.DataFrame(run_rows)
    metrics = pd.concat(
        (pd.read_csv(path) for path in metric_checkpoint_paths),
        ignore_index=True,
    )
    phase2_metrics = pd.concat(
        (pd.read_csv(path) for path in phase2_checkpoint_paths),
        ignore_index=True,
    )
    summary = summarize_metrics(metrics)
    phase2_summary = summarize_phase2_metrics(phase2_metrics)
    model_winners = best_model_by_band(
        summary,
        ["suppressor_auprc", "spearman", "rmse"],
    )
    strategy_winners = best_strategy_by_band(
        summary,
        ["best_measured_biomass", "suppressor_auprc", "spearman"],
    )
    strategy_performance = strategy_model_performance(
        summary,
        ["suppressor_auprc", "spearman", "rmse"],
    )

    runs_path = output_path / "simulated_scaling_runs.csv"
    metrics_path = output_path / "simulated_scaling_metrics.csv"
    summary_path = output_path / "simulated_scaling_summary.csv"
    model_winners_path = output_path / "best_model_by_band.csv"
    strategy_winners_path = output_path / "best_strategy_by_band.csv"
    strategy_performance_path = output_path / "strategy_model_performance.csv"
    phase2_metrics_path = output_path / "phase2_optimizer_metrics.csv"
    phase2_summary_path = output_path / "phase2_optimizer_summary.csv"
    runs.to_csv(runs_path, index=False)
    write_csv(metrics, metrics_path)
    write_csv(summary, summary_path)
    write_csv(model_winners, model_winners_path)
    write_csv(strategy_winners, strategy_winners_path)
    write_csv(strategy_performance, strategy_performance_path)
    write_csv(phase2_metrics, phase2_metrics_path)
    write_csv(phase2_summary, phase2_summary_path)

    for metric in [
        "suppressor_auprc",
        "suppressor_precision",
        "suppressor_class_recall",
        "spearman",
        "rmse",
    ]:
        plot_metric(summary, metric, output_path / f"{metric}_by_species_count.png")
        plot_metric_by_partner_band(summary, metric, output_path)
    plot_discovery_metric(
        summary,
        "best_measured_biomass",
        output_path / "best_measured_biomass_by_species_count.png",
    )
    plot_discovery_by_partner_band(summary, "best_measured_biomass", output_path)
    for metric in ["suppressor_auprc", "spearman", "rmse"]:
        plot_best_model_by_band(
            model_winners,
            metric,
            output_path / f"best_model_by_partner_count_band_{metric}.png",
        )
    for metric in ["best_measured_biomass", "suppressor_auprc", "spearman"]:
        plot_best_strategy_by_band(
            strategy_winners,
            metric,
            output_path / f"best_strategy_by_partner_count_band_{metric}.png",
        )
    for metric in ["suppressor_auprc", "spearman", "rmse"]:
        plot_strategy_model_performance(
            strategy_performance,
            metric,
            output_path / f"strategy_model_performance_{metric}.png",
        )
    for metric in ["best_validated_biomass"]:
        plot_phase2_optimizer_metric(phase2_summary, metric, output_path)

    return runs_path, metrics_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark surrogate models on sampled simulated landscapes."
    )
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/scaling_laws_simulated")
    parser.add_argument("--species-counts", default="12,16,20,24,28")
    parser.add_argument("--partner-counts", default="3-18")
    parser.add_argument(
        "--partner-count-bands",
        default="small:3-5,medium:6-10,large:11-15,very_large:16-18",
    )
    parser.add_argument("--proposal-candidate-size", type=int, default=2000)
    parser.add_argument("--audit-size", type=int, default=2000)
    parser.add_argument("--audit-fraction", type=float, default=0.25)
    parser.add_argument(
        "--budgets",
        default="50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-species", default="sp_012")
    parser.add_argument("--models", default="ridge_pairwise,random_forest,hist_gradient_boosting")
    parser.add_argument(
        "--strategies",
        default="random,size_balanced,max_diversity,bayesian_optimization",
    )
    parser.add_argument(
        "--phase2-optimizers",
        default="predicted_best,greedy_forward,simulated_annealing,genetic_algorithm",
    )
    parser.add_argument("--phase2-top-k", type=int, default=5)
    parser.add_argument("--real-summary", default="GLV_ML/outputs/real_world/log/rw_summary.csv")
    parser.add_argument(
        "--effect-prior-csv",
        default="GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv",
    )
    parser.add_argument("--interaction-range", type=float, default=1.0)
    parser.add_argument("--off-diagonal-min", type=float, default=-0.5)
    parser.add_argument("--off-diagonal-max", type=float, default=0.2)
    parser.add_argument("--growth-rate", type=float, default=1.0)
    parser.add_argument("--self-interaction", type=float, default=-1.0)
    parser.add_argument(
        "--interaction-generator",
        choices=["legacy", "hierarchical"],
        default="legacy",
    )
    parser.add_argument("--carrying-capacity-min", type=float, default=1.0)
    parser.add_argument("--carrying-capacity-max", type=float, default=1.0)
    parser.add_argument("--hierarchy-strength", type=float, default=0.0)
    parser.add_argument("--hierarchy-noise", type=float, default=0.0)
    parser.add_argument("--target-interaction-scale", type=float, default=1.0)
    parser.add_argument(
        "--interaction-response",
        choices=["saturating"],
        default="saturating",
    )
    parser.add_argument("--saturation-pressure", type=float, default=1.0)
    parser.add_argument("--endpoint-initial-density", type=float, default=0.5)
    parser.add_argument("--endpoint-max-time", type=float, default=500.0)
    parser.add_argument("--target-self-interaction", type=float, default=-1.0)
    parser.add_argument("--target-effect-scale", type=float, default=0.3)
    parser.add_argument("--pair-effect-scale", type=float, default=2.5)
    # Disabled by default so community-size suppression is measured, not imposed.
    parser.add_argument("--partner-count-effect-scale", type=float, default=0.0)
    parser.add_argument("--partner-count-effect-center", type=float, default=5.5)
    parser.add_argument("--partner-count-effect-width", type=float, default=2.0)
    parser.add_argument("--assay-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--target-scale-mapping",
        choices=["zscore", "quantile", "latent"],
        default="quantile",
    )
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument("--buffer-z", type=float, default=1.96)
    parser.add_argument("--extinction-threshold", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_summary_path = args.real_summary.strip() or None
    effect_prior_csv = args.effect_prior_csv.strip() or None
    if real_summary_path and not Path(real_summary_path).exists():
        raise FileNotFoundError(real_summary_path)
    if effect_prior_csv and not Path(effect_prior_csv).exists():
        raise FileNotFoundError(effect_prior_csv)
    runs_path, metrics_path, summary_path = run_simulated_scaling(
        output_dir=args.output_dir,
        species_counts=parse_int_grid(args.species_counts),
        partner_counts=parse_partner_counts(args.partner_counts),
        proposal_candidate_size=args.proposal_candidate_size,
        audit_size=args.audit_size,
        audit_fraction=args.audit_fraction,
        budgets=parse_int_grid(args.budgets),
        batch_size=args.batch_size,
        partner_count_bands=parse_partner_count_bands(args.partner_count_bands),
        phase2_optimizers=[item.strip() for item in args.phase2_optimizers.split(",") if item.strip()],
        phase2_top_k=args.phase2_top_k,
        seeds=args.seeds,
        base_seed=args.seed,
        target_species=args.target_species,
        models=parse_model_names(args.models),
        strategies=[item.strip() for item in args.strategies.split(",") if item.strip()],
        real_summary_path=real_summary_path,
        effect_prior_csv=effect_prior_csv,
        interaction_range=args.interaction_range,
        off_diagonal_min=args.off_diagonal_min,
        off_diagonal_max=args.off_diagonal_max,
        growth_rate=args.growth_rate,
        self_interaction=args.self_interaction,
        interaction_generator=args.interaction_generator,
        carrying_capacity_min=args.carrying_capacity_min,
        carrying_capacity_max=args.carrying_capacity_max,
        hierarchy_strength=args.hierarchy_strength,
        hierarchy_noise=args.hierarchy_noise,
        target_interaction_scale=args.target_interaction_scale,
        interaction_response=args.interaction_response,
        saturation_pressure=args.saturation_pressure,
        endpoint_initial_density=args.endpoint_initial_density,
        endpoint_max_time=args.endpoint_max_time,
        target_self_interaction=args.target_self_interaction,
        target_effect_scale=args.target_effect_scale,
        pair_effect_scale=args.pair_effect_scale,
        partner_count_effect_scale=args.partner_count_effect_scale,
        partner_count_effect_center=args.partner_count_effect_center,
        partner_count_effect_width=args.partner_count_effect_width,
        assay_noise_scale=args.assay_noise_scale,
        target_scale_mapping=args.target_scale_mapping,
        suppressor_fold=args.suppressor_fold,
        buffer_z=args.buffer_z,
        extinction_threshold=args.extinction_threshold,
    )
    print(f"Wrote simulated scaling runs to {runs_path}")
    print(f"Wrote simulated scaling metrics to {metrics_path}")
    print(f"Wrote simulated scaling summary to {summary_path}")


if __name__ == "__main__":
    main()
