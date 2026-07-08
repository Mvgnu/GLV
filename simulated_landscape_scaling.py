#!/usr/bin/env python3
"""Sampled simulated-landscape scaling benchmarks for target suppression models."""

from __future__ import annotations

import argparse
import itertools
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
)
from active_learning import (
    train_model as train_active_model,
)
from lotka_volterra import generate_interaction_data, saturating_endpoint
from ml_benchmark import (
    GLVIdentityGNNRegressor,
    add_pairwise_features,
    build_regressor,
    load_dataset,
    model_configs,
    model_features,
    parse_model_names,
    regression_metrics,
    suppressor_classification_metrics,
    write_csv,
)


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
            partner_count = int(rng.choice(available_counts))
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
    real_summary_path: str | None,
    suppressor_fold: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    assay_noise_scale: float,
    target_scale_mapping: str,
    seed: int,
) -> pd.DataFrame:
    if not real_summary_path:
        summary = latent_summary.copy()
        summary["final_target_biomass"] = summary["latent_target_biomass"]
        summary["pathogen_signal_std"] = 0.0
        summary["replicate_count"] = 1
        summary["target_transform"] = "latent"
        return summary

    real_summary = pd.read_csv(real_summary_path)
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


def fill_remaining_explore_groups(
    measured: list[tuple[str, ...]],
    measured_set: set[tuple[str, ...]],
    partners: list[str],
    partner_counts: list[int],
    rng: np.random.Generator,
    budget: int,
    excluded: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    if len(measured) >= budget:
        return measured[:budget]
    remaining = sample_partner_groups(
        partners,
        partner_counts,
        rng,
        set(excluded) | measured_set,
        count=budget - len(measured),
    )
    measured.extend(remaining)
    return measured[:budget]


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
    remaining = list(range(len(candidates)))
    order = [remaining.pop(int(rng.integers(len(remaining))))]
    while remaining:
        selected_presence = presence[np.array(order, dtype=int)]
        best_position = 0
        best_distance = -1.0
        for position, candidate in enumerate(remaining):
            min_distance = float(np.abs(selected_presence - presence[candidate]).sum(axis=1).min())
            if min_distance > best_distance:
                best_distance = min_distance
                best_position = position
        order.append(remaining.pop(best_position))
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


def evaluate_group(
    group: tuple[str, ...],
    evaluator,
    measured: list[tuple[str, ...]],
    measured_set: set[tuple[str, ...]],
) -> float:
    if group not in measured_set:
        measured.append(group)
        measured_set.add(group)
    return evaluator.measure(group)


def greedy_forward_explore_groups(
    partners: list[str],
    partner_counts: list[int],
    evaluator,
    rng: np.random.Generator,
    budget: int,
) -> list[tuple[str, ...]]:
    measured: list[tuple[str, ...]] = []
    measured_set: set[tuple[str, ...]] = set()
    while len(measured) < budget:
        current = sample_partner_groups(partners, partner_counts, rng, measured_set, count=1)
        if not current:
            break
        current_group = current[0]
        evaluate_group(current_group, evaluator, measured, measured_set)
        while len(measured) < budget:
            add_neighbors = [
                tuple(sorted((*current_group, partner)))
                for partner in partners
                if partner not in current_group
                and len(current_group) + 1 in set(partner_counts)
            ]
            add_neighbors = [group for group in add_neighbors if group not in measured_set]
            if not add_neighbors:
                break
            scores = [evaluate_group(group, evaluator, measured, measured_set) for group in add_neighbors]
            best = add_neighbors[int(np.argmin(scores))]
            if evaluator.measure(best) >= evaluator.measure(current_group):
                break
            current_group = best
    return fill_remaining_explore_groups(measured, measured_set, partners, partner_counts, rng, budget, set())


def simulated_annealing_explore_groups(
    partners: list[str],
    partner_counts: list[int],
    evaluator,
    rng: np.random.Generator,
    budget: int,
) -> list[tuple[str, ...]]:
    measured: list[tuple[str, ...]] = []
    measured_set: set[tuple[str, ...]] = set()
    while len(measured) < budget:
        start = sample_partner_groups(partners, partner_counts, rng, measured_set, count=1)
        if not start:
            break
        current = start[0]
        current_score = evaluate_group(current, evaluator, measured, measured_set)
        temperature = 0.5
        for _iteration in range(250):
            if len(measured) >= budget:
                break
            neighbors = valid_neighbor_groups(current, partners, partner_counts)
            if not neighbors:
                break
            proposal = neighbors[int(rng.integers(len(neighbors)))]
            proposal_score = evaluate_group(proposal, evaluator, measured, measured_set)
            delta = proposal_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                current = proposal
                current_score = proposal_score
            temperature *= 0.98
    return fill_remaining_explore_groups(measured, measured_set, partners, partner_counts, rng, budget, set())


def genetic_algorithm_explore_groups(
    partners: list[str],
    partner_counts: list[int],
    evaluator,
    rng: np.random.Generator,
    budget: int,
) -> list[tuple[str, ...]]:
    measured: list[tuple[str, ...]] = []
    measured_set: set[tuple[str, ...]] = set()
    population = sample_partner_groups(partners, partner_counts, rng, set(), count=24)
    while len(measured) < budget and population:
        fitness = np.array([evaluate_group(group, evaluator, measured, measured_set) for group in population])
        elite = population[int(np.argmin(fitness))]
        offspring = [elite]
        while len(offspring) < 24:
            parent_a = population[int(rng.integers(len(population)))]
            parent_b = population[int(rng.integers(len(population)))]
            genes = sorted(set(parent_a) | set(parent_b))
            child = [gene for gene in genes if rng.random() < 0.5]
            for partner in partners:
                if rng.random() < 0.02:
                    if partner in child:
                        child.remove(partner)
                    else:
                        child.append(partner)
            if len(child) not in partner_counts:
                target_size = int(rng.choice(partner_counts))
                if len(child) > target_size:
                    child = list(rng.choice(child, size=target_size, replace=False))
                else:
                    additions = [partner for partner in partners if partner not in child]
                    needed = target_size - len(child)
                    if needed > 0 and additions:
                        child.extend(rng.choice(additions, size=min(needed, len(additions)), replace=False))
            group = tuple(sorted(child))
            if group in offspring:
                replacement = sample_partner_groups(partners, partner_counts, rng, set(offspring), count=1)
                if replacement:
                    group = replacement[0]
            offspring.append(group)
        population = offspring
    return fill_remaining_explore_groups(measured, measured_set, partners, partner_counts, rng, budget, set())


def explore_groups(
    method: str,
    partners: list[str],
    partner_counts: list[int],
    evaluator,
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
    if method == "greedy_forward":
        return greedy_forward_explore_groups(partners, partner_counts, evaluator, rng, budget)
    if method == "simulated_annealing":
        return simulated_annealing_explore_groups(partners, partner_counts, evaluator, rng, budget)
    if method == "genetic_algorithm":
        return genetic_algorithm_explore_groups(partners, partner_counts, evaluator, rng, budget)
    raise ValueError(f"Unknown strategy: {method}")


def simulate_partner_groups(
    interaction_data: pd.DataFrame,
    partner_groups: list[tuple[str, ...]],
    target_species: str,
    extinction_threshold: float,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
    real_summary_path: str | None,
    suppressor_fold: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    assay_noise_scale: float,
    target_scale_mapping: str,
    seed: int,
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
        real_summary_path,
        suppressor_fold,
        partner_count_effect_scale,
        partner_count_effect_center,
        partner_count_effect_width,
        assay_noise_scale,
        target_scale_mapping,
        seed,
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
        self.cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def measure(self, group: tuple[str, ...]) -> float:
        if group not in self.cache:
            self.cache[group] = simulate_partner_groups(
                self.interaction_data,
                [group],
                seed=stable_group_seed(self.seed, group),
                **self.simulation_kwargs,
            )
        return float(self.cache[group]["final_target_biomass"].iloc[0])

    def summary_for(self, groups: list[tuple[str, ...]]) -> pd.DataFrame:
        for group in groups:
            self.measure(group)
        return pd.concat([self.cache[group] for group in groups], ignore_index=True)


def build_evaluation_dataset(
    measured_summary: pd.DataFrame,
    audit_summary: pd.DataFrame,
    path: Path,
    target_species: str,
    species_ids: list[str],
) -> tuple[object, np.ndarray, np.ndarray, pd.DataFrame]:
    measured = measured_summary.copy()
    audit = audit_summary.copy()
    measured["pool"] = "measured"
    audit["pool"] = "audit"
    evaluation_summary = pd.concat([measured, audit], ignore_index=True)
    evaluation_summary.to_csv(path, index=False)
    dataset = load_dataset(str(path), target_species, species_ids)
    measured_indices = np.flatnonzero(evaluation_summary["pool"].eq("measured").to_numpy())
    audit_indices = np.flatnonzero(evaluation_summary["pool"].eq("audit").to_numpy())
    return dataset, measured_indices, audit_indices, evaluation_summary


def bayesian_iterative_groups(
    partners: list[str],
    partner_counts: list[int],
    evaluator: LazyCommunityEvaluator,
    audit_summary: pd.DataFrame,
    workspace_path: Path,
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
        dataset, measured_indices, _audit_indices, _evaluation_summary = build_evaluation_dataset(
            measured_summary,
            audit_summary,
            workspace_path,
            target_species,
            species_ids,
        )
        model, _features = train_active_model(
            dataset,
            "ridge_pairwise",
            "pairwise",
            measured_indices,
            seed + len(measured_groups),
        )
        candidate_groups = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            blocked | measured_set,
            count=proposal_candidate_size,
        )
        if not candidate_groups:
            break

        candidate_presence = presence_from_groups(candidate_groups, partners).astype(float)
        candidate_features, _feature_names = add_pairwise_features(candidate_presence, partners)
        predictions = model.predict(candidate_features)
        take = min(batch_size, budget - len(measured_groups), len(candidate_groups))
        order = np.argsort(predictions)[:take]
        for position in order:
            group = candidate_groups[int(position)]
            measured_groups.append(group)
            measured_set.add(group)

    return fill_remaining_explore_groups(
        measured_groups,
        measured_set,
        partners,
        partner_counts,
        rng,
        budget,
        blocked,
    )


def features_for_partner_groups(
    groups: list[tuple[str, ...]],
    partners: list[str],
    feature_set: str,
) -> np.ndarray:
    presence = presence_from_groups(groups, partners).astype(float)
    if feature_set == "pairwise":
        features, _feature_names = add_pairwise_features(presence, partners)
        return features
    return presence


def fit_model(dataset, model_name: str, feature_set: str, train_indices: np.ndarray, seed: int):
    features = model_features(dataset, feature_set)
    if model_name == "gnn":
        model = GLVIdentityGNNRegressor(
            partner_count=len(dataset.partner_ids),
            seed=seed,
        )
    else:
        model = build_regressor(model_name, seed)
    model.fit(features[train_indices], dataset.target_biomass[train_indices])
    return model, features


def optimizer_recommendations(
    optimizer: str,
    score_groups,
    partners: list[str],
    partner_counts: list[int],
    seed: int,
    top_k: int,
    proposal_candidate_size: int,
) -> tuple[list[tuple[str, ...]], np.ndarray, int]:
    rng = np.random.default_rng(seed)
    candidates = sample_partner_groups(
        partners,
        partner_counts,
        rng,
        set(),
        count=max(proposal_candidate_size, top_k),
    )

    if optimizer == "predicted_best":
        score_groups(candidates)
    elif optimizer == "greedy_forward":
        starts = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            set(),
            count=min(32, max(1, proposal_candidate_size // 20)),
        )
        candidates.extend(starts)
        for start in starts:
            current = start
            current_score = float(score_groups([current])[0])
            for _step in range(40):
                neighbors = valid_neighbor_groups(current, partners, partner_counts)
                if not neighbors:
                    break
                scores = score_groups(neighbors)
                best_index = int(np.argmin(scores))
                best_score = float(scores[best_index])
                candidates.append(neighbors[best_index])
                if best_score >= current_score:
                    break
                current = neighbors[best_index]
                current_score = best_score
    elif optimizer == "simulated_annealing":
        starts = sample_partner_groups(
            partners,
            partner_counts,
            rng,
            set(),
            count=12,
        )
        candidates.extend(starts)
        for start in starts:
            current = start
            current_score = float(score_groups([current])[0])
            temperature = 0.5
            for _step in range(120):
                neighbors = valid_neighbor_groups(current, partners, partner_counts)
                if not neighbors:
                    break
                proposal = neighbors[int(rng.integers(len(neighbors)))]
                proposal_score = float(score_groups([proposal])[0])
                candidates.append(proposal)
                delta = proposal_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                    current = proposal
                    current_score = proposal_score
                temperature *= 0.98
    elif optimizer == "genetic_algorithm":
        population = sample_partner_groups(partners, partner_counts, rng, set(), count=48)
        candidates.extend(population)
        for _generation in range(60):
            scores = score_groups(population)
            elite_indices = np.argsort(scores)[:8]
            offspring = [population[int(index)] for index in elite_indices]
            while len(offspring) < 48:
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
            candidates.extend(population)
    else:
        raise ValueError(f"Unknown phase2 optimizer: {optimizer}")

    unique_candidates = sorted(set(candidates))
    scores = score_groups(unique_candidates)
    order = np.argsort(scores)[:top_k]
    recommendations = [unique_candidates[int(index)] for index in order]
    return recommendations, scores[order], len(unique_candidates)


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
    output_path = Path(output_dir)
    inputs_path = output_path / "inputs"
    pools_path = output_path / "pools"
    inputs_path.mkdir(parents=True, exist_ok=True)
    pools_path.mkdir(parents=True, exist_ok=True)

    run_rows = []
    metric_rows = []
    phase2_rows = []
    model_setup = model_configs(models)
    max_species_count = max(species_counts)
    common_target_species = target_species or f"sp_{min(species_counts):03d}"

    for seed_index in range(seeds):
        universe_seed = base_seed + seed_index
        # One max-species universe keeps species-count comparisons nested.
        universe_interaction_data = generate_interaction_data(
            species_count=max_species_count,
            interaction_range=interaction_range,
            off_diagonal_min=off_diagonal_min,
            off_diagonal_max=off_diagonal_max,
            growth_rate=growth_rate,
            self_interaction=self_interaction,
            target_species=common_target_species,
            target_self_interaction=target_self_interaction,
            effect_prior_csv=effect_prior_csv,
            target_effect_scale=target_effect_scale,
            pair_effect_scale=pair_effect_scale,
            seed=universe_seed,
            interaction_generator=interaction_generator,
            carrying_capacity_min=carrying_capacity_min,
            carrying_capacity_max=carrying_capacity_max,
            hierarchy_strength=hierarchy_strength,
            hierarchy_noise=hierarchy_noise,
            target_interaction_scale=target_interaction_scale,
        )
        universe_path = inputs_path / f"universe_species_{max_species_count}_seed_{universe_seed}_interactions.csv"
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
            # Subset the shared universe instead of regenerating a new landscape.
            interaction_data = universe_interaction_data[
                universe_interaction_data["species_id"].isin(species_ids)
            ].copy()
            interaction_path = inputs_path / f"{run_id}_interactions.csv"
            interaction_data.to_csv(interaction_path, index=False)

            possible_by_count = {
                count: math.comb(len(partners), count)
                for count in valid_partner_counts
            }
            total_groups = sum(possible_by_count.values())
            audit_target = min(audit_size, max(1, int(total_groups * audit_fraction)))
            audit_rows_by_count = {}
            for count, possible_count in possible_by_count.items():
                rows = min(possible_count, int(audit_target * possible_count / total_groups))
                if rows > 0:
                    audit_rows_by_count[count] = rows
            if not audit_rows_by_count:
                largest_count = max(possible_by_count, key=possible_by_count.get)
                audit_rows_by_count[largest_count] = 1
            audit_path = pools_path / f"{run_id}_audit_pool.csv"
            if audit_path.exists():
                audit_summary = pd.read_csv(audit_path)
                audit_groups = partner_groups_from_summary(audit_summary, run_target_species)
            else:
                audit_groups = sample_partner_groups(
                    partners,
                    audit_rows_by_count,
                    rng,
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
                "real_summary_path": real_summary_path,
                "suppressor_fold": suppressor_fold,
                "partner_count_effect_scale": partner_count_effect_scale,
                "partner_count_effect_center": partner_count_effect_center,
                "partner_count_effect_width": partner_count_effect_width,
                "assay_noise_scale": assay_noise_scale,
                "target_scale_mapping": target_scale_mapping,
            }
            if not audit_path.exists():
                audit_summary = simulate_partner_groups(
                    interaction_data,
                    audit_groups,
                    seed=run_seed + 29,
                    **simulation_kwargs,
                )
                audit_summary["pool"] = "audit"
                audit_summary.to_csv(audit_path, index=False)
            suppressor_target_scale = "raw" if target_scale_mapping == "latent" else "log"

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
            })

            direct_evaluator = LazyCommunityEvaluator(
                interaction_data,
                run_target_species,
                simulation_kwargs,
                run_seed + 500_000,
            )
            direct_scores: dict[tuple[str, ...], float] = {}

            def score_direct_groups(groups: list[tuple[str, ...]]) -> np.ndarray:
                for group in groups:
                    if group not in direct_scores:
                        direct_scores[group] = direct_evaluator.measure(group)
                return np.array([direct_scores[group] for group in groups], dtype=float)

            direct_optimizer_results = {}
            for optimizer in phase2_optimizers:
                recommendation_groups, search_scores, evaluated_count = optimizer_recommendations(
                    optimizer,
                    score_direct_groups,
                    partners,
                    valid_partner_counts,
                    run_seed + 600_000 + sum(ord(char) for char in optimizer),
                    phase2_top_k,
                    proposal_candidate_size,
                )
                validated_summary = direct_evaluator.summary_for(recommendation_groups)
                validated_values = validated_summary["final_target_biomass"].to_numpy(dtype=float)
                best_position = int(np.argmin(validated_values))
                direct_optimizer_results[optimizer] = {
                    "optimizer": optimizer,
                    "recommended_count": int(len(recommendation_groups)),
                    "optimizer_evaluated_count": int(evaluated_count),
                    "best_search_score": float(np.min(search_scores)),
                    "best_validated_biomass": float(validated_values[best_position]),
                    "mean_validated_biomass": float(np.mean(validated_values)),
                    "best_recommended_community": validated_summary["community"].iloc[best_position],
                    "best_recommended_partner_count": int(validated_summary["partner_count"].iloc[best_position]),
                }

            for strategy in strategies:
                strategy_seed = run_seed + sum(ord(char) for char in strategy)
                strategy_rng = np.random.default_rng(strategy_seed)
                evaluator = direct_evaluator
                strategy_workspace = pools_path / f"{run_id}_{strategy}_evaluation.csv"
                final_measured_path = pools_path / f"{run_id}_{strategy}_measured.csv"
                if final_measured_path.exists():
                    final_measured_summary = pd.read_csv(final_measured_path)
                    ordered_groups = partner_groups_from_summary(final_measured_summary, run_target_species)
                elif strategy == "bayesian_optimization":
                    ordered_groups = bayesian_iterative_groups(
                        partners,
                        valid_partner_counts,
                        evaluator,
                        audit_summary,
                        strategy_workspace,
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
                    ordered_groups = explore_groups(
                        strategy,
                        partners,
                        valid_partner_counts,
                        evaluator,
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
                    measured_summary = final_measured_summary.iloc[:measured_count].copy()
                    evaluation_path = pools_path / f"{run_id}_{strategy}_{measured_count}_evaluation.csv"
                    dataset, measured_indices, audit_indices, _evaluation_summary = build_evaluation_dataset(
                        measured_summary,
                        audit_summary,
                        evaluation_path,
                        run_target_species,
                        species_ids,
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
                        in_band = lambda counts: (counts >= band.min_count) & (counts <= band.max_count)
                        audit_mask = in_band(audit_partner_counts)
                        if not np.any(audit_mask):
                            continue
                        train_mask = in_band(train_partner_counts)
                        if not np.any(train_mask):
                            continue
                        evaluation_slices.append((
                            band.display_label,
                            audit_indices[audit_mask],
                            np.flatnonzero(audit_mask),
                            float(np.min(dataset.target_biomass[audit_indices[audit_mask]])),
                            float(np.min(dataset.target_biomass[train_indices[train_mask]])),
                        ))

                    for optimizer, result in direct_optimizer_results.items():
                        phase2_rows.append({
                            "run_id": run_id,
                            "species_count": species_count,
                            "seed": run_seed,
                            "strategy": strategy,
                            "search_source": "simulator",
                            "model": "simulator_baseline",
                            "feature_set": "truth",
                            "measured_count": int(measured_count),
                            **result,
                        })

                    for model_name, feature_set in model_setup:
                        model, features = fit_model(
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
                            metric_rows.append(row)

                        surrogate_scores: dict[tuple[str, ...], float] = {}

                        def score_surrogate_groups(groups: list[tuple[str, ...]]) -> np.ndarray:
                            missing = [group for group in groups if group not in surrogate_scores]
                            if missing:
                                group_features = features_for_partner_groups(missing, partners, feature_set)
                                predictions = model.predict(group_features)
                                for group, prediction in zip(missing, predictions, strict=True):
                                    surrogate_scores[group] = float(prediction)
                            return np.array([surrogate_scores[group] for group in groups], dtype=float)

                        for optimizer in phase2_optimizers:
                            recommendation_groups, search_scores, evaluated_count = optimizer_recommendations(
                                optimizer,
                                score_surrogate_groups,
                                partners,
                                valid_partner_counts,
                                run_seed + measured_count + sum(ord(char) for char in optimizer),
                                phase2_top_k,
                                proposal_candidate_size,
                            )
                            validated_summary = evaluator.summary_for(recommendation_groups)
                            validated_values = validated_summary["final_target_biomass"].to_numpy(dtype=float)
                            best_position = int(np.argmin(validated_values))
                            best_validated = float(validated_values[best_position])
                            phase2_rows.append({
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

    runs = pd.DataFrame(run_rows)
    metrics = pd.DataFrame(metric_rows)
    phase2_metrics = pd.DataFrame(phase2_rows)
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
    parser.add_argument("--target-effect-scale", type=float, default=0.25)
    parser.add_argument("--pair-effect-scale", type=float, default=-0.5)
    # Disabled by default so community-size suppression is measured, not imposed.
    parser.add_argument("--partner-count-effect-scale", type=float, default=0.0)
    parser.add_argument("--partner-count-effect-center", type=float, default=5.5)
    parser.add_argument("--partner-count-effect-width", type=float, default=2.0)
    parser.add_argument("--assay-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--target-scale-mapping",
        choices=["zscore", "latent"],
        default="zscore",
    )
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument("--buffer-z", type=float, default=1.96)
    parser.add_argument("--extinction-threshold", type=float, default=1e-8)
    return parser.parse_args()


def optional_existing_path(path: str) -> str | None:
    return path if path and Path(path).exists() else None


def main() -> None:
    args = parse_args()
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
        real_summary_path=optional_existing_path(args.real_summary),
        effect_prior_csv=optional_existing_path(args.effect_prior_csv),
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
