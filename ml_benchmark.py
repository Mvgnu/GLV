#!/usr/bin/env python3
"""Benchmark sklearn regressors on target-species GLV biomass outputs."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass(frozen=True)
class TargetBiomassDataset:
    target_species: str
    partner_ids: list[str]
    feature_names: list[str]
    communities: list[str]
    presence: np.ndarray
    pairwise_features: np.ndarray
    target_biomass: np.ndarray
    partner_counts: np.ndarray
    # Standard error of each community's target biomass (0 when no replicate info).
    target_se: np.ndarray


@dataclass(frozen=True)
class SplitDataset:
    train_indices: np.ndarray
    test_indices: np.ndarray


def parse_species_ids(species_ids: str | None) -> list[str] | None:
    if species_ids is None:
        return None

    return [species.strip() for species in species_ids.split(",") if species.strip()]


def parse_community(community: str) -> list[str]:
    return [species for species in str(community).split(";") if species]


def infer_target_species(summary: pd.DataFrame, target_species: str | None) -> str:
    # Prefer the CLI target, otherwise read the single target recorded by simulation.
    if target_species:
        return target_species

    target_values = summary.get("target_species")
    if target_values is None:
        raise ValueError("--target-species is required when summary has no target_species column")

    targets = sorted({str(value) for value in target_values.dropna().unique() if str(value)})
    if len(targets) != 1:
        raise ValueError("--target-species is required when summary contains multiple targets")

    return targets[0]


def infer_partner_ids(
    summary: pd.DataFrame,
    parsed_communities: list[list[str]],
    target_species: str,
    species_ids: list[str] | None,
) -> list[str]:
    # Model inputs are partner identities, target is excluded.
    if species_ids:
        return [species for species in species_ids if species != target_species]

    final_species = [
        column.removeprefix("final_")
        for column in summary.columns
        if column.startswith("final_") and column != "final_target_biomass"
    ]
    if final_species:
        return sorted(species for species in final_species if species != target_species)

    return sorted({
        species
        for community in parsed_communities
        for species in community
        if species != target_species
    })


def add_pairwise_features(
    presence: np.ndarray,
    partner_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return main plus pairwise partner-presence features."""
    feature_blocks = [presence]
    feature_names = list(partner_ids)

    partner_count = presence.shape[1]
    for first_index in range(partner_count):
        for second_index in range(first_index + 1, partner_count):
            feature_blocks.append(
                (presence[:, first_index] * presence[:, second_index])[:, None]
            )
            feature_names.append(
                f"{partner_ids[first_index]}:{partner_ids[second_index]}"
            )

    return np.column_stack(feature_blocks), feature_names


def dataset_from_summary(
    summary: pd.DataFrame,
    target_species: str | None,
    species_ids: list[str] | None,
) -> TargetBiomassDataset:
    """Convert target-species simulation summaries into model-ready arrays."""
    # Parse communities once, then derive the fixed partner features
    target_species = infer_target_species(summary, target_species)
    communities = summary["community"].astype(str).tolist()
    parsed_communities = [parse_community(community) for community in communities]
    partner_ids = infer_partner_ids(
        summary,
        parsed_communities,
        target_species,
        species_ids,
    )
    partner_index = {species: index for index, species in enumerate(partner_ids)}

    target_column = "final_target_biomass"
    if target_column not in summary.columns:
        target_column = f"final_{target_species}"
    if target_column not in summary.columns:
        raise ValueError(
            "summary must contain final_target_biomass or final_<target_species>"
        )

    # Multi-hot encode introduced partners while leaving the target out.
    presence = np.zeros((len(summary), len(partner_ids)), dtype=float)
    for row_index, species_group in enumerate(parsed_communities):
        for species in species_group:
            if species != target_species:
                presence[row_index, partner_index[species]] = 1.0

    if "partner_count" in summary.columns:
        partner_counts = summary["partner_count"].to_numpy(dtype=int)
    else:
        partner_counts = presence.sum(axis=1).astype(int)

    # Standard error of each community's target biomass from replicate scatter.
    # Real plate data carries pathogen_signal_std + replicate_count; noiseless
    # simulation summaries lack them, so the SE (and the noise buffer) is 0 there.
    if "pathogen_signal_std" in summary.columns and "replicate_count" in summary.columns:
        replicate_count = summary["replicate_count"].to_numpy(dtype=float)
        target_se = summary["pathogen_signal_std"].to_numpy(dtype=float) / np.sqrt(
            np.maximum(replicate_count, 1.0)
        )
        target_se = np.nan_to_num(target_se, nan=0.0)
    else:
        target_se = np.zeros(len(summary), dtype=float)

    # Pairwise features expose partner interactions to linear baselines.
    pairwise_features, feature_names = add_pairwise_features(presence, partner_ids)

    return TargetBiomassDataset(
        target_species=target_species,
        partner_ids=partner_ids,
        feature_names=feature_names,
        communities=communities,
        presence=presence,
        pairwise_features=pairwise_features,
        target_biomass=summary[target_column].to_numpy(dtype=float),
        partner_counts=partner_counts,
        target_se=target_se,
    )


def load_dataset(
    summary_path: str,
    target_species: str | None,
    species_ids: list[str] | None,
) -> TargetBiomassDataset:
    """Load target-species simulation summaries into model-ready arrays."""
    return dataset_from_summary(pd.read_csv(summary_path), target_species, species_ids)


def split_dataset(dataset: TargetBiomassDataset, test_size: float, seed: int) -> SplitDataset:
    """Create a deterministic split stratified by partner count."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    test_index_groups = []
    train_index_groups = []

    # Split within each partner count so large size classes do not dominate the split.
    for partner_count in np.unique(dataset.partner_counts):
        size_indices = np.flatnonzero(dataset.partner_counts == partner_count)
        size_indices = rng.permutation(size_indices)
        if len(size_indices) == 1:
            size_test_count = 0
        else:
            size_test_count = max(1, int(round(len(size_indices) * test_size)))
            size_test_count = min(size_test_count, len(size_indices) - 1)

        test_index_groups.append(size_indices[:size_test_count])
        train_index_groups.append(size_indices[size_test_count:])

    return SplitDataset(
        train_indices=rng.permutation(np.concatenate(train_index_groups)),
        test_indices=rng.permutation(np.concatenate(test_index_groups)),
    )


def parse_train_sizes(train_sizes: str) -> list[int | str]:
    parsed_sizes = []
    for raw_size in train_sizes.split(","):
        raw_size = raw_size.strip()
        if not raw_size:
            continue
        parsed_sizes.append("all" if raw_size == "all" else int(raw_size))

    return parsed_sizes


def model_features(
    dataset: TargetBiomassDataset,
    feature_set: str,
) -> np.ndarray:
    if feature_set == "pairwise":
        return dataset.pairwise_features

    return dataset.presence


def _build_glv_gnn_module(node_dim: int, hidden: int, layers: int):
    import torch
    import torch.nn as nn

    class GLVCommunityGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
            self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.neighbor_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.head = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )

        def forward(self, node_features, mask, target_mask):
            mask_e = mask.unsqueeze(-1)
            h = self.node_encoder(node_features) * mask_e
            for self_layer, neighbor_layer in zip(self.self_layers, self.neighbor_layers):
                node_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                total = (h * mask_e).sum(dim=1, keepdim=True)
                neighbor_sum = total - h
                neighbor_count = (node_count - 1.0).clamp(min=1.0).unsqueeze(-1)
                neighbor_mean = neighbor_sum / neighbor_count
                h = torch.relu(self_layer(h) + neighbor_layer(neighbor_mean)) * mask_e

            node_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pool = (h * mask_e).sum(dim=1) / node_count
            neg_inf = torch.finfo(h.dtype).min
            max_pool = torch.where(mask_e.bool(), h, torch.full_like(h, neg_inf)).max(dim=1).values
            target_embedding = (h * target_mask.unsqueeze(-1)).sum(dim=1)
            return self.head(torch.cat([target_embedding, mean_pool, max_pool], dim=-1)).squeeze(-1)

    return GLVCommunityGNN()


class GLVIdentityGNNRegressor:
    """Target-aware community GNN over species identity nodes."""

    def __init__(
        self,
        partner_count: int,
        seed: int,
        hidden: int = 64,
        layers: int = 2,
        epochs: int = 80,
        batch_size: int = 256,
        learning_rate: float = 1e-2,
    ) -> None:
        self.partner_count = partner_count
        self.seed = seed
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.node_dim = partner_count + 2
        self.y_mean = 0.0
        self.y_std = 1.0
        self.module = None

    def _batch_tensors(self, presence: np.ndarray):
        import torch

        row_count = presence.shape[0]
        node_counts = presence.sum(axis=1).astype(int) + 1
        max_nodes = int(node_counts.max())
        node_features = np.zeros((row_count, max_nodes, self.node_dim), dtype=np.float32)
        mask = np.zeros((row_count, max_nodes), dtype=np.float32)
        target_mask = np.zeros((row_count, max_nodes), dtype=np.float32)

        for row_index, partner_presence in enumerate(presence):
            # Node 0 is always the target; remaining nodes are present partners.
            node_features[row_index, 0, 0] = 1.0
            node_features[row_index, 0, -1] = 1.0
            mask[row_index, 0] = 1.0
            target_mask[row_index, 0] = 1.0
            write_index = 1
            for partner_index, is_present in enumerate(partner_presence):
                if is_present:
                    node_features[row_index, write_index, partner_index + 1] = 1.0
                    mask[row_index, write_index] = 1.0
                    write_index += 1

        return (
            torch.from_numpy(node_features),
            torch.from_numpy(mask),
            torch.from_numpy(target_mask),
        )

    def fit(self, presence: np.ndarray, target_biomass: np.ndarray) -> "GLVIdentityGNNRegressor":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.module = _build_glv_gnn_module(self.node_dim, self.hidden, self.layers)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        targets = np.asarray(target_biomass, dtype=np.float32)
        self.y_mean = float(targets.mean())
        self.y_std = float(targets.std()) or 1.0
        y = torch.from_numpy((targets - self.y_mean) / self.y_std)
        rng = np.random.default_rng(self.seed)

        self.module.train()
        for _epoch in range(self.epochs):
            order = rng.permutation(len(presence))
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                node_features, mask, target_mask = self._batch_tensors(presence[indices])
                optimizer.zero_grad()
                prediction = self.module(node_features, mask, target_mask)
                loss = loss_fn(prediction, y[indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5.0)
                optimizer.step()

        return self

    def predict(self, presence: np.ndarray) -> np.ndarray:
        import torch

        self.module.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(presence), self.batch_size):
                batch = presence[start:start + self.batch_size]
                node_features, mask, target_mask = self._batch_tensors(batch)
                prediction = self.module(node_features, mask, target_mask)
                predictions.append((prediction * self.y_std + self.y_mean).cpu().numpy())

        return np.concatenate(predictions) if predictions else np.zeros(0)


def build_regressor(model_name: str, seed: int):
    """Create the sklearn regressor for one training fold."""
    # Baseline records how much signal exists beyond the target mean.
    if model_name == "mean_baseline":
        return DummyRegressor(strategy="mean")

    if model_name in {"ridge_main", "ridge_pairwise"}:
        return Ridge(alpha=1.0)

    # Tree models capture nonlinear partner effects.
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            # Serial fitting avoids parallel memory peaks across long model sweeps.
            n_jobs=1,
        )

    if model_name == "hist_gradient_boosting":
        # Boosted trees fit residuals sequentially, useful for nonlinear partner effects.
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=seed,
        )

    raise ValueError(f"Unknown model: {model_name}")


def model_configs(model_names: list[str] | None = None) -> list[tuple[str, str]]:
    configs = [
        ("mean_baseline", "main"),
        ("ridge_main", "main"),
        ("ridge_pairwise", "pairwise"),
        ("random_forest", "pairwise"),
        ("hist_gradient_boosting", "pairwise"),
        ("gnn", "graph"),
    ]
    if model_names is None:
        return configs

    known_models = {model_name for model_name, _feature_set in configs}
    unknown_models = sorted(set(model_names) - known_models)
    if unknown_models:
        raise ValueError(f"Unknown model(s): {', '.join(unknown_models)}")

    return [
        (model_name, feature_set)
        for model_name, feature_set in configs
        if model_name in model_names
    ]


def parse_model_names(model_names: str | None) -> list[str] | None:
    if model_names is None:
        return None

    return [model_name.strip() for model_name in model_names.split(",") if model_name.strip()]


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Constant predictions have undefined rank correlation; report no rank signal.
    true_ranks = pd.Series(y_true).rank().to_numpy(dtype=float)
    predicted_ranks = pd.Series(y_pred).rank().to_numpy(dtype=float)
    if np.std(true_ranks) == 0 or np.std(predicted_ranks) == 0:
        return 0.0

    correlation = np.corrcoef(true_ranks, predicted_ranks)[0, 1]
    return 0.0 if pd.isna(correlation) else float(correlation)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Point-prediction quality plus Spearman (global ordering, robust to the tail noise).
    r2 = float("nan") if len(y_true) < 2 else float(r2_score(y_true, y_pred))
    return {
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "spearman": spearman_correlation(y_true, y_pred),
    }


def suppressor_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_se: np.ndarray,
    median_biomass: float,
    suppressor_fold: float,
    buffer_z: float,
    suppressor_target_scale: str,
) -> dict[str, float]:
    """Score suppressive-vs-not calls the way a real screen would experience them.

    A community is suppressive when its target biomass is at least `suppressor_fold`-fold
    below the global median; the model calls
    "suppressive" when it predicts below that cutoff. Every call counts: we do NOT drop
    communities near the boundary, because prospectively you cannot see which calls are
    borderline -- you validate them and pay for the misses, so hiding them would inflate
    the reported precision. Reported:

      suppressor_precision           operational precision (= 1 - FDR): of all called
                                     communities, the share truly below the cutoff.
      ambiguous_call_fraction        share of the model's calls that land within
                                     `buffer_z` SEs of the cutoff -- a transparency flag
                                     for how fragile the precision is.

    `buffer_z` only feeds the transparency flag; it never excludes data from scoring.
    """
    if suppressor_target_scale == "raw":
        cutoff = median_biomass / suppressor_fold
    else:
        cutoff = median_biomass - float(np.log(suppressor_fold))
    called = y_pred < cutoff
    true_suppressor = y_true < cutoff
    called_count = int(called.sum())
    true_suppressor_count = int(true_suppressor.sum())
    true_positives = int((called & true_suppressor).sum())

    precision = float(true_positives / called_count) if called_count else float("nan")
    recall = (
        float(true_positives / true_suppressor_count)
        if true_suppressor_count
        else float("nan")
    )
    fdr = float(1.0 - precision) if called_count else float("nan")

    if called_count:
        ambiguous_call_fraction = float(
            np.mean(np.abs(y_true[called] - cutoff) < buffer_z * y_se[called])
        )
    else:
        ambiguous_call_fraction = float("nan")

    # Summary over test communities (point-estimate labels).
    if 0 < true_suppressor_count < len(y_true):
        auprc = float(average_precision_score(true_suppressor.astype(int), -y_pred))
    else:
        auprc = float("nan")

    return {
        "suppressor_cutoff": float(cutoff),
        "true_suppressor_count": true_suppressor_count,
        "called_suppressor_count": called_count,
        "suppressor_precision": precision,
        "suppressor_fdr": fdr,
        "suppressor_class_recall": recall,
        "ambiguous_call_fraction": ambiguous_call_fraction,
        "suppressor_auprc": auprc,
    }


def resolvable_pair_concordance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_se: np.ndarray,
    suppressor_cutoff: float,
    z: float,
) -> dict[str, float]:
    """Fraction of statistically-resolvable community pairs the model orders correctly.

    A pair is "resolvable" when its true biomass difference exceeds `z` SEs of that
    difference, i.e. the order is real rather than noise; we score the model only on
    those pairs (random 0.5, perfect 1.0). Prediction ties score 0.5, so a constant
    predictor lands at 0.5 rather than 0. This conditions on ground-truth resolvability,
    not on the model's calls, so it is not the precision buffer in disguise.

    Reported overall and within the suppressor set (true biomass below the cutoff). The
    within-suppressor value is the honest answer to "can the model rank suppressors
    against each other?"; the overall value is dominated by the suppressor-vs-non split.
    `*_resolvable_fraction` reports how much of the ordering is even decidable.
    """
    def concord(y: np.ndarray, predicted: np.ndarray, se: np.ndarray) -> tuple[float, float]:
        if len(y) < 2:
            return float("nan"), float("nan")
        first, second = np.triu_indices(len(y), 1)
        true_diff = y[first] - y[second]
        se_diff = np.sqrt(se[first] ** 2 + se[second] ** 2)
        resolvable = np.abs(true_diff) > z * se_diff
        resolvable_fraction = float(resolvable.mean())
        if not resolvable.any():
            return float("nan"), resolvable_fraction
        predicted_sign = np.sign(predicted[first] - predicted[second])
        # Concordant = 1, discordant = 0, prediction tie = 0.5 (no information).
        score = np.where(
            predicted_sign == 0,
            0.5,
            (predicted_sign == np.sign(true_diff)).astype(float),
        )
        return float(score[resolvable].mean()), resolvable_fraction

    overall, overall_fraction = concord(y_true, y_pred, y_se)
    mask = y_true < suppressor_cutoff
    within, within_fraction = concord(y_true[mask], y_pred[mask], y_se[mask])

    return {
        "concordance_overall": overall,
        "concordance_overall_resolvable_fraction": overall_fraction,
        "concordance_within_suppressors": within,
        "concordance_within_suppressors_resolvable_fraction": within_fraction,
    }


def summarize_metric_rows(
    rows: list[dict[str, object]],
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    raw_metrics = pd.DataFrame(rows)
    # Average repeated splits per model and training set size for learning curves.
    group_columns = group_columns or ["model", "feature_set", "train_rows"]
    metric_columns = [
        column
        for column in raw_metrics.columns
        if column not in {*group_columns, "seed", "test_rows"}
    ]
    summary_rows = []

    for group_values, group_metrics in raw_metrics.groupby(group_columns, sort=False):
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row["runs"] = len(group_metrics)
        row["test_rows"] = int(round(group_metrics["test_rows"].mean()))

        for metric_column in metric_columns:
            row[metric_column] = float(group_metrics[metric_column].mean())
            row[f"{metric_column}_std"] = float(group_metrics[metric_column].std(ddof=0))

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def describe_feature(feature_name: str) -> dict[str, str]:
    if feature_name == "intercept":
        return {"feature_type": "intercept", "species_a": "", "species_b": ""}

    if ":" in feature_name:
        species_a, species_b = feature_name.split(":", maxsplit=1)
        return {
            "feature_type": "pairwise",
            "species_a": species_a,
            "species_b": species_b,
        }

    return {
        "feature_type": "main_effect",
        "species_a": feature_name,
        "species_b": "",
    }


def ridge_pairwise_coefficient_rows(
    dataset: TargetBiomassDataset,
    train_indices: np.ndarray,
    seed: int,
    train_rows: int,
) -> list[dict[str, object]]:
    # Export interpretable main and pairwise effects for the ridge baseline.
    model = build_regressor("ridge_pairwise", seed)
    # Community feature vector: partner main effects plus pairwise interactions.
    x_train = dataset.pairwise_features[train_indices]
    # Numeric label: final target biomass.
    y_train = dataset.target_biomass[train_indices]
    # Learn a linear approximation from community features to target biomass.
    model.fit(x_train, y_train)
    # build_regressor returns a bare Ridge
    ridge = model

    rows = [{
        "seed": seed,
        "model": "ridge_pairwise",
        "feature_set": "pairwise",
        "train_rows": train_rows,
        "feature": "intercept",
        "coefficient": float(ridge.intercept_),
        "abs_coefficient": float(abs(ridge.intercept_)),
        **describe_feature("intercept"),
    }]

    for feature_name, coefficient in zip(
        dataset.feature_names,
        ridge.coef_,
        strict=True,
    ):
        rows.append({
            "seed": seed,
            "model": "ridge_pairwise",
            "feature_set": "pairwise",
            "train_rows": train_rows,
            "feature": feature_name,
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
            **describe_feature(feature_name),
        })

    return rows


def summarize_coefficient_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    coefficients = pd.DataFrame(rows)
    grouped = coefficients.groupby(
        ["model", "feature_set", "train_rows", "feature", "feature_type", "species_a", "species_b"],
        sort=False,
    )
    summary = grouped.agg(
        runs=("seed", "count"),
        coefficient=("coefficient", "mean"),
        coefficient_std=("coefficient", lambda values: values.std(ddof=0)),
        abs_coefficient=("abs_coefficient", "mean"),
        abs_coefficient_std=("abs_coefficient", lambda values: values.std(ddof=0)),
    ).reset_index()
    return summary.sort_values(["train_rows", "abs_coefficient"], ascending=[True, False])


def prediction_rows(
    dataset: TargetBiomassDataset,
    split: SplitDataset,
    model_name: str,
    feature_set: str,
    seed: int,
    train_rows: int,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    # Keep per-community predictions so errors can be traced back to combinations.
    return [
        {
            "seed": seed,
            "model": model_name,
            "feature_set": feature_set,
            "train_rows": train_rows,
            "row_index": int(row_index),
            "community": dataset.communities[row_index],
            "partner_count": int(dataset.partner_counts[row_index]),
            "target_biomass": float(dataset.target_biomass[row_index]),
            "predicted_target_biomass": float(prediction),
            "absolute_error": float(abs(dataset.target_biomass[row_index] - prediction)),
        }
        for row_index, prediction in zip(split.test_indices, y_pred, strict=True)
    ]


def plot_learning_curve(metrics: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Write a learning-curve plot for one target-biomass metric."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    for model_name, model_metrics in metrics.groupby("model"):
        model_metrics = model_metrics.sort_values("train_rows")
        ax.plot(
            model_metrics["train_rows"],
            model_metrics[metric],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title(f"Target Biomass {metric.upper()} by Training Rows", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training rows")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_size(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Plot one metric against partner count at the largest training size."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    final_train_rows = metrics["train_rows"].max()
    final_metrics = metrics[metrics["train_rows"] == final_train_rows]

    for model_name, model_metrics in final_metrics.groupby("model"):
        model_metrics = model_metrics.sort_values("partner_count")
        ax.plot(
            model_metrics["partner_count"],
            model_metrics[metric],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Partner count")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cutoff_sweep(
    predictions: pd.DataFrame,
    biomass: np.ndarray,
    output_path: Path,
    value: str = "precision",
    percentiles: tuple[float, ...] = (3, 5, 7, 10, 15, 20, 28),
    train_sizes_to_show: tuple[int, ...] = (50, 200, 1000),
) -> None:
    """Suppressor-call precision/recall vs cutoff strictness ("bottom X% of biomass").

    One panel per model, one line per training size. Metrics are averaged across seeds and
    a cutoff with no calls is shown as 0, so the curve is gap-free. Evaluated on the
    held-out test set; cutoff values are anchored on the full-dataset percentiles.
    """
    cutoffs = [float(np.percentile(biomass, pct)) for pct in percentiles]

    # Skip constant predictors (e.g. mean_baseline): they never cross a suppressor cutoff,
    # so their panel would be empty and read as a bug.
    def varies(model_name: str) -> bool:
        group = predictions[predictions["model"] == model_name]
        spread = group.groupby(["seed", "train_rows"])["predicted_target_biomass"].std()
        return float(spread.fillna(0.0).max()) > 1e-9

    models = [name for name in sorted(predictions["model"].unique()) if varies(name)]
    max_train = int(predictions["train_rows"].max())
    available = sorted(int(rows) for rows in predictions["train_rows"].unique())
    train_sizes: list[int] = []
    for target in (*train_sizes_to_show, max_train):
        nearest = min(available, key=lambda candidate: abs(candidate - target))
        if nearest not in train_sizes:
            train_sizes.append(nearest)
    train_sizes.sort()

    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows), dpi=120, squeeze=False)
    panels = axes.ravel()

    for index, model_name in enumerate(models):
        ax = panels[index]
        model_rows = predictions[predictions["model"] == model_name]
        for train_rows in train_sizes:
            size_rows = model_rows[model_rows["train_rows"] == train_rows]
            if size_rows.empty:
                continue
            series = []
            for cutoff in cutoffs:
                per_seed = []
                for _, seed_rows in size_rows.groupby("seed"):
                    y_true = seed_rows["target_biomass"].to_numpy(dtype=float)
                    y_pred = seed_rows["predicted_target_biomass"].to_numpy(dtype=float)
                    called = y_pred < cutoff
                    true = y_true < cutoff
                    hits = int((called & true).sum())
                    if value == "recall":
                        per_seed.append(hits / int(true.sum()) if true.any() else np.nan)
                    elif called.any():
                        per_seed.append(hits / int(called.sum()))
                    # precision: a seed with no calls contributes no estimate
                values = np.array([v for v in per_seed if not np.isnan(v)], dtype=float)
                # Average across seeds; a cutoff with no calls anywhere shows as 0.
                series.append(float(values.mean()) if values.size else 0.0)
            label = "all" if train_rows == max_train else str(train_rows)
            ax.plot(list(percentiles), series, marker="o", linewidth=2, label=f"{label} rows")

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("cutoff = bottom X%")
        ax.set_ylabel(f"suppressor {value}")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    for empty in range(len(models), len(panels)):
        panels[empty].axis("off")

    fig.suptitle(f"Suppressor-call {value} by cutoff strictness", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_percentile_overlap(
    predictions: pd.DataFrame,
    output_path: Path,
    percentiles: tuple[float, ...] = (3, 5, 7, 10, 15, 20, 28),
    train_sizes_to_show: tuple[int, ...] = (50, 200, 1000),
) -> None:
    """How well the model's sorted predictions place communities into the bottom X%.

    For each percentile X, take the model's lowest-predicted X% of communities and the
    truly-lowest X% (equal-sized sets), and report the overlap fraction -- "of the
    communities the model sorts into the bottom X%, how many really belong there".
    Because both sets are the same fixed size, the model always picks something, so this
    is gap-free (no zero-call issue). Averaged across seeds; one line per training size.
    The dashed grey line is the chance level (overlap = X% for a random pick); a model
    above it has real ranking signal even if it is far from perfect placement.
    """
    def varies(model_name: str) -> bool:
        group = predictions[predictions["model"] == model_name]
        spread = group.groupby(["seed", "train_rows"])["predicted_target_biomass"].std()
        return float(spread.fillna(0.0).max()) > 1e-9

    models = [name for name in sorted(predictions["model"].unique()) if varies(name)]
    max_train = int(predictions["train_rows"].max())
    available = sorted(int(rows) for rows in predictions["train_rows"].unique())
    train_sizes: list[int] = []
    for target in (*train_sizes_to_show, max_train):
        nearest = min(available, key=lambda candidate: abs(candidate - target))
        if nearest not in train_sizes:
            train_sizes.append(nearest)
    train_sizes.sort()

    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows), dpi=120, squeeze=False)
    panels = axes.ravel()

    for index, model_name in enumerate(models):
        ax = panels[index]
        model_rows = predictions[predictions["model"] == model_name]
        for train_rows in train_sizes:
            size_rows = model_rows[model_rows["train_rows"] == train_rows]
            if size_rows.empty:
                continue
            series = []
            for pct in percentiles:
                per_seed = []
                for _, seed_rows in size_rows.groupby("seed"):
                    y_true = seed_rows["target_biomass"].to_numpy(dtype=float)
                    y_pred = seed_rows["predicted_target_biomass"].to_numpy(dtype=float)
                    k = max(1, int(round(pct / 100 * len(y_true))))
                    true_bottom = set(np.argsort(y_true)[:k].tolist())
                    pred_bottom = set(np.argsort(y_pred)[:k].tolist())
                    per_seed.append(len(true_bottom & pred_bottom) / k)
                series.append(float(np.mean(per_seed)))
            label = "all" if train_rows == max_train else str(train_rows)
            ax.plot(list(percentiles), series, marker="o", linewidth=2, label=f"{label} rows")

        ax.plot(list(percentiles), [pct / 100 for pct in percentiles],
                color="grey", linestyle="--", linewidth=1, label="chance")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("bottom X%")
        ax.set_ylabel("overlap with true bottom-X%")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    for empty in range(len(models), len(panels)):
        panels[empty].axis("off")

    fig.suptitle("Predicted vs true bottom-X% overlap (grey = chance)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def skill_vs_baseline_rows(
    metrics: pd.DataFrame,
    baseline_model: str = "mean_baseline",
) -> pd.DataFrame:
    """Express each model's headline metrics as improvement over the mean baseline.

    Makes "is this better than predicting the average?" explicit: spearman_skill is 0
    when a model matches the baseline, and rmse_skill is the fraction of the baseline's
    error it removes (the R2-style read).
    """
    if baseline_model not in set(metrics["model"]):
        return pd.DataFrame()

    baseline = metrics[metrics["model"] == baseline_model].set_index("train_rows")
    rows = []
    for _, row in metrics[metrics["model"] != baseline_model].iterrows():
        base = baseline.loc[row["train_rows"]]
        rows.append({
            "model": row["model"],
            "feature_set": row["feature_set"],
            "train_rows": row["train_rows"],
            "spearman": float(row["spearman"]),
            "spearman_skill": float(row["spearman"] - base["spearman"]),
            "rmse_skill": float(1 - row["rmse"] / max(base["rmse"], 1e-12)),
        })
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    """Write a CSV with float columns rounded so the output stays human-readable.

    Rounds a copy only, so in-memory frames keep full precision for plotting.
    """
    rounded = frame.copy()
    float_columns = rounded.select_dtypes(include="float").columns
    # Adding 0.0 collapses "-0.0" to "0.0" so near-zero values do not read as bugs.
    rounded[float_columns] = rounded[float_columns].round(decimals) + 0.0
    rounded.to_csv(path, index=False)


def run_benchmarks(
    summary_path: str,
    output_dir: str,
    test_size: float,
    seed: int,
    species_ids: list[str] | None,
    target_species: str | None,
    repeat_count: int,
    train_sizes: list[int | str],
    suppressor_fold: float,
    buffer_z: float,
    concordance_z: float,
    suppressor_target_scale: str,
    selected_models: list[str] | None,
) -> tuple[Path, Path]:
    """Run target-biomass benchmarks and write metrics."""
    dataset = load_dataset(summary_path, target_species, species_ids)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Global-median anchor for the suppressor class cutoff: a fixed biological
    # reference so the suppressive/non-suppressive labels do not drift with the split.
    median_biomass = float(np.median(dataset.target_biomass))
    if suppressor_target_scale == "raw":
        classification_cutoff = median_biomass / suppressor_fold
    else:
        classification_cutoff = median_biomass - float(np.log(suppressor_fold))

    # Accumulate split-level records first, then summarize at the end.
    metric_rows = []
    suppression_size_rows = []
    prediction_output_rows = []
    coefficient_rows = []
    classification_rows = []

    for run_index in range(repeat_count):
        split_seed = seed + run_index
        split = split_dataset(dataset, test_size, split_seed)
        rng = np.random.default_rng(split_seed)
        seen_train_rows = set()

        # Learning curves answer how many measured communities are needed.
        for requested_train_size in train_sizes:
            if requested_train_size == "all" or requested_train_size >= len(split.train_indices):
                train_indices = split.train_indices
            else:
                train_indices = rng.choice(
                    split.train_indices,
                    size=requested_train_size,
                    replace=False,
                )

            train_rows = len(train_indices)
            if train_rows in seen_train_rows:
                continue
            seen_train_rows.add(train_rows)

            # Coefficients are exported at every effective train size.
            coefficient_rows.extend(
                ridge_pairwise_coefficient_rows(
                    dataset,
                    train_indices,
                    split_seed,
                    train_rows,
                )
            )

            for model_name, feature_set in model_configs(selected_models):
                # Feature selection before fitting
                features = model_features(dataset, feature_set)
                x_train = features[train_indices]
                y_train = dataset.target_biomass[train_indices]
                x_test = features[split.test_indices]
                y_test = dataset.target_biomass[split.test_indices]
                # Initialize the model; GNN uses community graphs instead of sklearn matrices.
                if model_name == "gnn":
                    model = GLVIdentityGNNRegressor(
                        partner_count=len(dataset.partner_ids),
                        seed=split_seed,
                    )
                else:
                    model = build_regressor(model_name, split_seed)
                # Train the model
                model.fit(x_train, y_train)
                # Make predictions on test set
                y_pred = model.predict(x_test)

                metric_rows.append({
                    "seed": split_seed,
                    "model": model_name,
                    "feature_set": feature_set,
                    "train_rows": train_rows,
                    "test_rows": len(split.test_indices),
                    **regression_metrics(y_test, y_pred),
                })
                # Noise-buffered suppressor classification, scored on the held-out set.
                classification_rows.append({
                    "seed": split_seed,
                    "model": model_name,
                    "feature_set": feature_set,
                    "train_rows": train_rows,
                    "test_rows": len(split.test_indices),
                    **suppressor_classification_metrics(
                        y_test,
                        y_pred,
                        dataset.target_se[split.test_indices],
                        median_biomass,
                        suppressor_fold,
                        buffer_z,
                        suppressor_target_scale,
                    ),
                    **resolvable_pair_concordance(
                        y_test,
                        y_pred,
                        dataset.target_se[split.test_indices],
                        classification_cutoff,
                        concordance_z,
                    ),
                })
                test_partner_counts = dataset.partner_counts[split.test_indices]
                for partner_count in np.unique(test_partner_counts):
                    size_mask = test_partner_counts == partner_count
                    # Per-size point-prediction and ranking quality.
                    suppression_size_rows.append({
                        "seed": split_seed,
                        "model": model_name,
                        "feature_set": feature_set,
                        "train_rows": train_rows,
                        "partner_count": int(partner_count),
                        "test_rows": int(size_mask.sum()),
                        **regression_metrics(y_test[size_mask], y_pred[size_mask]),
                    })
                prediction_output_rows.extend(
                    prediction_rows(
                        dataset,
                        split,
                        model_name,
                        feature_set,
                        split_seed,
                        train_rows,
                        y_pred,
                    )
                )

    split_metrics_path = output_path / "target_biomass_split_metrics.csv"
    metrics_path = output_path / "target_biomass_metrics.csv"
    suppression_by_size_split_path = output_path / "target_suppression_by_size_split.csv"
    suppression_by_size_path = output_path / "target_suppression_by_size.csv"
    prediction_path = output_path / "target_biomass_predictions.csv"
    coefficient_split_path = output_path / "target_biomass_coefficients_split.csv"
    coefficient_path = output_path / "target_biomass_coefficients.csv"
    skill_path = output_path / "target_biomass_skill_vs_baseline.csv"
    classification_split_path = output_path / "target_suppressor_classification_split.csv"
    classification_path = output_path / "target_suppressor_classification.csv"

    # Write raw split outputs.
    write_csv(pd.DataFrame(metric_rows), split_metrics_path)
    write_csv(pd.DataFrame(suppression_size_rows), suppression_by_size_split_path)
    write_csv(pd.DataFrame(prediction_output_rows), prediction_path)
    write_csv(pd.DataFrame(coefficient_rows), coefficient_split_path)
    write_csv(pd.DataFrame(classification_rows), classification_split_path)

    metrics = summarize_metric_rows(metric_rows)
    suppression_by_size = summarize_metric_rows(
        suppression_size_rows,
        ["model", "feature_set", "train_rows", "partner_count"],
    )
    classification = summarize_metric_rows(classification_rows)
    coefficients = summarize_coefficient_rows(coefficient_rows)
    write_csv(metrics, metrics_path)
    write_csv(suppression_by_size, suppression_by_size_path)
    write_csv(classification, classification_path)
    write_csv(coefficients, coefficient_path)
    write_csv(skill_vs_baseline_rows(metrics), skill_path)

    plot_learning_curve(metrics, "rmse", output_path / "target_biomass_rmse_learning_curve.png")
    plot_learning_curve(metrics, "spearman", output_path / "target_biomass_spearman_learning_curve.png")
    plot_metric_by_size(
        suppression_by_size,
        "spearman",
        output_path / "target_spearman_by_size.png",
        "Ranking Quality (Spearman) by Partner Count",
        "Spearman rank correlation",
    )
    # Suppressor-classification data efficiency: how few measurements are needed
    plot_learning_curve(
        classification,
        "suppressor_precision",
        output_path / "target_suppressor_precision_learning_curve.png",
    )
    plot_learning_curve(
        classification,
        "suppressor_auprc",
        output_path / "target_suppressor_auprc_learning_curve.png",
    )
    # Coarse-ranking ability: high overall (separates suppressors from non-suppressors),
    # near-chance within the suppressor set (can it rank suppressors against each other?).
    plot_learning_curve(
        classification,
        "concordance_overall",
        output_path / "target_concordance_overall_learning_curve.png",
    )
    plot_learning_curve(
        classification,
        "concordance_within_suppressors",
        output_path / "target_concordance_within_suppressors_learning_curve.png",
    )
    # plot different cutoffs to see the performance trade-offs
    predictions_frame = pd.DataFrame(prediction_output_rows)
    plot_cutoff_sweep(
        predictions_frame,
        dataset.target_biomass,
        output_path / "target_cutoff_sweep_precision.png",
        value="precision",
    )
    plot_cutoff_sweep(
        predictions_frame,
        dataset.target_biomass,
        output_path / "target_cutoff_sweep_recall.png",
        value="recall",
    )
    # Rank view: how much of the model's predicted bottom-X% is truly bottom-X% (gap-free).
    plot_percentile_overlap(
        predictions_frame,
        output_path / "target_percentile_overlap.png",
    )

    return metrics_path, prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sklearn regressors using target-species GLV outputs."
    )
    parser.add_argument("summary_path")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/ml")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument(
        "--train-sizes",
        default="25,50,100,200,500,1000,all",
        help="Comma-separated training row counts, plus optional all.",
    )
    parser.add_argument(
        "--species-ids",
        help="Comma-separated fixed species universe. Defaults to final_* summary columns.",
    )
    parser.add_argument(
        "--target-species",
        help="Target species whose final biomass is predicted.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model list. Defaults to all configured models.",
    )
    parser.add_argument(
        "--suppressor-fold",
        type=float,
        default=2.0,
        help="A community is 'suppressive' when its target is this fold below the median (log target).",
    )
    parser.add_argument(
        "--suppressor-target-scale",
        choices=["log", "raw"],
        default="log",
        help="Use log offset median-ln(fold) or raw median/fold for suppressor labels.",
    )
    parser.add_argument(
        "--buffer-z",
        type=float,
        default=1.96,
        help="Calls within this many SEs of the suppressor cutoff are flagged as borderline (data introsprection, default: two-sided 95%% CI).",
    )
    parser.add_argument(
        "--concordance-z",
        type=float,
        default=1.96,
        help="A community pair counts toward ranking concordance only if its true biomass differs by this many SEs, default: two-sided 95%% CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path, prediction_path = run_benchmarks(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        species_ids=parse_species_ids(args.species_ids),
        target_species=args.target_species,
        repeat_count=args.repeat_count,
        train_sizes=parse_train_sizes(args.train_sizes),
        suppressor_fold=args.suppressor_fold,
        buffer_z=args.buffer_z,
        concordance_z=args.concordance_z,
        suppressor_target_scale=args.suppressor_target_scale,
        selected_models=parse_model_names(args.models),
    )

    print(f"Wrote target-biomass metrics to {metrics_path}")
    print(f"Wrote target-biomass predictions to {prediction_path}")


if __name__ == "__main__":
    main()
