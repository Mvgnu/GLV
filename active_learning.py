#!/usr/bin/env python3
"""Oracle-replay active learning for target-pathogen suppression screens."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ml_benchmark import (
    GLVIdentityGNNRegressor,
    build_regressor,
    load_dataset,
    model_configs,
    model_features,
    parse_model_names,
    parse_species_ids,
    regression_metrics,
    split_dataset,
    suppressor_classification_metrics,
    write_csv,
)


@dataclass(frozen=True)
class ActiveLearningConfig:
    initial_size: int
    batch_size: int
    rounds: int | None
    test_size: float
    seed: int
    seeds: int
    suppressor_fold: float
    diversity_weight: float
    ensemble_size: int
    uncertainty_beta: float
    phase2_top_k: int


def stratified_initial_indices(
    candidate_indices: np.ndarray,
    partner_counts: np.ndarray,
    requested_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the initial measured set while preserving partner-count coverage."""
    grouped = {
        int(partner_count): list(rng.permutation(candidate_indices[partner_counts[candidate_indices] == partner_count]))
        for partner_count in np.unique(partner_counts[candidate_indices])
    }
    selected: list[int] = []
    partner_count_order = sorted(grouped)

    # Round-robin across size classes so the first 50 does not overrepresent one size.
    while len(selected) < requested_size and any(grouped.values()):
        for partner_count in partner_count_order:
            if grouped[partner_count]:
                selected.append(int(grouped[partner_count].pop()))
                if len(selected) == requested_size:
                    break

    return np.array(selected, dtype=int)


def train_model(dataset, model_name: str, feature_set: str, train_indices: np.ndarray, seed: int):
    """Fit one active-learning surrogate on the currently measured rows."""
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


def select_acquisitions(
    strategy: str,
    dataset,
    measured_indices: np.ndarray,
    candidate_indices: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    acquisition_scores: np.ndarray,
    batch_size: int,
    diversity_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if strategy == "random":
        return rng.choice(candidate_indices, size=batch_size, replace=False)
    if strategy == "max_diversity":
        remaining = rng.permutation(candidate_indices)
        min_distances = np.full(len(remaining), dataset.presence.shape[1] + 1)
        for measured_index in measured_indices:
            min_distances = np.minimum(
                min_distances,
                np.abs(
                    dataset.presence[remaining] - dataset.presence[measured_index]
                ).sum(axis=1),
            )
        selected: list[int] = []
        while len(selected) < batch_size:
            best_position = int(np.argmax(min_distances))
            chosen = int(remaining[best_position])
            selected.append(chosen)
            min_distances = np.minimum(
                min_distances,
                np.abs(dataset.presence[remaining] - dataset.presence[chosen]).sum(axis=1),
            )
            remaining = np.delete(remaining, best_position)
            min_distances = np.delete(min_distances, best_position)
        return np.array(selected, dtype=int)
    if strategy == "predicted_best":
        order = np.argsort(predictions[candidate_indices])
        return candidate_indices[order[:batch_size]]
    if strategy in {
        "ensemble_uncertainty",
        "ridge_posterior_uncertainty",
        "committee_disagreement",
    }:
        # uncertainty calculation varies, but method is shared
        order = np.argsort(-uncertainty[candidate_indices])
        return candidate_indices[order[:batch_size]]
    if strategy in {
        "ucb_suppression",
        "ridge_posterior_ucb",
        "bayesian_optimization",
    }:
        # acquisition_scores calculation varies, but method is shared
        order = np.argsort(acquisition_scores[candidate_indices])
        return candidate_indices[order[:batch_size]]
    if strategy == "diverse_predicted_best":
        selected: list[int] = []
        remaining = candidate_indices.copy()

        # Greedy tradeoff: predicted suppression first, with a small diversity reward.
        while len(selected) < batch_size and len(remaining):
            scores = []
            for candidate_index in remaining:
                if selected:
                    selected_presence = dataset.presence[np.array(selected, dtype=int)]
                    candidate_presence = dataset.presence[int(candidate_index)]
                    distances = np.abs(
                        selected_presence - candidate_presence
                    ).sum(axis=1)
                    min_distance = float(distances.min())
                else:
                    min_distance = float(dataset.presence.shape[1])
                scores.append(
                    float(predictions[candidate_index] - diversity_weight * min_distance)
                )
            best_position = int(np.argmin(scores))
            selected.append(int(remaining[best_position]))
            # remove selected from remaining
            remaining = np.delete(remaining, best_position)

        return np.array(selected, dtype=int)
    if strategy == "size_balanced_predicted_best":
        selected: list[int] = []
        grouped = {}
        for partner_count in sorted(np.unique(dataset.partner_counts[candidate_indices])):
            group = candidate_indices[
                dataset.partner_counts[candidate_indices] == partner_count
            ]
            grouped[int(partner_count)] = list(group[np.argsort(predictions[group])])

        # Round-robin best predicted candidates across partner counts.
        while len(selected) < batch_size and any(grouped.values()):
            for partner_count in sorted(grouped):
                if grouped[partner_count]:
                    selected.append(int(grouped[partner_count].pop(0)))
                    if len(selected) == batch_size:
                        break

        return np.array(selected, dtype=int)

    raise ValueError(f"Unknown acquisition strategy: {strategy}")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensemble_prediction_statistics(
    dataset,
    model_name: str,
    feature_set: str,
    train_indices: np.ndarray,
    seed: int,
    ensemble_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap fits provide a real data-sensitivity uncertainty estimate."""
    features = model_features(dataset, feature_set)
    predictions = []
    rng = np.random.default_rng(seed)
    # trains multiple models with bootstrapped data to estimate uncertainty by sampling from the training data with replacement
    for ensemble_index in range(ensemble_size):
        bootstrap_indices = rng.choice(
            train_indices,
            size=len(train_indices),
            replace=True,
        )
        model, _features = train_model(
            dataset,
            model_name,
            feature_set,
            bootstrap_indices,
            seed + 1009 * (ensemble_index + 1),
        )
        predictions.append(model.predict(features))

    # calculates the mean and standard deviation (== model uncertainty) of the predictions across the ensemble
    stacked = np.vstack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def ridge_posterior_statistics(
    dataset,
    train_indices: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Bayesian-ridge predictive mean and std -- no resampling.

    Ridge is the MAP of Bayesian linear regression (Gaussian weight prior <-> L2
    penalty), so the weight posterior is Gaussian with covariance
    ``Sigma = sigma^2 (X^T X + alpha I)^-1``. The predictive variance at x is
    ``sigma^2 (1 + x^T (X^T X + alpha I)^-1 x)``: the leverage term grows for
    inputs poorly spanned by the measured rows, giving a genuine off-manifold
    exploration signal that a single deterministic ridge fit cannot. ``alpha``
    matches ``Ridge(alpha=1.0)`` in ``build_regressor`` and the features are the
    same pairwise design the loop model sees, so this is
    the closed form of what bootstrap-ridge only approximates.
    """
    # Build numerical features for all candidate samples
    features = model_features(dataset, "pairwise")
   
    x_train = features[train_indices]
    x_all = features

    # take biomass of training rows
    y_train = dataset.target_biomass[train_indices]
    # mean of training target biomass
    y_mean = float(y_train.mean())
    # centered training target biomass around 0
    y_centered = y_train - y_mean

    # Gram matrix: feature-feature cross-products in training data, X^T X
    gram = x_train.T @ x_train

    # Add alpha * I (identity matrix, np.eye(x_train.shape[1])) 
    # large alpha -> more regularization -> smaller weights
    # less regularization -> smaller alpha -> larger weights
    # X^T X + alpha I
    a_matrix = gram + alpha * np.eye(x_train.shape[1])

    # inverse of the regularized a_matrix 
    # used to compute ridge weights & uncertainty/leverage
    a_inv = np.linalg.inv(a_matrix)

    # computes ridge weights for the centered target
    # (X^T X + alpha I)^-1 X^T y
    weights = a_inv @ (x_train.T @ y_centered)
    
    # predicts centered biomass, then adds back the training target mean as the intercept
    mean = x_all @ weights + y_mean

    # take difference between actual (y_centered) and predicted biomass of training data
    residuals = y_centered - x_train @ weights

    # calculate effective dof (degrees of freedom) of the model
    effective_dof = float(np.sum((x_train @ a_inv) * x_train))

    # number of data points - effective dof 
    # estimates noise variance, correcting for model flexibility
    dof = max(len(train_indices) - effective_dof, 1.0)

    # mean squared residual error, corrected by effective degrees of freedom
    sigma_sq = float(residuals @ residuals) / dof

    # sigma_sq floor (minimum value to prevent 0 uncertainty) 
    sigma_sq = max(sigma_sq, 1e-9 * float(np.var(y_train)) + 1e-12)

    # leverage: large values -> feature "space" point not well covered in training data
    # for every candidate x: x^T (X^T X + alpha I)^-1 x
    leverage = np.sum((x_all @ a_inv) * x_all, axis=1)

    # predictive standard deviation = sqrt(sigma_sq * (1 + leverage))
    # "1.0 +" -> variance for a future observed target, not just uncertainty in the mean prediction
    # (noise + parameter uncertainty)
    std = np.sqrt(np.maximum(sigma_sq * (1.0 + leverage), 0.0))
    return mean, std


COMMITTEE_MODELS: tuple[tuple[str, str], ...] = (
    ("ridge_pairwise", "pairwise"),
    ("random_forest", "pairwise"),
    ("hist_gradient_boosting", "pairwise"),
)


def committee_prediction_statistics(
    dataset,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query-by-committee uncertainty: disagreement across distinct model
    families (linear vs. random forest vs. boosted trees) all fitted on the
    SAME measured rows. Captures structural / inductive-bias uncertainty, the
    regions where different model classes diverge.
    """
    predictions = []
    # train each model once on the same measured rows and append predictions
    for offset, (model_name, feature_set) in enumerate(COMMITTEE_MODELS):
        features = model_features(dataset, feature_set)
        model, _features = train_model(
            dataset,
            model_name,
            feature_set,
            train_indices,
            seed + 7919 * (offset + 1),
        )
        predictions.append(model.predict(features))

    # stack predictions and compute mean and std (== committee uncertainty)
    stacked = np.vstack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def bayesian_optimization_statistics(
    dataset,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-process Expected-Improvement acquisition (Bayesian optimization).

    Fits a GP surrogate on the measured rows and scores Expected Improvement for
    minimization (lower target biomass = stronger suppression), balancing "test what looks
    suppressive" (mean) against "test where the model is unsure" (std). Returns the GP mean,
    std, and negated EI so the shared minimizing selector takes the highest-EI batch.
    """
    # build feature matrix of all communities including pairs
    features = model_features(dataset, "pairwise")
    # convert train_indices to numpy array
    train = np.asarray(train_indices, dtype=int)
    # take biomass of training rows
    y_train = dataset.target_biomass[train]
    # define GP kernel with constant kernel (overall scaling), RBF kernel (smoothness), and WhiteKernel (noise)
    kernel = (
        # output scale / amplitude of the GP function.
        # larger = allows larger variation in predicted biomass.
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        # RBF kernel (smoothness): how far apart points can be while still being considered similar
        * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
        # add observation noise
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )
    x_train = features[train]
    x_all = features
    
    # gp regression model 
    model = GaussianProcessRegressor(
        kernel=kernel,
        # add tiny jitter for numerical stability
        alpha=1e-6,
        # normalize y_train to improve GP fitting stability
        normalize_y=True,
        # one additional optimizer restart from a different initialization
        n_restarts_optimizer=1,
        # random state for reproducibility
        random_state=seed,
    )

    # Fit gaussian process to training data
    model.fit(x_train, y_train)

    # for all candidate rows predict biomass, output uncertainty
    mean, std = model.predict(x_all, return_std=True)

    # best known suppressor
    best_seen = float(np.min(y_train))

    # improvement best seen - predicted
    improvement = best_seen - mean

    # compute improvement relative to uncertainty if std not 0, otherwise set improvement to 0 
    # (out=np.zeros_like sets improvement to zero on where path)
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)

    # Expected Improvement balances exploitation and exploration
    # since GP gives a continuous distribution of predictions, score the candidates expected improvement given that continuous distribution
    # norm.cdf(z) -> probability that the candidate beats the best observed value (left tail of standard normal distribution)
    # norm.pdf(z) -> weights the std-based exploration bonus (height of standard normal curve for given z)
    expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)

    # Selection minimizes acquisition scores, so negate EI to pick the highest-EI batch.
    return mean, std, -expected_improvement


def acquisition_statistics(
    strategy: str,
    dataset,
    model,
    model_name: str,
    feature_set: str,
    features: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    ensemble_size: int,
    uncertainty_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "bayesian_optimization":
        return bayesian_optimization_statistics(dataset, train_indices, seed)

    if strategy in {"ensemble_uncertainty", "ucb_suppression"}:
        mean, uncertainty = ensemble_prediction_statistics(
            dataset,
            model_name,
            feature_set,
            train_indices,
            seed,
            ensemble_size,
        )
        if strategy == "ucb_suppression":
            # substract uncertainty weighed by param beta for acquisitions score
            return mean, uncertainty, mean - uncertainty_beta * uncertainty
        return mean, uncertainty, -uncertainty

    if strategy in {"ridge_posterior_uncertainty", "ridge_posterior_ucb"}:
        mean, uncertainty = ridge_posterior_statistics(dataset, train_indices)
        if strategy == "ridge_posterior_ucb":
            return mean, uncertainty, mean - uncertainty_beta * uncertainty
        return mean, uncertainty, -uncertainty

    if strategy == "committee_disagreement":
        mean, uncertainty = committee_prediction_statistics(dataset, train_indices, seed)
        return mean, uncertainty, -uncertainty

    # predicted best strategies
    predictions = model.predict(features)
    uncertainty = np.zeros_like(predictions, dtype=float)
    return predictions, uncertainty, predictions


def summarize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    group_columns = ["model", "strategy", "measured_count"]
    metric_columns = [
        column
        for column in frame.columns
        if column not in {*group_columns, "seed", "round"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary_rows = []

    for group_values, group in frame.groupby(group_columns, sort=False):
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row["runs"] = len(group)
        for metric_column in metric_columns:
            row[metric_column] = float(group[metric_column].mean())
            row[f"{metric_column}_std"] = float(group[metric_column].std(ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def round_metrics(
    dataset,
    model,
    features: np.ndarray,
    audit_indices: np.ndarray,
    measured_indices: np.ndarray,
    discoverable_indices: np.ndarray,
    median_biomass: float,
    suppressor_fold: float,
) -> dict[str, float]:
    y_true = dataset.target_biomass[audit_indices]
    y_pred = model.predict(features[audit_indices])
    suppressor_metrics = suppressor_classification_metrics(
        y_true,
        y_pred,
        dataset.target_se[audit_indices],
        median_biomass,
        suppressor_fold,
        buffer_z=1.96,
        suppressor_target_scale="log",
    )

    best_measured = float(np.min(dataset.target_biomass[measured_indices]))
    global_best = float(np.min(dataset.target_biomass[discoverable_indices]))
    global_best_gap = best_measured - global_best
    complete_global_best = float(np.min(dataset.target_biomass))
    complete_global_best_gap = best_measured - complete_global_best

    return {
        "audit_rows": int(len(audit_indices)),
        "best_measured_biomass": best_measured,
        "global_best_biomass": global_best,
        "global_best_gap": float(global_best_gap),
        "global_best_gap_fraction": float(global_best_gap / max(abs(global_best), 1e-12)),
        "complete_global_best_biomass": complete_global_best,
        "complete_global_best_gap": float(complete_global_best_gap),
        "complete_global_best_gap_fraction": float(
            complete_global_best_gap / max(abs(complete_global_best), 1e-12)
        ),
        **regression_metrics(y_true, y_pred),
        **suppressor_metrics,
    }


def surrogate_recommendation_metrics(
    dataset,
    model,
    features: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Validate the model's exact top-k recommendations on the finite screen."""
    predictions = model.predict(features)
    recommended = np.argsort(predictions, kind="stable")[:top_k]
    validated = dataset.target_biomass[recommended]
    global_best = float(np.min(dataset.target_biomass))
    gap = float(np.min(validated) - global_best)
    return (
        {
            "surrogate_best_validated_biomass": float(np.min(validated)),
            "surrogate_mean_validated_biomass": float(np.mean(validated)),
            "surrogate_global_best_biomass": global_best,
            "surrogate_global_best_gap": gap,
            "surrogate_global_best_gap_fraction": gap / max(abs(global_best), 1e-12),
        },
        recommended,
        predictions,
    )


def acquisition_rows(
    dataset,
    model_name: str,
    strategy: str,
    seed: int,
    round_index: int,
    selected_indices: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    acquisition_scores: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for rank, row_index in enumerate(selected_indices, start=1):
        rows.append({
            "seed": seed,
            "model": model_name,
            "strategy": strategy,
            "round": round_index,
            "acquisition_rank": rank,
            "row_index": int(row_index),
            "community": dataset.communities[row_index],
            "partner_count": int(dataset.partner_counts[row_index]),
            "predicted_target_biomass": float(predictions[row_index]),
            "predicted_uncertainty": float(uncertainty[row_index]),
            "true_target_biomass": float(dataset.target_biomass[row_index]),
            "acquisition_score": float(acquisition_scores[row_index]),
        })
    return rows


def panel_grid(items: list[str]) -> tuple[plt.Figure, np.ndarray]:
    columns = min(3, len(items))
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4 * rows),
        dpi=120,
        squeeze=False,
    )
    panels = axes.ravel()
    for empty in range(len(items), len(panels)):
        panels[empty].axis("off")
    return fig, panels


def plot_metric_by_model(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    fixed_unit_axis: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    models = sorted(summary["model"].unique())
    fig, panels = panel_grid(models)

    for index, model_name in enumerate(models):
        ax = panels[index]
        model_rows = summary[summary["model"] == model_name]
        for strategy, group in model_rows.groupby("strategy"):
            group = group.sort_values("measured_count")
            ax.plot(
                group["measured_count"],
                group[metric],
                marker="o",
                linewidth=1.8,
                label=strategy,
            )

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Measured communities")
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        elif fixed_unit_axis:
            ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_strategy(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    fixed_unit_axis: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    strategies = sorted(summary["strategy"].unique())
    fig, panels = panel_grid(strategies)

    for index, strategy in enumerate(strategies):
        ax = panels[index]
        strategy_rows = summary[summary["strategy"] == strategy]
        for model_name, group in strategy_rows.groupby("model"):
            group = group.sort_values("measured_count")
            ax.plot(
                group["measured_count"],
                group[metric],
                marker="o",
                linewidth=1.8,
                label=model_name,
            )

        ax.set_title(strategy, fontsize=11, fontweight="bold")
        ax.set_xlabel("Measured communities")
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        elif fixed_unit_axis:
            ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def write_active_learning_plots(summary: pd.DataFrame, output_path: Path) -> None:
    plot_metric_by_model(
        summary,
        "global_best_gap_fraction",
        output_path / "best_suppressor_gap_model_dependent_by_model.png",
        "Model-Dependent Suppressor Discovery by Model",
        "Relative gap to best suppressor",
        y_limits=(0, 0.4),
    )
    plot_metric_by_model(
        summary,
        "surrogate_global_best_gap_fraction",
        output_path / "surrogate_recommendation_gap_by_model.png",
        "Validated Surrogate Recommendations by Model",
        "Relative gap to global best",
    )
    plot_metric_by_model(
        summary,
        "rmse",
        output_path / "model_rmse_by_model.png",
        "RMSE by Model",
        "RMSE",
    )
    plot_metric_by_strategy(
        summary,
        "rmse",
        output_path / "model_rmse_by_strategy.png",
        "RMSE by Strategy",
        "RMSE",
    )
    plot_metric_by_model(
        summary,
        "suppressor_precision",
        output_path / "suppressor_precision_by_model.png",
        "Suppressor Precision by Model",
        "Suppressor precision",
        fixed_unit_axis=True,
    )
    plot_metric_by_strategy(
        summary,
        "suppressor_precision",
        output_path / "suppressor_precision_by_strategy.png",
        "Suppressor Precision by Strategy",
        "Suppressor precision",
        fixed_unit_axis=True,
    )
    plot_metric_by_model(
        summary,
        "suppressor_class_recall",
        output_path / "suppressor_recall_by_model.png",
        "Suppressor Recall by Model",
        "Suppressor recall",
        fixed_unit_axis=True,
    )
    plot_metric_by_strategy(
        summary,
        "suppressor_class_recall",
        output_path / "suppressor_recall_by_strategy.png",
        "Suppressor Recall by Strategy",
        "Suppressor recall",
        fixed_unit_axis=True,
    )
    plot_metric_by_model(
        summary,
        "suppressor_auprc",
        output_path / "suppressor_auprc_by_model.png",
        "Suppressor AUPRC by Model",
        "Suppressor AUPRC",
        fixed_unit_axis=True,
    )
    plot_metric_by_strategy(
        summary,
        "suppressor_auprc",
        output_path / "suppressor_auprc_by_strategy.png",
        "Suppressor AUPRC by Strategy",
        "Suppressor AUPRC",
        fixed_unit_axis=True,
    )


def run_active_learning(
    summary_path: str,
    output_dir: str,
    species_ids: list[str] | None,
    target_species: str | None,
    selected_models: list[str] | None,
    strategies: list[str],
    config: ActiveLearningConfig,
) -> tuple[Path, Path, Path, Path]:
    dataset = load_dataset(summary_path, target_species, species_ids)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    median_biomass = float(np.median(dataset.target_biomass))

    round_output_rows: list[dict[str, object]] = []
    acquisition_output_rows: list[dict[str, object]] = []
    recommendation_output_rows: list[dict[str, object]] = []

    for run_index in range(config.seeds):
        split_seed = config.seed + run_index
        split = split_dataset(dataset, config.test_size, split_seed)
        discoverable_indices = np.array(split.train_indices, dtype=int)
        audit_indices = np.array(split.test_indices, dtype=int)
        initial_rng = np.random.default_rng(split_seed)
        initial_indices = stratified_initial_indices(
            discoverable_indices,
            dataset.partner_counts,
            min(config.initial_size, len(discoverable_indices)),
            initial_rng,
        )

        for model_name, feature_set in model_configs(selected_models):
            for strategy in strategies:
                measured_indices = initial_indices.copy()
                candidate_indices = np.setdiff1d(
                    discoverable_indices,
                    measured_indices,
                    assume_unique=False,
                )
                last_new_measurements = 0
                max_rounds = config.rounds
                if max_rounds is None:
                    max_rounds = int(np.ceil(len(candidate_indices) / config.batch_size))

                for round_index in range(max_rounds + 1):
                    model, features = train_model(
                        dataset,
                        model_name,
                        feature_set,
                        measured_indices,
                        split_seed + round_index,
                    )
                    metrics = round_metrics(
                        dataset,
                        model,
                        features,
                        audit_indices,
                        measured_indices,
                        discoverable_indices,
                        median_biomass,
                        config.suppressor_fold,
                    )
                    recommendation_metrics, recommended_indices, all_predictions = (
                        surrogate_recommendation_metrics(
                            dataset,
                            model,
                            features,
                            config.phase2_top_k,
                        )
                    )
                    round_output_rows.append({
                        "seed": split_seed,
                        "model": model_name,
                        "feature_set": feature_set,
                        "strategy": strategy,
                        "round": round_index,
                        "measured_count": int(len(measured_indices)),
                        "new_measurements": int(last_new_measurements),
                        "pool_rows_remaining": int(len(candidate_indices)),
                        **metrics,
                        **recommendation_metrics,
                    })
                    for rank, row_index in enumerate(recommended_indices, start=1):
                        recommendation_output_rows.append({
                            "seed": split_seed,
                            "model": model_name,
                            "strategy": strategy,
                            "round": round_index,
                            "measured_count": int(len(measured_indices)),
                            "recommendation_rank": rank,
                            "row_index": int(row_index),
                            "community": dataset.communities[row_index],
                            "partner_count": int(dataset.partner_counts[row_index]),
                            "predicted_target_biomass": float(all_predictions[row_index]),
                            "true_target_biomass": float(dataset.target_biomass[row_index]),
                        })

                    if round_index == max_rounds or len(candidate_indices) == 0:
                        break

                    predictions, uncertainty, acquisition_scores = acquisition_statistics(
                        strategy,
                        dataset,
                        model,
                        model_name,
                        feature_set,
                        features,
                        measured_indices,
                        split_seed + round_index,
                        config.ensemble_size,
                        config.uncertainty_beta,
                    )
                    selected_indices = select_acquisitions(
                        strategy,
                        dataset,
                        measured_indices,
                        candidate_indices,
                        predictions,
                        uncertainty,
                        acquisition_scores,
                        min(config.batch_size, len(candidate_indices)),
                        config.diversity_weight,
                        np.random.default_rng(
                            split_seed
                            + round_index
                            + sum(
                                (index + 1) * ord(character)
                                for index, character in enumerate(strategy)
                            )
                        ),
                    )
                    acquisition_output_rows.extend(
                        acquisition_rows(
                            dataset,
                            model_name,
                            strategy,
                            split_seed,
                            round_index + 1,
                            selected_indices,
                            predictions,
                            uncertainty,
                            acquisition_scores,
                        )
                    )
                    last_new_measurements = len(selected_indices)
                    # update measured and candidate indices for next round
                    measured_indices = np.concatenate([measured_indices, selected_indices])
                    # remove "newly measured" from candidates
                    candidate_indices = np.setdiff1d(candidate_indices, selected_indices)

    rounds_path = output_path / "active_learning_rounds.csv"
    acquisitions_path = output_path / "active_learning_acquisitions.csv"
    recommendations_path = output_path / "active_learning_recommendations.csv"
    summary_path_out = output_path / "active_learning_summary.csv"
    rounds = pd.DataFrame(round_output_rows)
    acquisitions = pd.DataFrame(acquisition_output_rows)
    recommendations = pd.DataFrame(recommendation_output_rows)
    summary = summarize_rows(round_output_rows)

    write_csv(rounds, rounds_path)
    write_csv(acquisitions, acquisitions_path)
    write_csv(recommendations, recommendations_path)
    write_csv(summary, summary_path_out)

    write_active_learning_plots(summary, output_path)

    return rounds_path, acquisitions_path, recommendations_path, summary_path_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run oracle-replay active learning over target-biomass summaries."
    )
    parser.add_argument("summary_path")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/active_learning")
    parser.add_argument("--target-species")
    parser.add_argument("--species-ids")
    parser.add_argument(
        "--models",
        default="ridge_pairwise,random_forest,hist_gradient_boosting",
        help="Comma-separated model list. GNN is supported but slower.",
    )
    parser.add_argument(
        "--strategies",
        default="random,max_diversity",
        help=(
            "Comma-separated acquisition strategies. Available: random, max_diversity, "
            "predicted_best, diverse_predicted_best, size_balanced_predicted_best, "
            "ensemble_uncertainty (bootstrap-ridge std), ucb_suppression, "
            "ridge_posterior_uncertainty/ridge_posterior_ucb (analytic Bayesian-ridge "
            "variance), committee_disagreement (cross-model query-by-committee), "
            "bayesian_optimization (GP + Expected Improvement; batch BO when batch-size>1)."
        ),
    )
    parser.add_argument("--initial-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.05,
        help="Reward for Hamming-distance diversity in diverse_predicted_best.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Seeded surrogate fits used by ensemble uncertainty strategies.",
    )
    parser.add_argument(
        "--uncertainty-beta",
        type=float,
        default=1.0,
        help="Uncertainty weight for ucb_suppression; higher explores more.",
    )
    parser.add_argument(
        "--phase2-top-k",
        type=int,
        default=5,
        help="Exact lowest-predicted communities validated after each model fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds_path, acquisitions_path, recommendations_path, summary_path = run_active_learning(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        species_ids=parse_species_ids(args.species_ids),
        target_species=args.target_species,
        selected_models=parse_model_names(args.models),
        strategies=parse_csv_list(args.strategies),
        config=ActiveLearningConfig(
            initial_size=args.initial_size,
            batch_size=args.batch_size,
            rounds=args.rounds,
            test_size=args.test_size,
            seed=args.seed,
            seeds=args.seeds,
            suppressor_fold=args.suppressor_fold,
            diversity_weight=args.diversity_weight,
            ensemble_size=args.ensemble_size,
            uncertainty_beta=args.uncertainty_beta,
            phase2_top_k=args.phase2_top_k,
        ),
    )

    print(f"Wrote active-learning rounds to {rounds_path}")
    print(f"Wrote active-learning acquisitions to {acquisitions_path}")
    print(f"Wrote active-learning recommendations to {recommendations_path}")
    print(f"Wrote active-learning summary to {summary_path}")


if __name__ == "__main__":
    main()
