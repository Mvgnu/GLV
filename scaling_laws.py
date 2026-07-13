#!/usr/bin/env python3
"""Species-universe scaling benchmarks for suppressor prediction."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_benchmark import (
    parse_model_names,
    parse_species_ids,
    parse_train_sizes,
    run_benchmarks,
    write_csv,
)
from scaling_law_fits import crossing_bracket


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def log_train_sizes(rows: int, minimum: int = 4, ratio: float = 2.0 ** 0.5) -> list[int | str]:
    """Geometric row budgets, so every universe size is resolved proportionally.

    A fraction grid spaces checkpoints by `rows`, which at k=11 puts the first
    checkpoint at 102 measurements -- already past the requirement it is meant to
    locate. Geometric spacing keeps the relative step constant instead, so the
    bracket around `n_tau` is a fixed ratio wide at every `k`.
    """
    sizes: list[int | str] = []
    value = float(minimum)
    while True:
        size = int(round(value))
        if size >= rows:
            break
        if not sizes or size > sizes[-1]:
            sizes.append(size)
        value *= ratio
    # "all" resolves to every training row the split leaves available.
    sizes.append("all")
    return sizes


def train_sizes_for_universe(
    rows: int,
    train_sizes: list[int | str] | None,
    train_fractions: list[float],
    train_grid: str,
    minimum_rows: int,
) -> list[int | str]:
    if train_sizes is not None:
        return train_sizes
    if train_grid == "log":
        return log_train_sizes(rows, minimum_rows)

    sizes: list[int | str] = []
    for fraction in train_fractions:
        if fraction >= 1:
            sizes.append("all")
        else:
            sizes.append(max(1, int(round(rows * fraction))))

    deduped: list[int | str] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    return deduped


def parse_community(community: str) -> set[str]:
    return {species for species in str(community).split(";") if species}


def infer_partner_ids(summary: pd.DataFrame, target_species: str | None) -> list[str]:
    if target_species is None and "target_species" in summary.columns:
        targets = sorted(
            {str(value) for value in summary["target_species"].dropna().unique()}
        )
        target_species = targets[0] if len(targets) == 1 else None

    species = sorted(
        species_id
        for community in summary["community"].astype(str)
        for species_id in parse_community(community)
        if species_id != target_species
    )
    return sorted(set(species))


def filtered_summary(
    summary: pd.DataFrame,
    retained_species: list[str],
) -> pd.DataFrame:
    retained = set(retained_species)
    rows = []
    for _, row in summary.iterrows():
        community_species = parse_community(row["community"])
        if community_species and community_species.issubset(retained):
            rows.append(row)

    filtered = pd.DataFrame(rows).reset_index(drop=True)
    filtered["partner_count"] = filtered["community"].map(
        lambda value: len(parse_community(value))
    )
    filtered["community_size"] = filtered["partner_count"] + 1
    return filtered


def dataset_row(
    seed: int,
    species_count: int,
    retained_species: list[str],
    summary: pd.DataFrame,
    suppressor_fold: float,
) -> dict[str, object]:
    cutoff = float(np.median(summary["final_target_biomass"])) - float(np.log(suppressor_fold))
    suppressor_count = int((summary["final_target_biomass"] < cutoff).sum())
    return {
        "seed": seed,
        "species_count": species_count,
        "retained_species": ";".join(retained_species),
        "rows": int(len(summary)),
        "suppressor_count": suppressor_count,
        "suppressor_rate": float(suppressor_count / len(summary)),
    }


def load_metric_outputs(
    report_dir: Path,
    seed: int,
    species_count: int,
    retained_species: list[str],
    rows: int,
) -> list[dict[str, object]]:
    classification = pd.read_csv(report_dir / "target_suppressor_classification.csv")
    # A suppressor metric is undefined, not zero, when the held-out split holds no
    # true suppressor. Filling 0.0 makes an unmeasurable universe look like a model
    # that measured everything and failed, and those universes cluster at small k.
    # Leave the NaN so downstream threshold rules and fits can tell the two apart.
    biomass = pd.read_csv(report_dir / "target_biomass_metrics.csv")
    merged = classification.merge(
        biomass[["model", "feature_set", "train_rows", "rmse", "spearman"]],
        on=["model", "feature_set", "train_rows"],
        how="left",
        suffixes=("", "_biomass"),
    )

    metric_rows = []
    for _, row in merged.iterrows():
        metric_rows.append(
            {
                "universe_seed": seed,
                "species_count": species_count,
                "retained_species": ";".join(retained_species),
                "rows": rows,
                "measured_fraction": float(row["train_rows"] / rows),
                **row.to_dict(),
            }
        )
    return metric_rows


def required_row_summary(
    metrics: pd.DataFrame,
    precision_threshold: float,
    recall_threshold: float,
    auprc_threshold: float,
) -> pd.DataFrame:
    """Bracket the budget at which each universe reaches each threshold.

    A universe that never reaches the threshold is right-censored at its largest
    budget, not missing. Reporting only the universes that crossed averages over the
    survivors and understates the requirement wherever crossing is hard, which is
    exactly where the requirement is largest. Both bounds and the censoring flag are
    written so `scaling_law_fits.py` can fit the censored likelihood; the plots that
    read `*_required_rows` remain descriptive and drop censored universes.

    `test_rows` and `true_suppressor_count` travel with each row because a suppressor
    metric is undefined when the split holds no true suppressor, and that must not be
    confused with a genuine failure.

    Averaging `*_required_fraction` across universes is not the same as the fraction
    at which the mean learning curve crosses the threshold. One unlearnable retained
    species set can hold the mean curve below `tau` at every budget while most
    universes cross early.
    """
    thresholds = {
        "precision": ("suppressor_precision", precision_threshold),
        "recall": ("suppressor_class_recall", recall_threshold),
        "auprc": ("suppressor_auprc", auprc_threshold),
    }
    rows = []
    group_columns = ["species_count", "model", "feature_set", "universe_seed"]
    for group_values, group in metrics.groupby(group_columns, sort=False):
        group = group.sort_values("train_rows")
        budgets = group["train_rows"].to_numpy(dtype=float)
        universe_rows = int(group["rows"].iloc[0])
        base = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        base["rows"] = universe_rows
        base["max_budget"] = float(budgets[-1])
        base["test_rows"] = float(group["test_rows"].iloc[0])
        base["true_suppressor_count"] = float(group["true_suppressor_count"].max())

        for prefix, (column, threshold) in thresholds.items():
            values = group[column].to_numpy(dtype=float)
            lower, upper = crossing_bracket(budgets, values, threshold)
            censored = bool(np.isinf(upper))
            base[f"{prefix}_threshold"] = float(threshold)
            base[f"{prefix}_censored"] = censored
            base[f"{prefix}_lower_rows"] = float(lower)
            base[f"{prefix}_required_rows"] = np.nan if censored else float(upper)
            base[f"{prefix}_required_fraction"] = (
                np.nan if censored else float(upper) / universe_rows
            )
        rows.append(base)

    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["species_count", "model", "feature_set", "train_rows"]
    numeric_columns = [
        column
        for column in metrics.columns
        if column not in {*group_columns, "universe_seed"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    rows = []
    for group_values, group in metrics.groupby(group_columns, sort=False):
        row = {
            column: value
            for column, value in zip(group_columns, group_values, strict=True)
        }
        row["universes"] = int(group["universe_seed"].nunique())
        for column in numeric_columns:
            row[column] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def model_panels(frame: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    models = sorted(frame["model"].unique())
    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4 * rows),
        dpi=120,
        squeeze=False,
    )
    panels = axes.ravel()
    for empty in range(len(models), len(panels)):
        panels[empty].axis("off")
    return fig, panels


def plot_required_fraction(
    required: pd.DataFrame,
    metric_prefix: str,
    output_path: Path,
    title: str,
) -> None:
    plot_required_metric(
        required,
        f"{metric_prefix}_required_fraction",
        output_path,
        title,
        "Measured fraction required",
        y_limits=(0, 1.02),
    )


def plot_required_rows(
    required: pd.DataFrame,
    metric_prefix: str,
    output_path: Path,
    title: str,
) -> None:
    plot_required_metric(
        required,
        f"{metric_prefix}_required_rows",
        output_path,
        title,
        "Measured rows required",
    )


def plot_required_metric(
    required: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    valid = required.dropna(subset=[value_column])
    fig, panels = model_panels(required)

    for index, model_name in enumerate(sorted(required["model"].unique())):
        ax = panels[index]
        group = valid[valid["model"] == model_name]
        grouped = (
            group.groupby("species_count")[value_column]
            .mean()
            .reset_index()
        )
        ax.plot(
            grouped["species_count"],
            grouped[value_column],
            marker="o",
            linewidth=2,
        )
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Retained partner species")
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_model(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    x_column: str = "measured_fraction",
    xlabel: str = "Measured fraction",
    y_limits: tuple[float, float] | None = (0, 1.02),
) -> None:
    fig, panels = model_panels(metrics)
    legend_entries: dict[str, object] = {}

    for index, model_name in enumerate(sorted(metrics["model"].unique())):
        ax = panels[index]
        model_rows = metrics[metrics["model"] == model_name]
        for species_count, group in model_rows.groupby("species_count"):
            label = f"{species_count} species"
            if species_count == int(model_rows["species_count"].max()):
                label = f"{species_count} species (full)"
            grouped = (
                group.groupby(x_column)[metric]
                .mean()
                .reset_index()
            )
            line, = ax.plot(
                grouped[x_column],
                grouped[metric],
                marker="o",
                linewidth=1.8,
                label=label,
            )
            legend_entries[label] = line

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="lower center",
        ncol=min(5, max(1, len(legend_entries))),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_by_species(
    metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    x_column: str = "measured_fraction",
    xlabel: str = "Measured fraction",
    y_limits: tuple[float, float] | None = (0, 1.02),
) -> None:
    species_counts = sorted(metrics["species_count"].unique())
    columns = min(3, len(species_counts))
    rows = int(np.ceil(len(species_counts) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4 * rows),
        dpi=120,
        squeeze=False,
    )
    panels = axes.ravel()

    for index, species_count in enumerate(species_counts):
        ax = panels[index]
        species_rows = metrics[metrics["species_count"] == species_count]
        for model_name, group in species_rows.groupby("model"):
            grouped = (
                group.groupby(x_column)[metric]
                .mean()
                .reset_index()
            )
            ax.plot(
                grouped[x_column],
                grouped[metric],
                marker="o",
                linewidth=1.8,
                label=model_name,
            )

        ax.set_title(f"{species_count} retained partners", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    for empty in range(len(species_counts), len(panels)):
        panels[empty].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_precision(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    for (species_count, model_name), group in metrics.groupby(
        ["species_count", "model"]
    ):
        grouped = (
            group.groupby("measured_fraction")["suppressor_precision"]
            .mean()
            .reset_index()
        )
        ax.plot(
            grouped["measured_fraction"],
            grouped["suppressor_precision"],
            marker="o",
            linewidth=1.5,
            label=f"{species_count} | {model_name}",
        )

    ax.set_title(
        "Suppressor Precision by Universe Size", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Measured fraction")
    ax.set_ylabel("Suppressor precision")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_scaling_laws(
    summary_path: str,
    output_dir: str,
    target_species: str | None,
    species_ids: list[str] | None,
    species_counts: list[int],
    universes_per_size: int,
    seed: int,
    repeat_count: int,
    train_sizes: list[int | str] | None,
    train_fractions: list[float],
    train_grid: str,
    minimum_train_rows: int,
    models: list[str] | None,
    suppressor_fold: float,
    precision_threshold: float,
    recall_threshold: float,
    auprc_threshold: float,
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    dataset_dir = output_path / "datasets"
    benchmark_dir = output_path / "benchmarks"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    partner_ids = species_ids or infer_partner_ids(summary, target_species)
    dataset_rows = []
    metric_rows = []

    for species_count in species_counts:
        for universe_index in range(universes_per_size):
            universe_seed = seed + species_count * 1000 + universe_index
            rng = np.random.default_rng(universe_seed)
            retained_species = sorted(
                rng.choice(
                    partner_ids,
                    size=min(species_count, len(partner_ids)),
                    replace=False,
                )
            )
            universe_summary = filtered_summary(summary, retained_species)
            if universe_summary.empty:
                continue
            universe_path = (
                dataset_dir / f"species_{species_count}_seed_{universe_seed}.csv"
            )
            universe_summary.to_csv(universe_path, index=False)
            dataset_rows.append(
                dataset_row(
                    universe_seed,
                    species_count,
                    retained_species,
                    universe_summary,
                    suppressor_fold,
                )
            )
            report_dir = benchmark_dir / f"species_{species_count}_seed_{universe_seed}"
            run_benchmarks(
                summary_path=str(universe_path),
                output_dir=str(report_dir),
                test_size=0.25,
                seed=universe_seed,
                species_ids=retained_species,
                target_species=target_species,
                repeat_count=repeat_count,
                train_sizes=train_sizes_for_universe(
                    len(universe_summary),
                    train_sizes,
                    train_fractions,
                    train_grid,
                    minimum_train_rows,
                ),
                suppressor_fold=suppressor_fold,
                buffer_z=1.96,
                concordance_z=1.96,
                suppressor_target_scale="log",
                selected_models=models,
            )
            metric_rows.extend(
                load_metric_outputs(
                    report_dir,
                    universe_seed,
                    species_count,
                    retained_species,
                    len(universe_summary),
                )
            )

    datasets = pd.DataFrame(dataset_rows)
    metrics = pd.DataFrame(metric_rows)
    summary_metrics = summarize_metrics(metrics)
    required = required_row_summary(
        metrics,
        precision_threshold,
        recall_threshold,
        auprc_threshold,
    )

    datasets_path = output_path / "scaling_law_datasets.csv"
    metrics_path = output_path / "scaling_law_metrics.csv"
    summary_path_out = output_path / "scaling_law_summary.csv"
    required_path = output_path / "scaling_law_required_rows.csv"
    write_csv(datasets, datasets_path)
    write_csv(metrics, metrics_path)
    write_csv(summary_metrics, summary_path_out)
    write_csv(required, required_path)
    plot_required_fraction(
        required,
        "precision",
        output_path / "scaling_law_precision_required_fraction.png",
        "Rows Required for Suppressor Precision Threshold",
    )
    plot_required_rows(
        required,
        "precision",
        output_path / "scaling_law_precision_required_rows.png",
        "Rows Required for Suppressor Precision Threshold",
    )
    plot_required_fraction(
        required,
        "auprc",
        output_path / "scaling_law_auprc_required_fraction.png",
        "Rows Required for Suppressor AUPRC Threshold",
    )
    plot_required_rows(
        required,
        "auprc",
        output_path / "scaling_law_auprc_required_rows.png",
        "Rows Required for Suppressor AUPRC Threshold",
    )
    plot_required_fraction(
        required,
        "recall",
        output_path / "scaling_law_recall_required_fraction.png",
        "Rows Required for Suppressor Recall Threshold",
    )
    plot_required_rows(
        required,
        "recall",
        output_path / "scaling_law_recall_required_rows.png",
        "Rows Required for Suppressor Recall Threshold",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_precision",
        output_path / "scaling_law_suppressor_precision_by_model.png",
        "Suppressor Precision by Universe Size",
        "Suppressor precision",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_precision",
        output_path / "scaling_law_suppressor_precision_by_model_rows.png",
        "Suppressor Precision by Universe Size",
        "Suppressor precision",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_precision",
        output_path / "scaling_law_suppressor_precision_by_species.png",
        "Suppressor Precision by Universe Size",
        "Suppressor precision",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_precision",
        output_path / "scaling_law_suppressor_precision_by_species_rows.png",
        "Suppressor Precision by Universe Size",
        "Suppressor precision",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_auprc",
        output_path / "scaling_law_suppressor_auprc_by_model.png",
        "Suppressor AUPRC by Universe Size",
        "Suppressor AUPRC",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_auprc",
        output_path / "scaling_law_suppressor_auprc_by_model_rows.png",
        "Suppressor AUPRC by Universe Size",
        "Suppressor AUPRC",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_class_recall",
        output_path / "scaling_law_suppressor_recall_by_model.png",
        "Suppressor Recall by Universe Size",
        "Suppressor recall",
    )
    plot_metric_by_model(
        metrics,
        "suppressor_class_recall",
        output_path / "scaling_law_suppressor_recall_by_model_rows.png",
        "Suppressor Recall by Universe Size",
        "Suppressor recall",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_auprc",
        output_path / "scaling_law_suppressor_auprc_by_species.png",
        "Suppressor AUPRC by Universe Size",
        "Suppressor AUPRC",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_auprc",
        output_path / "scaling_law_suppressor_auprc_by_species_rows.png",
        "Suppressor AUPRC by Universe Size",
        "Suppressor AUPRC",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_class_recall",
        output_path / "scaling_law_suppressor_recall_by_species.png",
        "Suppressor Recall by Universe Size",
        "Suppressor recall",
    )
    plot_metric_by_species(
        metrics,
        "suppressor_class_recall",
        output_path / "scaling_law_suppressor_recall_by_species_rows.png",
        "Suppressor Recall by Universe Size",
        "Suppressor recall",
        x_column="train_rows",
        xlabel="Measured rows",
    )
    plot_metric_by_model(
        metrics,
        "rmse",
        output_path / "scaling_law_rmse_by_model.png",
        "RMSE by Universe Size",
        "RMSE",
        y_limits=None,
    )
    plot_metric_by_model(
        metrics,
        "rmse",
        output_path / "scaling_law_rmse_by_model_rows.png",
        "RMSE by Universe Size",
        "RMSE",
        x_column="train_rows",
        xlabel="Measured rows",
        y_limits=None,
    )
    plot_metric_by_species(
        metrics,
        "rmse",
        output_path / "scaling_law_rmse_by_species.png",
        "RMSE by Universe Size",
        "RMSE",
        y_limits=None,
    )
    plot_metric_by_species(
        metrics,
        "rmse",
        output_path / "scaling_law_rmse_by_species_rows.png",
        "RMSE by Universe Size",
        "RMSE",
        x_column="train_rows",
        xlabel="Measured rows",
        y_limits=None,
    )

    return datasets_path, metrics_path, summary_path_out, required_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run species-universe scaling-law benchmarks."
    )
    parser.add_argument("summary_path")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/scaling_laws_real")
    parser.add_argument("--target-species")
    parser.add_argument("--species-ids")
    # k=3 and k=4 are omitted by default: the suppressor cutoff sits 2-fold below each
    # universe's own median, and universes that small hold almost no such community
    # (prevalence 2.9% and 4.0%), so every suppressor metric there is undefined.
    parser.add_argument("--species-counts", default="5,6,7,8,9,10,11")
    parser.add_argument("--universes-per-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--train-sizes",
        help="Optional comma-separated row-count budgets. Overrides --train-grid.",
    )
    parser.add_argument(
        "--train-grid",
        default="log",
        choices=("log", "fraction"),
        help=(
            "log: geometric row budgets, constant relative resolution at every k. "
            "fraction: the --train-fractions grid, which is coarse at large k."
        ),
    )
    parser.add_argument(
        "--minimum-train-rows",
        type=int,
        default=4,
        help="Smallest budget in the log grid.",
    )
    parser.add_argument(
        "--train-fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.0",
        help="Measured-fraction budgets, used only when --train-grid fraction.",
    )
    parser.add_argument(
        "--models", default="ridge_pairwise,random_forest,hist_gradient_boosting"
    )
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument("--precision-threshold", type=float, default=0.8)
    parser.add_argument("--recall-threshold", type=float, default=0.8)
    parser.add_argument("--auprc-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_path, metrics_path, summary_path, required_path = run_scaling_laws(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        target_species=args.target_species,
        species_ids=parse_species_ids(args.species_ids),
        species_counts=parse_csv_ints(args.species_counts),
        universes_per_size=args.universes_per_size,
        seed=args.seed,
        repeat_count=args.repeat_count,
        train_sizes=parse_train_sizes(args.train_sizes) if args.train_sizes else None,
        train_fractions=parse_csv_floats(args.train_fractions),
        train_grid=args.train_grid,
        minimum_train_rows=args.minimum_train_rows,
        models=parse_model_names(args.models),
        suppressor_fold=args.suppressor_fold,
        precision_threshold=args.precision_threshold,
        recall_threshold=args.recall_threshold,
        auprc_threshold=args.auprc_threshold,
    )

    print(f"Wrote scaling-law datasets to {datasets_path}")
    print(f"Wrote scaling-law metrics to {metrics_path}")
    print(f"Wrote scaling-law summary to {summary_path}")
    print(f"Wrote scaling-law required rows to {required_path}")


if __name__ == "__main__":
    main()
