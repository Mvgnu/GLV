#!/usr/bin/env python3
"""Compare model reports across target transforms such as log1p and raw."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml_benchmark import write_csv


EXCLUDED_MODELS = {"mean_baseline", "gnn"}
COMPARISON_METRICS = [
    ("spearman", "Spearman"),
    ("suppressor_precision", "Suppressor precision"),
    ("suppressor_class_recall", "Suppressor recall"),
    ("suppressor_auprc", "Suppressor AUPRC"),
    ("rmse_norm", "Normalized RMSE"),
]
# Metrics bounded to [0, 1] keep a fixed axis; others (normalized RMSE) autoscale.
UNIT_INTERVAL_METRICS = {
    "spearman",
    "suppressor_precision",
    "suppressor_class_recall",
    "suppressor_auprc",
}


def load_report(report_dir: str, transform: str) -> pd.DataFrame:
    report_path = Path(report_dir)
    biomass = pd.read_csv(report_path / "target_biomass_metrics.csv")
    # RMSE is on the transform's own scale, so normalize by target spread to make
    # log1p and raw comparable. NaN where target_std is non-positive (degenerate).
    biomass["rmse_norm"] = biomass["rmse"] / biomass["target_std"].mask(
        biomass["target_std"] <= 0
    )
    classification = pd.read_csv(report_path / "target_suppressor_classification.csv")
    merged = biomass.merge(
        classification[
            [
                "model",
                "feature_set",
                "train_rows",
                "suppressor_precision",
                "suppressor_class_recall",
                "suppressor_auprc",
                "concordance_overall",
                "concordance_within_suppressors",
            ]
        ],
        on=["model", "feature_set", "train_rows"],
        how="left",
    )
    merged.insert(0, "target_transform", transform)
    return merged


def plot_metric_by_model(
    comparison: pd.DataFrame,
    metric: str,
    label: str,
    output_path: Path,
) -> None:
    models = sorted(comparison["model"].unique())
    columns = min(3, len(models))
    rows = int((len(models) + columns - 1) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4 * rows),
        dpi=120,
        squeeze=False,
    )
    panels = axes.ravel()

    for index, model_name in enumerate(models):
        ax = panels[index]
        model_rows = comparison[comparison["model"] == model_name]
        for target_transform, group in model_rows.groupby("target_transform"):
            group = group.sort_values("train_rows")
            ax.plot(
                group["train_rows"],
                group[metric],
                marker="o",
                linewidth=2,
                label=target_transform,
            )

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training rows")
        ax.set_ylabel(label)
        if metric in UNIT_INTERVAL_METRICS:
            ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    for empty_index in range(len(models), len(panels)):
        panels[empty_index].axis("off")

    fig.suptitle(f"Raw vs log1p target comparison: {label}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_metrics(comparison: pd.DataFrame, model_name: str, output_path: Path) -> None:
    model_rows = comparison[comparison["model"] == model_name]
    columns = min(3, len(COMPARISON_METRICS))
    rows = int((len(COMPARISON_METRICS) + columns - 1) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4 * rows),
        dpi=120,
        squeeze=False,
    )
    panels = axes.ravel()

    for index, (metric, label) in enumerate(COMPARISON_METRICS):
        ax = panels[index]
        for target_transform, group in model_rows.groupby("target_transform"):
            group = group.sort_values("train_rows")
            ax.plot(
                group["train_rows"],
                group[metric],
                marker="o",
                linewidth=2,
                label=target_transform,
            )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training rows")
        ax.set_ylabel(label)
        if metric in UNIT_INTERVAL_METRICS:
            ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

    for empty_index in range(len(COMPARISON_METRICS), len(panels)):
        panels[empty_index].axis("off")

    fig.suptitle(f"Raw vs log1p target comparison: {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def suppressor_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spearman",
        "suppressor_precision",
        "suppressor_class_recall",
        "suppressor_auprc",
        "rmse_norm",
    ]
    id_columns = ["model", "feature_set", "train_rows"]
    wide = comparison.pivot_table(
        index=id_columns,
        columns="target_transform",
        values=metrics,
        aggfunc="first",
    )
    rows = []
    for index_values, row in wide.iterrows():
        output = dict(zip(id_columns, index_values))
        for metric in metrics:
            log_value = row.get((metric, "log1p"))
            raw_value = row.get((metric, "raw"))
            output[f"{metric}_log1p"] = log_value
            output[f"{metric}_raw"] = raw_value
            output[f"{metric}_raw_minus_log1p"] = raw_value - log_value
        rows.append(output)
    return pd.DataFrame(rows).sort_values(["model", "feature_set", "train_rows"])


def compare_target_transforms(
    log_report_dir: str,
    raw_report_dir: str,
    output_dir: str,
) -> tuple[Path, Path, list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat(
        [
            load_report(log_report_dir, "log1p"),
            load_report(raw_report_dir, "raw"),
        ],
        ignore_index=True,
    )
    comparison = comparison[~comparison["model"].isin(EXCLUDED_MODELS)].reset_index(drop=True)
    comparison_path = output_path / "target_transform_comparison.csv"
    write_csv(comparison, comparison_path)
    delta_path = output_path / "target_transform_suppressor_deltas.csv"
    write_csv(suppressor_deltas(comparison), delta_path)

    plot_paths = []
    for metric, label in COMPARISON_METRICS:
        path = output_path / f"target_transform_comparison_{metric}.png"
        plot_metric_by_model(comparison, metric, label, path)
        plot_paths.append(path)

    for model_name in sorted(comparison["model"].unique()):
        safe_name = model_name.replace("/", "_")
        path = output_path / f"target_transform_comparison_{safe_name}.png"
        plot_model_metrics(comparison, model_name, path)
        plot_paths.append(path)

    return comparison_path, delta_path, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare log1p and raw target-transform benchmark reports."
    )
    parser.add_argument("--log-report-dir", default="GLV_ML/outputs/benchmarks/ml/rw_log")
    parser.add_argument("--raw-report-dir", default="GLV_ML/outputs/benchmarks/ml/rw_raw")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/real_world/comparisons/target_transform")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path, delta_path, plot_paths = compare_target_transforms(
        log_report_dir=args.log_report_dir,
        raw_report_dir=args.raw_report_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote target-transform comparison to {comparison_path}")
    print(f"Wrote suppressor deltas to {delta_path}")
    for path in plot_paths:
        print(f"Wrote plot to {path}")


if __name__ == "__main__":
    main()
