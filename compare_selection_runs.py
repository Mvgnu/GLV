#!/usr/bin/env python3
"""Compare model-dependent (active_learning) vs model-independent (selection_baselines) runs.

Ingests only already-materialized summary CSVs. It does not train models or select
communities. Both inputs share the active-learning rounds/summary schema (the ``strategy``
column holds the method name), so this just tags each by family and unions them, then plots
the two discovery/learning axes with one line per family|method.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml_benchmark import write_csv


COMPARISON_COLUMNS = [
    "global_best_gap_fraction",
    "global_best_gap",
    "rmse",
    "spearman",
    "suppressor_auprc",
]


def load_summary(path: str | None, method_family: str) -> list[dict[str, object]]:
    if path is None:
        return []
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        record = {
            "method_family": method_family,
            "model": row["model"],
            "strategy": row["strategy"],
            "measured_count": int(row["measured_count"]),
            "runs": int(row.get("runs", 1)),
        }
        for column in COMPARISON_COLUMNS:
            record[column] = float(row.get(column, float("nan")))
        rows.append(record)
    return rows


def rank_strategies(
    grouped: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> pd.Series:
    """Score each strategy by its mean metric over the budget grid the family shares.

    Baselines cover uneven budget ranges (some stop early), so a naive mean over each
    strategy's own points would penalise the ones evaluated only at small budgets. We
    restrict scoring to the measured_counts common to every strategy in the family, which
    makes the comparison apples-to-apples. Returns a Series (index=strategy) sorted best
    first; the value is the aggregate score used for ranking.
    """
    per_count = grouped.pivot_table(index="strategy", columns="measured_count", values=metric)
    common = per_count.dropna(axis=1)
    scored = (common if not common.empty else per_count).mean(axis=1)
    return scored.sort_values(ascending=not higher_is_better)


def plot_metric(
    comparison: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_n: int,
    higher_is_better: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if metric not in comparison.columns or comparison[metric].isna().all():
        return
    # One panel per family. To keep the comparison readable we only draw the top_n
    # strategies per family (8 strategies each otherwise overloads the panel). The model
    # dimension is collapsed (mean over models) -- exact for model-independent methods,
    # where the metric does not depend on the model, and a per-model mean for the
    # model-dependent family.
    families = sorted(comparison["method_family"].unique())
    fig, axes = plt.subplots(
        1, len(families), figsize=(7 * len(families), 5), dpi=120, sharey=True, squeeze=False
    )
    panels = axes.ravel()
    for index, family in enumerate(families):
        ax = panels[index]
        grouped = (
            comparison[comparison["method_family"] == family]
            .groupby(["strategy", "measured_count"])[metric]
            .mean()
            .reset_index()
        )
        ranking = rank_strategies(grouped, metric, higher_is_better)
        kept = ranking.head(top_n)
        for strategy, score in kept.items():
            group = grouped[grouped["strategy"] == strategy].sort_values("measured_count")
            ax.plot(
                group["measured_count"],
                group[metric],
                marker="o",
                linewidth=1.8,
                label=f"{strategy} ({score:.3g})",
            )
        panel_title = family.replace("_", "-")
        dropped = len(ranking) - len(kept)
        if dropped > 0:
            panel_title += f"  (top {len(kept)} of {len(ranking)})"
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Measured communities")
        if index == 0:
            ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(
            title="strategy (mean over budgets)",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            fontsize=8,
            title_fontsize=8,
        )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_combined(
    comparison: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_n: int,
    higher_is_better: bool = False,
    x_max: float | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Both families overlaid on one axes for a head-to-head read.

    Restricted to measured_count <= x_max because the families only overlap on the low
    end (model-dependent stops early), and ranking is done within that same window so the
    kept strategies are the best over the range actually plotted. Family is encoded by
    line style (first family solid, second dashed) so the two are distinguishable at a
    glance; colour still separates strategies.
    """
    if metric not in comparison.columns or comparison[metric].isna().all():
        return
    data = comparison if x_max is None else comparison[comparison["measured_count"] <= x_max]
    families = sorted(data["method_family"].unique())
    linestyles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    for family_index, family in enumerate(families):
        grouped = (
            data[data["method_family"] == family]
            .groupby(["strategy", "measured_count"])[metric]
            .mean()
            .reset_index()
        )
        ranking = rank_strategies(grouped, metric, higher_is_better)
        kept = ranking.head(top_n)
        style = linestyles[family_index % len(linestyles)]
        for strategy, score in kept.items():
            group = grouped[grouped["strategy"] == strategy].sort_values("measured_count")
            ax.plot(
                group["measured_count"],
                group[metric],
                marker="o",
                linewidth=1.8,
                linestyle=style,
                label=f"{family.replace('_', '-')} · {strategy} ({score:.3g})",
            )
    ax.set_xlabel("Measured communities")
    ax.set_ylabel(ylabel)
    if x_max is not None:
        ax.set_xlim(right=x_max)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        title="method-family · strategy (mean over budgets)",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=8,
        title_fontsize=8,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compare_runs(
    active_summary: str | None,
    selection_summary: str | None,
    output_dir: str,
    top_n: int = 3,
    combined_x_max: float | None = 450,
    plot_combined: bool = True,
) -> tuple[Path, list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = load_summary(active_summary, "model_dependent") + load_summary(
        selection_summary, "model_independent"
    )
    comparison = pd.DataFrame(rows)

    comparison_path = output_path / "selection_comparison.csv"
    write_csv(comparison, comparison_path)

    plot_paths = []
    if not comparison.empty:
        gap_path = output_path / "selection_comparison_gap.png"
        plot_metric(
            comparison,
            "global_best_gap_fraction",
            gap_path,
            "Suppressor Discovery: model-dependent vs model-independent",
            "Relative gap to best suppressor",
            top_n=top_n,
            higher_is_better=False,
        )
        plot_paths.append(gap_path)
        rmse_path = output_path / "selection_comparison_rmse.png"
        plot_metric(
            comparison,
            "rmse",
            rmse_path,
            "Landscape Learning: model-dependent vs model-independent",
            "RMSE",
            top_n=top_n,
            higher_is_better=False,
        )
        plot_paths.append(rmse_path)

        if plot_combined:
            gap_combined_path = output_path / "selection_comparison_gap_combined.png"
            plot_metric_combined(
                comparison,
                "global_best_gap_fraction",
                gap_combined_path,
                "Suppressor Discovery: model-dependent vs model-independent",
                "Relative gap to best suppressor",
                top_n=top_n,
                higher_is_better=False,
                x_max=combined_x_max,
            )
            plot_paths.append(gap_combined_path)
            rmse_combined_path = output_path / "selection_comparison_rmse_combined.png"
            plot_metric_combined(
                comparison,
                "rmse",
                rmse_combined_path,
                "Landscape Learning: model-dependent vs model-independent",
                "RMSE",
                top_n=top_n,
                higher_is_better=False,
                x_max=combined_x_max,
            )
            plot_paths.append(rmse_combined_path)

    return comparison_path, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare materialized active-learning and selection-baseline summaries."
    )
    parser.add_argument("--active-summary")
    parser.add_argument("--selection-summary")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/benchmarks/selection_comparison")
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Strategies per family to plot, ranked by mean metric over the shared budget grid.",
    )
    parser.add_argument(
        "--combined-x-max",
        type=float,
        default=450,
        help="Cap the overlaid single-axes plot at this many measured communities "
        "(families only overlap on the low end).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path, plot_paths = compare_runs(
        active_summary=args.active_summary,
        selection_summary=args.selection_summary,
        output_dir=args.output_dir,
        top_n=args.top_n,
        combined_x_max=args.combined_x_max,
    )
    print(f"Wrote selection comparison to {comparison_path}")
    for path in plot_paths:
        print(f"Wrote comparison plot to {path}")


if __name__ == "__main__":
    main()
