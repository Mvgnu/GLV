#!/usr/bin/env python3
"""Prepare real-world community measurements for the GLV_ML benchmark."""

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def design_id(path: Path) -> int:
    match = re.search(r"_D(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse design id from {path.name}")

    return int(match.group(1))


def parse_combination(value) -> tuple[int, ...]:
    if pd.isna(value):
        return tuple()

    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Combination must be a list, got {value!r}")

    return tuple(sorted(int(species_id) for species_id in parsed))


def species_name(species_id: int, species_prefix: str) -> str:
    return f"{species_prefix}_{species_id:03d}"


def community_name(combination: tuple[int, ...], species_prefix: str) -> str:
    return ";".join(species_name(species_id, species_prefix) for species_id in combination)


def matched_files(input_dir: Path) -> list[tuple[int, Path, Path]]:
    biolum_dir = input_dir / "biolum"
    od_dir = input_dir / "OD"
    biolum_files = {design_id(path): path for path in biolum_dir.glob("*.csv")}
    od_files = {design_id(path): path for path in od_dir.glob("*.csv")}
    missing_biolum = sorted(set(od_files) - set(biolum_files))
    missing_od = sorted(set(biolum_files) - set(od_files))

    if missing_biolum or missing_od:
        raise ValueError(
            f"Unmatched designs, missing biolum={missing_biolum}, missing OD={missing_od}"
        )

    return [(design, biolum_files[design], od_files[design]) for design in sorted(biolum_files)]


def load_measurements(input_dir: Path, species_prefix: str) -> pd.DataFrame:
    rows = []

    for design, biolum_path, od_path in matched_files(input_dir):
        biolum = pd.read_csv(biolum_path).rename(columns={"val": "biolum_raw"})
        od = pd.read_csv(od_path).rename(columns={"val": "od600_raw"})
        # OD and biolum are paired endpoint measurements for the same wells.
        merged = biolum.merge(
            od[["Well_No", "Combination", "od600_raw"]],
            on=["Well_No", "Combination"],
            validate="one_to_one",
        )
        merged["design_id"] = design
        merged["combination_tuple"] = merged["Combination"].map(parse_combination)
        merged["partner_count"] = merged["combination_tuple"].map(len)
        merged["is_control"] = merged["partner_count"].eq(0)
        merged["community"] = merged["combination_tuple"].map(
            lambda combination: community_name(combination, species_prefix)
        )
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def add_background_corrected_columns(measurements: pd.DataFrame) -> pd.DataFrame:
    controls = measurements[measurements["is_control"]]
    if controls.empty:
        raise ValueError("Expected [] control wells for background correction")

    global_biolum_background = float(controls["biolum_raw"].median())
    global_od_background = float(controls["od600_raw"].median())
    biolum_by_design = controls.groupby("design_id")["biolum_raw"].median()
    od_by_design = controls.groupby("design_id")["od600_raw"].median()

    measurements = measurements.copy()
    # Prefer same-design controls; D17 currently falls back to the global control median.
    measurements["biolum_background"] = measurements["design_id"].map(biolum_by_design)
    measurements["od600_background"] = measurements["design_id"].map(od_by_design)
    measurements["biolum_background"] = measurements["biolum_background"].fillna(
        global_biolum_background
    )
    measurements["od600_background"] = measurements["od600_background"].fillna(
        global_od_background
    )
    measurements["biolum_background_corrected"] = np.maximum(
        measurements["biolum_raw"] - measurements["biolum_background"],
        0.0,
    )
    measurements["od600_background_corrected"] = np.maximum(
        measurements["od600_raw"] - measurements["od600_background"],
        0.0,
    )
    # Log transform keeps high pathogen-signal wells from dominating regression loss.
    measurements["pathogen_signal_log1p"] = np.log1p(
        measurements["biolum_background_corrected"]
    )
    measurements["pathogen_signal_raw"] = measurements["biolum_background_corrected"]

    return measurements


def pathogen_signal_column(target_transform: str) -> str:
    if target_transform == "log1p":
        return "pathogen_signal_log1p"
    if target_transform == "raw":
        return "pathogen_signal_raw"

    raise ValueError("target_transform must be 'log1p' or 'raw'")


def summarize_communities(
    measurements: pd.DataFrame,
    target_species: str,
    species_prefix: str,
    target_transform: str,
) -> pd.DataFrame:
    non_control = measurements[~measurements["is_control"]].copy()
    grouped = non_control.groupby("combination_tuple", sort=True)
    rows = []
    signal_column = pathogen_signal_column(target_transform)

    for combination, group in grouped:
        rows.append({
            "community": community_name(combination, species_prefix),
            "partner_count": len(combination),
            "community_size": len(combination) + 1,
            "target_species": target_species,
            "target_transform": target_transform,
            "final_target_biomass": float(group[signal_column].mean()),
            "biolum_mean": float(group["biolum_raw"].mean()),
            "biolum_std": float(group["biolum_raw"].std(ddof=0)),
            "biolum_background_corrected_mean": float(
                group["biolum_background_corrected"].mean()
            ),
            "biolum_background_corrected_std": float(
                group["biolum_background_corrected"].std(ddof=0)
            ),
            "pathogen_signal_std": float(group[signal_column].std(ddof=0)),
            "pathogen_signal_log1p_mean": float(group["pathogen_signal_log1p"].mean()),
            "pathogen_signal_log1p_std": float(group["pathogen_signal_log1p"].std(ddof=0)),
            "pathogen_signal_raw_mean": float(group["pathogen_signal_raw"].mean()),
            "pathogen_signal_raw_std": float(group["pathogen_signal_raw"].std(ddof=0)),
            "od600_mean": float(group["od600_raw"].mean()),
            "od600_std": float(group["od600_raw"].std(ddof=0)),
            "od600_background_corrected_mean": float(
                group["od600_background_corrected"].mean()
            ),
            "replicate_count": int(len(group)),
            "design_ids": ";".join(str(value) for value in sorted(group["design_id"].unique())),
        })

    return pd.DataFrame(rows)


def plot_noise_diagnostics(
    summary: pd.DataFrame,
    output_dir: Path,
    target_transform: str,
) -> Path:
    """Show that replicate noise concentrates in the suppressor tail.

    Four panels make the case that confident *within-suppressor* ranking is
    not supported by the measurements:
      A. all communities sorted by biomass, with a 95% CI band (overview);
      B. the strongest 50 suppressors with per-community 95% CIs (they overlap,
         so there is no statistically resolvable "single best");
      C. replicate SD vs mean on the log (modelled) scale - noise rises as
         biomass falls, so the suppressors are the noisiest wells;
      D. replicate SD vs mean on the raw scale - noise instead grows with
         signal, which is why the target is log-transformed before modelling.

    Panels A/B use +-1 standard error of the mean (SE = SD / sqrt(replicates),
    n=3 here) rather than a wider 95% CI: SE is the conservative choice, so
    overlapping SE bars already show two community means are indistinguishable,
    and SE shrinks only with more replicates - which is the experimental point.
    Panels C/D plot the raw replicate SD directly.
    """
    data = summary.sort_values("final_target_biomass").reset_index(drop=True)
    target_label = (
        "log target biomass"
        if target_transform == "log1p"
        else "raw background-corrected biolum"
    )
    biomass = data["final_target_biomass"].to_numpy()
    sd_log = data["pathogen_signal_std"].to_numpy()
    se = sd_log / np.sqrt(data["replicate_count"].to_numpy())
    rank = np.arange(1, len(data) + 1)
    cutoff = float(np.median(biomass)) - float(np.log(2))
    suppressor = biomass <= cutoff
    n_suppressor = int(suppressor.sum())

    blue, orange, red = "#4c72b0", "#dd8452", "#c44e52"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A: overview, sorted, with 95% CI band.
    ax = axes[0, 0]
    ax.fill_between(rank, biomass - se, biomass + se, color=blue, alpha=0.25,
                    linewidth=0, label="± 1 SE")
    ax.plot(rank, biomass, color=blue, linewidth=0.8)
    ax.axhline(cutoff, color="#555555", linestyle="--", linewidth=1, label="2x cutoff")
    ax.axvspan(0, n_suppressor, color=orange, alpha=0.10)
    ax.set_xlabel("community rank (1 = lowest biomass)")
    ax.set_ylabel(target_label)
    ax.set_title("Rank-ordered target biomass with standard-error bands")
    ax.legend(loc="lower right", fontsize=8)

    # B: strongest suppressors, individual 95% CIs.
    ax = axes[0, 1]
    k = min(50, len(data))
    ax.errorbar(rank[:k], biomass[:k], yerr=se[:k], fmt="o", markersize=3,
                color=orange, ecolor=orange, elinewidth=0.8, capsize=2)
    ax.axhline(cutoff, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel(f"community rank (bottom {k})")
    ax.set_ylabel(target_label)
    ax.set_title(f"Per-community standard error for the {k} lowest-biomass communities")

    # C: replicate SD vs mean on the log scale.
    ax = axes[1, 0]
    ax.scatter(biomass[~suppressor], sd_log[~suppressor], s=6, alpha=0.25,
               color=blue, edgecolors="none", label="non-suppressor")
    ax.scatter(biomass[suppressor], sd_log[suppressor], s=6, alpha=0.40,
               color=orange, edgecolors="none", label="suppressor")
    bins = np.linspace(biomass.min(), biomass.max(), 13)
    idx = np.clip(np.digitize(biomass, bins), 1, len(bins) - 1)
    centers = [biomass[idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
    med = [np.median(sd_log[idx == b]) for b in range(1, len(bins)) if (idx == b).any()]
    ax.plot(centers, med, color=red, linewidth=2, label="binned median")
    ax.axvline(cutoff, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{target_label} (mean)")
    ax.set_ylabel(f"replicate SD ({target_transform} target)")
    ax.set_title(f"Replicate standard deviation versus mean ({target_transform} target)")
    ax.legend(loc="upper right", fontsize=8)

    # D: replicate SD vs mean on the raw scale (justifies the log transform).
    ax = axes[1, 1]
    raw_mean = data["biolum_mean"].to_numpy()
    raw_sd = data["biolum_std"].to_numpy()
    ax.scatter(raw_mean[~suppressor], raw_sd[~suppressor], s=6, alpha=0.25,
               color=blue, edgecolors="none", label="non-suppressor")
    ax.scatter(raw_mean[suppressor], raw_sd[suppressor], s=6, alpha=0.40,
               color=orange, edgecolors="none", label="suppressor")
    rbins = np.linspace(raw_mean.min(), raw_mean.max(), 13)
    ridx = np.clip(np.digitize(raw_mean, rbins), 1, len(rbins) - 1)
    rcenters = [raw_mean[ridx == b].mean() for b in range(1, len(rbins)) if (ridx == b).any()]
    rmed = [np.median(raw_sd[ridx == b]) for b in range(1, len(rbins)) if (ridx == b).any()]
    ax.plot(rcenters, rmed, color=red, linewidth=2, label="binned median")
    ax.set_xlabel("raw biolum (mean)")
    ax.set_ylabel("raw biolum SD")
    ax.set_title("Replicate standard deviation versus mean (linear scale)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Replicate measurement noise across the community biomass gradient",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = output_dir / "rw_suppressor_noise.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_outputs(
    measurements: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    target_species: str,
    target_transform: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = measurements[measurements["is_control"]].copy()
    replicate_output = measurements.copy()
    replicate_output["combination"] = replicate_output["combination_tuple"].map(list)
    replicate_output = replicate_output.drop(columns=["combination_tuple"])

    replicate_path = output_dir / "rw_replicates.csv"
    summary_path = output_dir / "rw_summary.csv"
    controls_path = output_dir / "rw_controls.csv"
    qc_path = output_dir / "rw_qc.json"

    replicate_output.to_csv(replicate_path, index=False)
    summary.to_csv(summary_path, index=False)
    controls.drop(columns=["combination_tuple"]).to_csv(controls_path, index=False)

    replicate_counts = summary["replicate_count"].value_counts().sort_index()
    qc = {
        "target_species": target_species,
        "target_transform": target_transform,
        "replicate_rows": int(len(measurements)),
        "summary_rows": int(len(summary)),
        "control_rows": int(len(controls)),
        "unique_noncontrol_communities": int(summary["community"].nunique()),
        "partner_count_distribution": {
            str(key): int(value)
            for key, value in summary["partner_count"].value_counts().sort_index().items()
        },
        "replicate_count_distribution": {
            str(key): int(value)
            for key, value in replicate_counts.items()
        },
        "global_biolum_control_median": float(controls["biolum_raw"].median()),
        "global_od600_control_median": float(controls["od600_raw"].median()),
        "designs_without_controls": [
            int(design)
            for design in sorted(
                set(measurements["design_id"]) - set(controls["design_id"])
            )
        ],
        "outputs": {
            "replicates": str(replicate_path),
            "summary": str(summary_path),
            "controls": str(controls_path),
            "qc": str(qc_path),
        },
    }
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")

    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real-world pathogen-signal community data for ML benchmarks."
    )
    parser.add_argument(
        "--input-dir",
        default="GLV_ML/outputs/inputs/real_world/13022025_data",
        help="Directory containing matched biolum and OD subdirectories.",
    )
    parser.add_argument("--output-dir", default="GLV_ML/outputs/real_world/log")
    parser.add_argument("--species-prefix", default="sp")
    parser.add_argument("--target-species", default="pathogen")
    parser.add_argument(
        "--target-transform",
        choices=["log1p", "raw"],
        default="log1p",
        help="Transform used for final_target_biomass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measurements = load_measurements(Path(args.input_dir), args.species_prefix)
    measurements = add_background_corrected_columns(measurements)
    measurements["pathogen_signal"] = measurements[
        pathogen_signal_column(args.target_transform)
    ]
    summary = summarize_communities(
        measurements,
        args.target_species,
        args.species_prefix,
        args.target_transform,
    )
    qc = write_outputs(
        measurements,
        summary,
        Path(args.output_dir),
        args.target_species,
        args.target_transform,
    )
    noise_plot = plot_noise_diagnostics(summary, Path(args.output_dir), args.target_transform)

    print(f"Wrote {qc['summary_rows']} community summaries to {qc['outputs']['summary']}")
    print(f"Wrote {qc['replicate_rows']} replicate rows to {qc['outputs']['replicates']}")
    print(f"Wrote {qc['control_rows']} control rows to {qc['outputs']['controls']}")
    print(f"Wrote noise diagnostic plot to {noise_plot}")


if __name__ == "__main__":
    main()
