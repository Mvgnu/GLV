#!/usr/bin/env python3
"""Calibrate real-world assay noise and apply it to GLV simulation summaries."""

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_benchmark import (
    load_dataset,
    parse_species_ids,
    ridge_pairwise_coefficient_rows,
)


@dataclass(frozen=True)
class AssayNoiseModel:
    target_scale: str
    epsilon: float
    log_sd_intercept: float
    log_sd_slope: float
    residual_log_sd: float
    observed_mean_min: float
    observed_mean_max: float
    observed_sd_min: float
    observed_sd_max: float
    replicate_count: int


def fit_assay_noise_model(summary: pd.DataFrame) -> tuple[AssayNoiseModel, pd.DataFrame]:
    """Fit log(SD) as a linear function of mean target biomass."""
    required = {"final_target_biomass", "pathogen_signal_std", "replicate_count"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    data = summary.dropna(subset=["final_target_biomass", "pathogen_signal_std"]).copy()
    data = data[data["pathogen_signal_std"] > 0]
    epsilon = max(float(data["pathogen_signal_std"].quantile(0.01)) * 0.1, 1e-6)
    means = data["final_target_biomass"].to_numpy(dtype=float)
    log_sd = np.log(data["pathogen_signal_std"].to_numpy(dtype=float) + epsilon)
    design = np.column_stack([np.ones(len(data)), means])
    intercept, slope = np.linalg.lstsq(design, log_sd, rcond=None)[0]
    fitted_log_sd = design @ np.array([intercept, slope])
    residual = log_sd - fitted_log_sd

    fitted = data.copy()
    fitted["fitted_pathogen_signal_std"] = np.exp(fitted_log_sd)
    fitted["noise_model_residual"] = residual

    model = AssayNoiseModel(
        target_scale=str(summary.get("target_transform", pd.Series(["log1p"])).iloc[0]),
        epsilon=epsilon,
        log_sd_intercept=float(intercept),
        log_sd_slope=float(slope),
        residual_log_sd=float(np.std(residual, ddof=0)),
        observed_mean_min=float(np.min(means)),
        observed_mean_max=float(np.max(means)),
        observed_sd_min=float(data["pathogen_signal_std"].min()),
        observed_sd_max=float(data["pathogen_signal_std"].max()),
        replicate_count=int(round(float(data["replicate_count"].median()))),
    )
    return model, fitted


def predict_assay_sd(model: AssayNoiseModel, means: np.ndarray) -> np.ndarray:
    clipped = np.clip(means, model.observed_mean_min, model.observed_mean_max)
    predicted = np.exp(model.log_sd_intercept + model.log_sd_slope * clipped)
    return np.clip(predicted, model.observed_sd_min, model.observed_sd_max)


def sample_assay_sd(
    model: AssayNoiseModel,
    means: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    base_sd = predict_assay_sd(model, means)
    # Residual scatter preserves the observed community-to-community noise spread.
    residual = rng.normal(0.0, model.residual_log_sd, size=len(means))
    sampled = base_sd * np.exp(residual)
    return np.clip(sampled, model.observed_sd_min, model.observed_sd_max)


def quantile_map(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map latent simulation values onto the empirical assay target distribution."""
    order = np.argsort(values, kind="mergesort")
    quantiles = np.linspace(0.0, 1.0, len(values))
    reference_quantiles = np.quantile(reference, quantiles)
    mapped = np.empty(len(values), dtype=float)
    mapped[order] = reference_quantiles
    return mapped


def apply_assay_noise(
    simulation_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    model: AssayNoiseModel,
    target_species: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary = simulation_summary.copy()
    latent = summary["final_target_biomass"].to_numpy(dtype=float)
    reference = reference_summary["final_target_biomass"].to_numpy(dtype=float)
    expected_mean = quantile_map(latent, reference)
    expected_sd = sample_assay_sd(model, expected_mean, rng)

    replicate_rows = []
    noisy_means = []
    noisy_sds = []
    for row_index, (mean, sd) in enumerate(zip(expected_mean, expected_sd, strict=True)):
        replicates = rng.normal(
            loc=mean,
            scale=sd,
            size=model.replicate_count,
        )
        replicates = np.clip(
            replicates,
            model.observed_mean_min,
            model.observed_mean_max,
        )
        noisy_means.append(float(np.mean(replicates)))
        noisy_sds.append(float(np.std(replicates, ddof=0)))
        for replicate_index, value in enumerate(replicates, start=1):
            replicate_rows.append({
                "row_index": row_index,
                "replicate": replicate_index,
                "community": summary["community"].iloc[row_index],
                "partner_count": int(summary["partner_count"].iloc[row_index]),
                "target_species": target_species,
                "latent_target_biomass": float(latent[row_index]),
                "assay_mean_expected": float(mean),
                "assay_sd_expected": float(sd),
                "observed_target_biomass": float(value),
            })

    summary["latent_target_biomass"] = latent
    summary["assay_mean_expected"] = expected_mean
    summary["assay_sd_expected"] = expected_sd
    summary["final_target_biomass"] = noisy_means
    summary["pathogen_signal_std"] = noisy_sds
    summary["replicate_count"] = model.replicate_count
    summary["target_species"] = target_species
    summary["target_transform"] = f"{model.target_scale}_assay_noise"

    return summary, pd.DataFrame(replicate_rows)


def matched_context_effects(
    dataset,
    target_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate species marginals and pair epistasis from matched communities."""
    values = dataset.target_biomass if target_values is None else target_values
    community_values = {
        tuple(np.flatnonzero(presence)): float(value)
        for presence, value in zip(
            dataset.presence,
            values,
            strict=True,
        )
    }
    main_rows = []
    for species_index, species in enumerate(dataset.partner_ids):
        effects = []
        for context, value in community_values.items():
            if species_index in context:
                continue
            augmented = tuple(sorted((*context, species_index)))
            if augmented in community_values:
                effects.append(community_values[augmented] - value)
        coefficient = float(np.mean(effects))
        main_rows.append({
            "feature_type": "main_effect",
            "species_a": species,
            "species_b": "",
            "coefficient": coefficient,
            "abs_coefficient": abs(coefficient),
            "context_sd": float(np.std(effects)),
            "negative_effect_rate": float(np.mean(np.asarray(effects) < 0)),
            "comparisons": len(effects),
        })

    pair_rows = []
    for first_index, second_index in itertools.combinations(
        range(len(dataset.partner_ids)),
        2,
    ):
        effects = []
        for context, value in community_values.items():
            if first_index in context or second_index in context:
                continue
            first = tuple(sorted((*context, first_index)))
            second = tuple(sorted((*context, second_index)))
            pair = tuple(sorted((*context, first_index, second_index)))
            if (
                first in community_values
                and second in community_values
                and pair in community_values
            ):
                effects.append(
                    community_values[pair]
                    - community_values[first]
                    - community_values[second]
                    + value
                )
        coefficient = float(np.mean(effects))
        pair_rows.append({
            "feature_type": "pairwise",
            "species_a": dataset.partner_ids[first_index],
            "species_b": dataset.partner_ids[second_index],
            "coefficient": coefficient,
            "abs_coefficient": abs(coefficient),
            "context_sd": float(np.std(effects)),
            "negative_effect_rate": float(np.mean(np.asarray(effects) < 0)),
            "comparisons": len(effects),
        })

    return pd.DataFrame(main_rows), pd.DataFrame(pair_rows)


def export_full_data_effects(
    summary_path: str,
    target_species: str | None,
    species_ids: list[str] | None,
    seed: int,
    output_dir: Path,
) -> dict[str, Path]:
    dataset = load_dataset(summary_path, target_species, species_ids)
    train_indices = np.arange(len(dataset.target_biomass), dtype=int)
    rows = ridge_pairwise_coefficient_rows(
        dataset,
        train_indices,
        seed,
        len(train_indices),
    )
    coefficients = pd.DataFrame(rows).sort_values(
        ["feature_type", "abs_coefficient"],
        ascending=[True, False],
    )
    effects_path = output_dir / "rw_ridge_pairwise_effects.csv"
    coefficients.to_csv(effects_path, index=False)

    main, pairwise = matched_context_effects(dataset)
    main = main.sort_values("coefficient")
    main_path = output_dir / "rw_main_effects.csv"
    main.to_csv(main_path, index=False)

    pairwise = pairwise.sort_values("coefficient")
    matrix = pd.DataFrame(
        np.nan,
        index=dataset.partner_ids,
        columns=dataset.partner_ids,
    )
    for _, row in pairwise.iterrows():
        species_a = str(row["species_a"])
        species_b = str(row["species_b"])
        coefficient = float(row["coefficient"])
        matrix.loc[species_a, species_b] = coefficient
        matrix.loc[species_b, species_a] = coefficient
    matrix_path = output_dir / "rw_pairwise_effect_matrix.csv"
    matrix.to_csv(matrix_path, index_label="species_id")

    prior_rows = []
    for _, row in main.iterrows():
        coefficient = float(row["coefficient"])
        prior_rows.append({
            "effect_scope": "target_partner_main",
            "species_a": str(row["species_a"]),
            "species_b": "",
            "coefficient": coefficient,
            "interpretation": "suppresses_target" if coefficient < 0 else "increases_target",
        })
    for _, row in pairwise.iterrows():
        coefficient = float(row["coefficient"])
        prior_rows.append({
            "effect_scope": "partner_pair",
            "species_a": str(row["species_a"]),
            "species_b": str(row["species_b"]),
            "coefficient": coefficient,
            "interpretation": "pair_suppresses_target" if coefficient < 0 else "pair_increases_target",
        })
    prior_path = output_dir / "interaction_effect_prior.csv"
    pd.DataFrame(prior_rows).to_csv(prior_path, index=False)

    return {
        "effects": effects_path,
        "main_effects": main_path,
        "pairwise_matrix": matrix_path,
        "interaction_prior": prior_path,
    }


def plot_noise_fit(
    fitted: pd.DataFrame,
    model: AssayNoiseModel,
    noisy_summary: pd.DataFrame | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)
    mean_grid = np.linspace(model.observed_mean_min, model.observed_mean_max, 200)
    sd_grid = predict_assay_sd(model, mean_grid)

    ax = axes[0]
    ax.scatter(
        fitted["final_target_biomass"],
        fitted["pathogen_signal_std"],
        s=9,
        alpha=0.35,
        label="real replicate SD",
    )
    ax.plot(mean_grid, sd_grid, color="#c44e52", linewidth=2, label="fitted SD")
    if noisy_summary is not None:
        ax.scatter(
            noisy_summary["final_target_biomass"],
            noisy_summary["pathogen_signal_std"],
            s=8,
            alpha=0.20,
            label="simulated assay SD",
        )
    ax.set_xlabel("target biomass")
    ax.set_ylabel("replicate SD")
    ax.set_title("Assay noise model")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.hist(fitted["noise_model_residual"], bins=30, color="#4c72b0", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("log-SD residual")
    ax.set_ylabel("communities")
    ax.set_title("Real-data noise residuals")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_assay_noise_calibration(
    rw_summary_path: str,
    output_dir: str,
    simulation_summary_path: str | None,
    target_species: str | None,
    simulation_target_species: str | None,
    species_ids: list[str] | None,
    seed: int,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rw_summary = pd.read_csv(rw_summary_path)
    model, fitted = fit_assay_noise_model(rw_summary)

    model_path = output_path / "assay_noise_model.json"
    fit_path = output_path / "assay_noise_fit.csv"
    write_json(model_path, asdict(model))
    fitted.to_csv(fit_path, index=False)

    effect_paths = export_full_data_effects(
        rw_summary_path,
        target_species,
        species_ids,
        seed,
        output_path,
    )

    noisy_summary = None
    paths = {
        "noise_model": model_path,
        "noise_fit": fit_path,
        **effect_paths,
    }
    if simulation_summary_path is not None:
        sim_summary = pd.read_csv(simulation_summary_path)
        noisy_summary, noisy_replicates = apply_assay_noise(
            sim_summary,
            rw_summary,
            model,
            simulation_target_species or target_species or "target",
            seed,
        )
        summary_path = output_path / "simulated_noisy_summary.csv"
        replicates_path = output_path / "simulated_noisy_replicates.csv"
        noisy_summary.to_csv(summary_path, index=False)
        noisy_replicates.to_csv(replicates_path, index=False)
        paths["simulated_noisy_summary"] = summary_path
        paths["simulated_noisy_replicates"] = replicates_path

    plot_path = output_path / "assay_noise_diagnostic.png"
    plot_noise_fit(fitted, model, noisy_summary, plot_path)
    paths["diagnostic_plot"] = plot_path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit real-world assay noise and apply it to GLV simulation summaries."
    )
    parser.add_argument("--rw-summary", default="GLV_ML/outputs/real_world/log/rw_summary.csv")
    parser.add_argument("--simulation-summary")
    parser.add_argument("--output-dir", default="GLV_ML/outputs/calibration/assay_noise")
    parser.add_argument("--target-species", default="pathogen")
    parser.add_argument("--simulation-target-species")
    parser.add_argument("--species-ids")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_assay_noise_calibration(
        rw_summary_path=args.rw_summary,
        output_dir=args.output_dir,
        simulation_summary_path=args.simulation_summary,
        target_species=args.target_species,
        simulation_target_species=args.simulation_target_species,
        species_ids=parse_species_ids(args.species_ids),
        seed=args.seed,
    )
    for label, path in paths.items():
        print(f"Wrote {label} to {path}")


if __name__ == "__main__":
    main()
