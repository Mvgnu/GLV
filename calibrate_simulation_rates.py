#!/usr/bin/env python3
"""Calibrate generated GLV interactions to real suppressor/non-suppressor rates."""

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from lotka_volterra import generate_interaction_data
from simulation_assay_noise import fit_assay_noise_model, sample_assay_sd


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def scale_seed(
    seed: int,
    target_scale: float,
    pair_scale: float,
    partner_count_scale: float,
) -> int:
    target_part = int(round((target_scale + 1000.0) * 1000))
    pair_part = int(round((pair_scale + 1000.0) * 10000))
    count_part = int(round((partner_count_scale + 1000.0) * 100000))
    return int(
        (seed + target_part * 1009 + pair_part * 9176 + count_part * 313)
        % (2**32 - 1)
    )


def suppressor_cutoff(summary: pd.DataFrame, suppressor_fold: float) -> float:
    target_transform = str(summary.get("target_transform", pd.Series(["log1p"])).iloc[0])
    median = float(summary["final_target_biomass"].median())
    if target_transform == "raw" or "latent" in target_transform:
        return median / suppressor_fold
    return median - float(np.log(suppressor_fold))


def suppressor_rate_rows(
    summary: pd.DataFrame,
    suppressor_fold: float,
    source: str,
) -> list[dict[str, object]]:
    cutoff = suppressor_cutoff(summary, suppressor_fold)
    rows = []
    groups = [("overall", summary)]
    groups.extend(
        (int(partner_count), group)
        for partner_count, group in summary.groupby("partner_count", sort=True)
    )

    for partner_count, group in groups:
        suppressor = group["final_target_biomass"] < cutoff
        rows.append({
            "source": source,
            "partner_count": partner_count,
            "rows": int(len(group)),
            "suppressor_count": int(suppressor.sum()),
            "suppressor_rate": float(suppressor.mean()),
            "suppressor_cutoff": cutoff,
        })
    return rows


def survivor_equilibrium(
    growth_rates: np.ndarray,
    interaction_matrix: np.ndarray,
    target_index: int,
    extinction_threshold: float,
) -> np.ndarray:
    """Approximate terminal densities by iteratively removing infeasible species."""
    active = list(range(len(growth_rates)))
    final = np.zeros(len(growth_rates), dtype=float)

    while target_index in active:
        active_array = np.array(active, dtype=int)
        submatrix = interaction_matrix[np.ix_(active_array, active_array)]
        subgrowth = growth_rates[active_array]
        try:
            equilibrium = np.linalg.solve(submatrix, -subgrowth)
        except np.linalg.LinAlgError:
            equilibrium = np.linalg.lstsq(submatrix, -subgrowth, rcond=None)[0]

        if np.all(np.isfinite(equilibrium)) and np.all(equilibrium > extinction_threshold):
            final[active_array] = equilibrium
            return final

        target_position = active.index(target_index)
        if not np.isfinite(equilibrium[target_position]) or equilibrium[target_position] <= extinction_threshold:
            return final

        remove_position = int(np.nanargmin(equilibrium))
        active.pop(remove_position)

    return final


def saturated_interaction_response(
    densities: np.ndarray,
    interaction_matrix: np.ndarray,
    saturation_pressure: float,
) -> np.ndarray:
    """Bound total off-diagonal pressure while retaining signed interaction effects."""
    off_diagonal = interaction_matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    raw_pressure = off_diagonal @ densities
    return saturation_pressure * np.tanh(raw_pressure / saturation_pressure)


def saturating_derivatives(
    _time: float,
    densities: np.ndarray,
    growth_rates: np.ndarray,
    interaction_matrix: np.ndarray,
    saturation_pressure: float,
) -> np.ndarray:
    densities = np.maximum(densities, 0.0)
    self_pressure = np.diag(interaction_matrix) * densities
    interaction_pressure = saturated_interaction_response(
        densities,
        interaction_matrix,
        saturation_pressure,
    )
    return densities * (growth_rates + self_pressure + interaction_pressure)


def saturating_endpoint(
    growth_rates: np.ndarray,
    interaction_matrix: np.ndarray,
    initial_density: float,
    max_time: float,
    saturation_pressure: float,
) -> np.ndarray:
    """Integrate the bounded-interaction system to a terminal endpoint."""
    initial = np.full(len(growth_rates), initial_density, dtype=float)
    solution = solve_ivp(
        saturating_derivatives,
        t_span=(0.0, max_time),
        y0=initial,
        args=(growth_rates, interaction_matrix, saturation_pressure),
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    return np.maximum(solution.y[:, -1], 0.0)


def terminal_endpoint(
    growth_rates: np.ndarray,
    interaction_matrix: np.ndarray,
    target_index: int,
    extinction_threshold: float,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
) -> np.ndarray:
    if interaction_response == "linear":
        return survivor_equilibrium(
            growth_rates,
            interaction_matrix,
            target_index,
            extinction_threshold,
        )
    if interaction_response == "saturating":
        return saturating_endpoint(
            growth_rates,
            interaction_matrix,
            endpoint_initial_density,
            endpoint_max_time,
            saturation_pressure,
        )
    raise ValueError("interaction_response must be 'linear' or 'saturating'")


def endpoint_summary(
    interaction_data: pd.DataFrame,
    target_species: str,
    extinction_threshold: float,
    interaction_response: str = "linear",
    saturation_pressure: float = 1.0,
    endpoint_initial_density: float = 0.5,
    endpoint_max_time: float = 500.0,
) -> pd.DataFrame:
    species_ids = sorted(interaction_data["species_id"].astype(str).tolist())
    data = interaction_data.set_index("species_id").loc[species_ids]
    growth_rates = data["growth_rate"].to_numpy(dtype=float)
    interaction_matrix = data.loc[species_ids, species_ids].to_numpy(dtype=float)
    index_by_species = {species: index for index, species in enumerate(species_ids)}
    target_index = index_by_species[target_species]
    partners = [species for species in species_ids if species != target_species]
    rows = []

    for partner_count in range(len(partners) + 1):
        for partner_group in itertools.combinations(partners, partner_count):
            community = tuple(sorted((*partner_group, target_species)))
            indices = [index_by_species[species] for species in community]
            local_target_index = community.index(target_species)
            final = terminal_endpoint(
                growth_rates[indices],
                interaction_matrix[np.ix_(indices, indices)],
                local_target_index,
                extinction_threshold,
                interaction_response,
                saturation_pressure,
                endpoint_initial_density,
                endpoint_max_time,
            )
            rows.append({
                "community": ";".join(community),
                "community_size": len(community),
                "partner_count": partner_count,
                "target_species": target_species,
                "latent_target_biomass": float(final[local_target_index]),
            })

    return pd.DataFrame(rows)


def zscore_map_to_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    value_std = float(np.std(values))
    reference_std = float(np.std(reference))
    if value_std <= 1e-12:
        mapped = np.full(len(values), float(np.mean(reference)))
    else:
        mapped = (
            (values - float(np.mean(values)))
            / value_std
            * reference_std
            + float(np.mean(reference))
        )
    return np.clip(mapped, float(np.min(reference)), float(np.max(reference)))


def partner_count_adjustment(
    partner_counts: np.ndarray,
    scale: float,
    center: float,
    width: float,
) -> np.ndarray:
    """Higher-order size proxy: mid-sized communities can be uniquely suppressive."""
    if scale == 0:
        return np.zeros(len(partner_counts), dtype=float)

    basis = np.square((partner_counts.astype(float) - center) / width)
    return scale * (basis - float(np.mean(basis)))


def add_assay_observations(
    latent_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
    suppressor_fold: float,
    partner_count_effect_scale: float,
    partner_count_effect_center: float,
    partner_count_effect_width: float,
    seed: int,
    assay_noise_scale: float = 1.0,
    target_scale_mapping: str = "zscore",
) -> pd.DataFrame:
    noise_model, _fitted = fit_assay_noise_model(real_summary)
    rng = np.random.default_rng(seed)
    summary = latent_summary.copy()
    latent_values = summary["latent_target_biomass"].to_numpy(dtype=float)
    if target_scale_mapping == "zscore":
        expected_mean = zscore_map_to_reference(
            latent_values,
            real_summary["final_target_biomass"].to_numpy(dtype=float),
        )
    elif target_scale_mapping == "latent":
        expected_mean = latent_values.copy()
    else:
        raise ValueError("target_scale_mapping must be 'zscore' or 'latent'")
    expected_mean = expected_mean + partner_count_adjustment(
        summary["partner_count"].to_numpy(dtype=float),
        partner_count_effect_scale,
        partner_count_effect_center,
        partner_count_effect_width,
    )
    if target_scale_mapping == "zscore":
        expected_mean = np.clip(
            expected_mean,
            float(real_summary["final_target_biomass"].min()),
            float(real_summary["final_target_biomass"].max()),
        )
    else:
        expected_mean = np.maximum(expected_mean, 0.0)
    expected_sd = sample_assay_sd(noise_model, expected_mean, rng) * assay_noise_scale
    replicate_values = []
    replicate_sds = []

    for mean, sd in zip(expected_mean, expected_sd, strict=True):
        values = rng.normal(mean, sd, size=noise_model.replicate_count)
        if target_scale_mapping == "zscore":
            values = np.clip(
                values,
                noise_model.observed_mean_min,
                noise_model.observed_mean_max,
            )
        else:
            values = np.maximum(values, 0.0)
        replicate_values.append(float(values.mean()))
        replicate_sds.append(float(values.std(ddof=0)))

    summary["assay_mean_expected"] = expected_mean
    summary["assay_sd_expected"] = expected_sd
    summary["final_target_biomass"] = replicate_values
    summary["pathogen_signal_std"] = replicate_sds
    summary["replicate_count"] = noise_model.replicate_count
    if assay_noise_scale == 1.0 and target_scale_mapping == "zscore":
        summary["target_transform"] = f"{noise_model.target_scale}_assay_noise_zscore"
    else:
        summary["target_transform"] = (
            f"{noise_model.target_scale}_assay_noise_{target_scale_mapping}"
            f"_scale_{assay_noise_scale:g}"
        )
    summary["suppressor_cutoff"] = suppressor_cutoff(summary, suppressor_fold)
    return summary


def calibration_loss(
    real_rates: pd.DataFrame,
    simulated_rates: pd.DataFrame,
    by_count_weight: float,
) -> tuple[float, float, float]:
    real_overall = real_rates[real_rates["partner_count"].eq("overall")].iloc[0]
    sim_overall = simulated_rates[simulated_rates["partner_count"].eq("overall")].iloc[0]
    overall_error = float(sim_overall["suppressor_rate"] - real_overall["suppressor_rate"])

    real_by_count = real_rates[~real_rates["partner_count"].eq("overall")].copy()
    sim_by_count = simulated_rates[~simulated_rates["partner_count"].eq("overall")].copy()
    real_by_count["partner_count"] = real_by_count["partner_count"].astype(int)
    sim_by_count["partner_count"] = sim_by_count["partner_count"].astype(int)
    merged = real_by_count.merge(
        sim_by_count,
        on="partner_count",
        suffixes=("_real", "_simulated"),
    )
    by_count_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    merged["suppressor_rate_simulated"]
                    - merged["suppressor_rate_real"]
                )
            )
        )
    )
    loss = float(overall_error**2 + by_count_weight * by_count_rmse**2)
    return loss, overall_error, by_count_rmse


def plot_calibration(calibration: pd.DataFrame, output_path: Path) -> None:
    pivot = calibration.pivot_table(
        index="target_effect_scale",
        columns="pair_effect_scale",
        values="loss",
        aggfunc="min",
    ).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel("pair effect scale")
    ax.set_ylabel("target effect scale")
    ax.set_title("Suppressor-rate calibration loss")
    fig.colorbar(image, ax=ax, label="loss")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_calibration(
    rw_summary_path: str,
    effect_prior_csv: str,
    output_dir: str,
    species_count: int,
    target_species: str,
    target_effect_scales: list[float],
    pair_effect_scales: list[float],
    partner_count_effect_scales: list[float],
    partner_count_effect_centers: list[float],
    partner_count_effect_widths: list[float],
    interaction_range: float,
    off_diagonal_min: float | None,
    off_diagonal_max: float | None,
    growth_rate: float,
    self_interaction: float,
    target_self_interaction: float | None,
    interaction_response: str,
    saturation_pressure: float,
    endpoint_initial_density: float,
    endpoint_max_time: float,
    suppressor_fold: float,
    by_count_weight: float,
    extinction_threshold: float,
    seed: int,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    real_summary = pd.read_csv(rw_summary_path)
    real_rates = pd.DataFrame(
        suppressor_rate_rows(real_summary, suppressor_fold, "real")
    )
    real_rates.to_csv(output_path / "suppressor_rate_targets.csv", index=False)

    calibration_rows = []
    rate_rows = []
    candidate_cache = {}
    for target_scale in target_effect_scales:
        for pair_scale in pair_effect_scales:
            interaction_data = generate_interaction_data(
                species_count=species_count,
                interaction_range=interaction_range,
                off_diagonal_min=off_diagonal_min,
                off_diagonal_max=off_diagonal_max,
                growth_rate=growth_rate,
                self_interaction=self_interaction,
                target_species=target_species,
                target_self_interaction=target_self_interaction,
                effect_prior_csv=effect_prior_csv,
                target_effect_scale=target_scale,
                pair_effect_scale=pair_scale,
                seed=seed,
            )
            latent = endpoint_summary(
                interaction_data,
                target_species,
                extinction_threshold,
                interaction_response,
                saturation_pressure,
                endpoint_initial_density,
                endpoint_max_time,
            )
            for partner_count_scale in partner_count_effect_scales:
                for partner_count_center in partner_count_effect_centers:
                    for partner_count_width in partner_count_effect_widths:
                        simulated = add_assay_observations(
                            latent,
                            real_summary,
                            suppressor_fold,
                            partner_count_scale,
                            partner_count_center,
                            partner_count_width,
                            scale_seed(seed, target_scale, pair_scale, partner_count_scale),
                        )
                        simulated_rates = pd.DataFrame(
                            suppressor_rate_rows(simulated, suppressor_fold, "simulated")
                        )
                        for rate_row in simulated_rates.to_dict("records"):
                            rate_row["target_effect_scale"] = target_scale
                            rate_row["pair_effect_scale"] = pair_scale
                            rate_row["partner_count_effect_scale"] = partner_count_scale
                            rate_row["partner_count_effect_center"] = partner_count_center
                            rate_row["partner_count_effect_width"] = partner_count_width
                            rate_rows.append(rate_row)
                        loss, overall_error, by_count_rmse = calibration_loss(
                            real_rates,
                            simulated_rates,
                            by_count_weight,
                        )
                        overall_rate = simulated_rates[
                            simulated_rates["partner_count"].eq("overall")
                        ]["suppressor_rate"].iloc[0]
                        calibration_rows.append({
                            "target_effect_scale": target_scale,
                            "pair_effect_scale": pair_scale,
                            "partner_count_effect_scale": partner_count_scale,
                            "partner_count_effect_center": partner_count_center,
                            "partner_count_effect_width": partner_count_width,
                            "loss": loss,
                            "overall_rate_error": overall_error,
                            "by_partner_count_rmse": by_count_rmse,
                            "simulated_overall_suppressor_rate": float(overall_rate),
                        })
                        candidate_cache[
                            (
                                target_scale,
                                pair_scale,
                                partner_count_scale,
                                partner_count_center,
                                partner_count_width,
                            )
                        ] = (interaction_data, simulated)

    calibration = pd.DataFrame(calibration_rows).sort_values("loss")
    calibration_path = output_path / "suppressor_rate_calibration.csv"
    calibration.to_csv(calibration_path, index=False)
    pd.DataFrame(rate_rows).to_csv(
        output_path / "suppressor_rate_by_scale.csv",
        index=False,
    )
    plot_calibration(calibration, output_path / "suppressor_rate_calibration.png")

    best = calibration.iloc[0]
    best_key = (
        float(best["target_effect_scale"]),
        float(best["pair_effect_scale"]),
        float(best["partner_count_effect_scale"]),
        float(best["partner_count_effect_center"]),
        float(best["partner_count_effect_width"]),
    )
    best_interactions, best_summary = candidate_cache[best_key]
    best_interactions_path = output_path / "best_calibrated_interactions.csv"
    best_summary_path = output_path / "best_calibrated_summary.csv"
    best_interactions.to_csv(best_interactions_path, index=False)
    best_summary.to_csv(best_summary_path, index=False)
    best_rates = pd.DataFrame(
        suppressor_rate_rows(best_summary, suppressor_fold, "best_simulated")
    )
    best_rate_comparison = real_rates.merge(
        best_rates,
        on="partner_count",
        suffixes=("_real", "_simulated"),
    )
    best_rate_comparison.to_csv(
        output_path / "best_suppressor_rates.csv",
        index=False,
    )

    return calibration_path, best_interactions_path, best_summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep GLV generator effect scales against real suppressor rates."
    )
    parser.add_argument("--rw-summary", default="GLV_ML/outputs/real_world/log/rw_summary.csv")
    parser.add_argument(
        "--effect-prior-csv",
        default="GLV_ML/outputs/calibration/assay_noise/interaction_effect_prior.csv",
    )
    parser.add_argument("--output-dir", default="GLV_ML/outputs/calibration/suppressor_rates")
    parser.add_argument("--species-count", type=int, default=12)
    parser.add_argument("--target-species", default="sp_012")
    parser.add_argument("--target-effect-scales", default="-0.5,-0.25,0,0.1,0.25,0.5")
    parser.add_argument("--pair-effect-scales", default="-1,-0.5,-0.25,0,0.25,0.5,1")
    parser.add_argument("--partner-count-effect-scales", default="0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--partner-count-effect-centers", default="5.5,6.0,6.5")
    parser.add_argument("--partner-count-effect-widths", default="2.0,2.5,3.0")
    parser.add_argument("--interaction-range", type=float, default=1.0)
    parser.add_argument("--off-diagonal-min", type=float, default=-0.5)
    parser.add_argument("--off-diagonal-max", type=float, default=0.2)
    parser.add_argument("--growth-rate", type=float, default=1.0)
    parser.add_argument("--self-interaction", type=float, default=-1.0)
    parser.add_argument("--target-self-interaction", type=float, default=-1.0)
    parser.add_argument(
        "--interaction-response",
        choices=["linear", "saturating"],
        default="linear",
    )
    parser.add_argument("--saturation-pressure", type=float, default=1.0)
    parser.add_argument("--endpoint-initial-density", type=float, default=0.5)
    parser.add_argument("--endpoint-max-time", type=float, default=500.0)
    parser.add_argument("--suppressor-fold", type=float, default=2.0)
    parser.add_argument("--by-count-weight", type=float, default=1.0)
    parser.add_argument("--extinction-threshold", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_path, best_interactions_path, best_summary_path = run_calibration(
        rw_summary_path=args.rw_summary,
        effect_prior_csv=args.effect_prior_csv,
        output_dir=args.output_dir,
        species_count=args.species_count,
        target_species=args.target_species,
        target_effect_scales=parse_float_grid(args.target_effect_scales),
        pair_effect_scales=parse_float_grid(args.pair_effect_scales),
        partner_count_effect_scales=parse_float_grid(args.partner_count_effect_scales),
        partner_count_effect_centers=parse_float_grid(args.partner_count_effect_centers),
        partner_count_effect_widths=parse_float_grid(args.partner_count_effect_widths),
        interaction_range=args.interaction_range,
        off_diagonal_min=args.off_diagonal_min,
        off_diagonal_max=args.off_diagonal_max,
        growth_rate=args.growth_rate,
        self_interaction=args.self_interaction,
        target_self_interaction=args.target_self_interaction,
        interaction_response=args.interaction_response,
        saturation_pressure=args.saturation_pressure,
        endpoint_initial_density=args.endpoint_initial_density,
        endpoint_max_time=args.endpoint_max_time,
        suppressor_fold=args.suppressor_fold,
        by_count_weight=args.by_count_weight,
        extinction_threshold=args.extinction_threshold,
        seed=args.seed,
    )
    print(f"Wrote calibration sweep to {calibration_path}")
    print(f"Wrote best calibrated interactions to {best_interactions_path}")
    print(f"Wrote best calibrated summary to {best_summary_path}")


if __name__ == "__main__":
    main()
