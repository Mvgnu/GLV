#!/usr/bin/env python3
"""
Generalized Lotka-Volterra (GLV) community simulations.

Input CSV format:
species_id,growth_rate,sp_a,sp_b,sp_c
sp_a,1.0,-1.0,-1.5,-0.5
sp_b,1.0,-0.5,-1.0,-1.5
sp_c,1.0,-1.5,-0.5,-1.0
"""

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import solve
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CommunityData:
    species_ids: list[str]
    growth_rates: np.ndarray
    interaction_matrix: np.ndarray


def generate_interaction_data(
    species_count,
    interaction_range,
    off_diagonal_min,
    off_diagonal_max,
    growth_rate,
    self_interaction,
    target_species,
    target_self_interaction,
    effect_prior_csv,
    target_effect_scale,
    pair_effect_scale,
    seed,
    interaction_generator="legacy",
    carrying_capacity_min=1.0,
    carrying_capacity_max=1.0,
    hierarchy_strength=0.0,
    hierarchy_noise=0.0,
    target_interaction_scale=1.0,
):
    """Generate synthetic GLV species data in the expected CSV shape."""
    rng = np.random.default_rng(seed)
    species_ids = [f"sp_{index + 1:03d}" for index in range(species_count)]
    # Bound off-diagonal effects separately so interspecies feedback can be damped.
    interaction_min = (
        off_diagonal_min
        if off_diagonal_min is not None
        else -interaction_range
    )
    interaction_max = (
        off_diagonal_max
        if off_diagonal_max is not None
        else interaction_range
    )
    if interaction_generator == "legacy":
        interaction_matrix = rng.uniform(
            interaction_min,
            interaction_max,
            size=(species_count, species_count),
        )
        # Negative self-interaction keeps each species density-limited.
        np.fill_diagonal(interaction_matrix, self_interaction)
    elif interaction_generator == "hierarchical":
        interaction_matrix = hierarchical_interaction_matrix(
            species_count,
            growth_rate,
            interaction_min,
            interaction_max,
            carrying_capacity_min,
            carrying_capacity_max,
            hierarchy_strength,
            hierarchy_noise,
            rng,
        )
    else:
        raise ValueError("--interaction-generator must be 'legacy' or 'hierarchical'")
    if (
        target_species
        and target_self_interaction is not None
        and interaction_generator == "legacy"
    ):
        # Target-specific damping bounds the biomass label without changing partners.
        target_index = species_ids.index(target_species)
        interaction_matrix[target_index, target_index] = target_self_interaction
    if effect_prior_csv:
        apply_effect_priors(
            interaction_matrix,
            species_ids,
            target_species,
            effect_prior_csv,
            target_effect_scale,
            pair_effect_scale,
        )
    if target_species and target_interaction_scale != 1.0:
        target_index = species_ids.index(target_species)
        partner_indices = [index for index in range(species_count) if index != target_index]
        interaction_matrix[target_index, partner_indices] *= target_interaction_scale

    data = pd.DataFrame(interaction_matrix, columns=species_ids)
    data.insert(0, "growth_rate", growth_rate)
    data.insert(0, "species_id", species_ids)

    return data


def hierarchical_interaction_matrix(
    species_count,
    growth_rate,
    interaction_min,
    interaction_max,
    carrying_capacity_min,
    carrying_capacity_max,
    hierarchy_strength,
    hierarchy_noise,
    rng,
):
    """Create trait-structured interactions with dominant broad-effect species."""
    if carrying_capacity_min <= 0 or carrying_capacity_max <= 0:
        raise ValueError("carrying capacities must be positive")

    dominance = np.linspace(1.0, 0.0, species_count)
    carrying_capacity = (
        carrying_capacity_min
        + (carrying_capacity_max - carrying_capacity_min) * dominance
    )
    matrix = rng.uniform(interaction_min, interaction_max, size=(species_count, species_count))

    for row_index in range(species_count):
        for column_index in range(species_count):
            if row_index == column_index:
                continue
            hierarchy_effect = -hierarchy_strength * (
                dominance[column_index] - dominance[row_index]
            )
            matrix[row_index, column_index] += hierarchy_effect
            if hierarchy_noise:
                matrix[row_index, column_index] += rng.normal(0.0, hierarchy_noise)

    matrix = np.clip(matrix, interaction_min, interaction_max)
    # For dN/dt = N(r + Aii*N), K_i = -r_i/Aii, so Aii = -r_i/K_i.
    np.fill_diagonal(matrix, -growth_rate / carrying_capacity)
    return matrix


def apply_effect_priors(
    interaction_matrix,
    species_ids,
    target_species,
    effect_prior_csv,
    target_effect_scale,
    pair_effect_scale,
):
    """Bias generated interactions with empirical target-effect priors."""
    if target_species is None:
        raise ValueError("--target-species is required when using --effect-prior-csv")

    species_index = {species: index for index, species in enumerate(species_ids)}
    target_index = species_index[target_species]
    priors = pd.read_csv(effect_prior_csv)

    for _, row in priors.iterrows():
        effect_scope = str(row["effect_scope"])
        species_a = str(row["species_a"])
        species_b = "" if pd.isna(row["species_b"]) else str(row["species_b"])
        coefficient = float(row["coefficient"])

        if effect_scope == "target_partner_main":
            if species_a not in species_index:
                continue
            # Negative ridge coefficients mean lower target biomass, so they map
            # naturally to a more negative partner effect in the target equation.
            partner_index = species_index[species_a]
            interaction_matrix[target_index, partner_index] += target_effect_scale * coefficient
        elif effect_scope == "partner_pair":
            if species_a not in species_index or species_b not in species_index:
                continue
            # Pairwise ridge effects are output epistasis, not direct GLV terms.
            # Positive pair coefficients mean the pair weakens suppression relative
            # to additive effects; partner-partner competition is the GLV proxy.
            first_index = species_index[species_a]
            second_index = species_index[species_b]
            pair_adjustment = -pair_effect_scale * coefficient
            interaction_matrix[first_index, second_index] += pair_adjustment
            interaction_matrix[second_index, first_index] += pair_adjustment


def write_generated_data(
    output_path,
    species_count,
    interaction_range,
    off_diagonal_min,
    off_diagonal_max,
    growth_rate,
    self_interaction,
    target_species,
    target_self_interaction,
    effect_prior_csv,
    target_effect_scale,
    pair_effect_scale,
    seed,
    interaction_generator,
    carrying_capacity_min,
    carrying_capacity_max,
    hierarchy_strength,
    hierarchy_noise,
    target_interaction_scale,
):
    """Write synthetic GLV input data to CSV."""
    # Persist the exact matrix shape consumed by the simulator.
    data = generate_interaction_data(
        species_count,
        interaction_range,
        off_diagonal_min,
        off_diagonal_max,
        growth_rate,
        self_interaction,
        target_species,
        target_self_interaction,
        effect_prior_csv,
        target_effect_scale,
        pair_effect_scale,
        seed,
        interaction_generator,
        carrying_capacity_min,
        carrying_capacity_max,
        hierarchy_strength,
        hierarchy_noise,
        target_interaction_scale,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    return output_path, data


def glv_derivatives(_time, densities, growth_rates, interaction_matrix):
    """Return density changes for the generalized Lotka-Volterra system."""
    densities = np.array(densities, dtype=float)

    # GLV dynamics: growth is scaled by current density.
    return densities * (growth_rates + interaction_matrix @ densities)


def load_community_data(csv_path):
    """Load species ids, growth rates, and interactions from a square CSV."""
    data = pd.read_csv(csv_path)
    # CSV rows and interaction columns must name the same species universe.
    required_columns = {"species_id", "growth_rate"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    species_ids = sorted(data["species_id"].astype(str).tolist())

    if len(species_ids) != len(set(species_ids)):
        raise ValueError("species_id values must be unique")

    missing_species_columns = set(species_ids) - set(data.columns)
    if missing_species_columns:
        missing = ", ".join(sorted(missing_species_columns))
        raise ValueError(f"Missing interaction column(s): {missing}")

    # Sort once so all downstream combinations and matrices are deterministic.
    data = data.set_index("species_id").loc[species_ids]
    growth_rates = data["growth_rate"].to_numpy(dtype=float)
    interaction_matrix = data.loc[species_ids, species_ids].to_numpy(dtype=float)

    return CommunityData(species_ids, growth_rates, interaction_matrix)


def build_evaluation_times(max_time, time_step):
    """Create stable evaluation times including max_time."""
    # Include max_time exactly even when step spacing does not land on it.
    times = np.arange(0, max_time, time_step)
    return np.unique(np.append(times, max_time))


def build_simulation_output(species_ids, times, densities, derivatives):
    """Build simulation output with derivative norms."""
    # effect of other species on a species
    derivative_columns = [f"d_{species}" for species in species_ids]
    column_names = ["time"] + list(species_ids) + derivative_columns + ["derivative_norm"]
    # RMS derivative keeps the fixed-point threshold comparable across community sizes.
    derivative_norms = np.sqrt(np.mean(np.square(derivatives), axis=1))
    # combine all the values into a single array
    simulation_data = np.column_stack([times, densities, derivatives, derivative_norms])

    return pd.DataFrame(simulation_data, columns=column_names)


def make_extinction_event(extinction_threshold, species_index=None):
    # extinction event when density is below extinction threshold
    def extinction_event(_time, densities, _growth_rates, _interaction_matrix):
        # if no target species
        if species_index is None:
            return np.min(densities) - extinction_threshold
        return densities[species_index] - extinction_threshold

    extinction_event.terminal = True
    extinction_event.direction = -1
    return extinction_event


def make_blow_up_event(blow_up_threshold):
    # Stop integration before runaway densities dominate the output table.
    def blow_up_event(_time, densities, _growth_rates, _interaction_matrix):
        return blow_up_threshold - np.max(np.abs(densities))

    blow_up_event.terminal = True
    blow_up_event.direction = -1
    return blow_up_event


def integrate_glv(
    species_ids,
    growth_rates,
    interaction_matrix,
    initial_densities,
    max_time=100,
    time_step=0.5,
    fixed_point_threshold=1e-6,
    extinction_threshold=1e-8,
    blow_up_threshold=1e6,
    orbit_threshold=1e-3,
    cycle_amplitude_tolerance=0.05,
    cycle_window=10,
    min_cycle_time=1,
    start_check_step=50,
    check_interval_steps=10,
    target_species=None,
):
    """Integrate the GLV equations with RK45 and return wide-format output."""
    growth_rates = np.asarray(growth_rates, dtype=float)
    interaction_matrix = np.asarray(interaction_matrix, dtype=float)
    initial_densities = np.asarray(initial_densities, dtype=float)
    species_ids = list(species_ids)
    target_index = species_ids.index(target_species) if target_species else None
    evaluation_times = build_evaluation_times(max_time, time_step)
    # Integrate in chunks so stabilization checks can stop before max_time.
    check_indices = list(range(start_check_step, len(evaluation_times), check_interval_steps))
    if check_indices[-1:] != [len(evaluation_times) - 1]:
        check_indices.append(len(evaluation_times) - 1)

    # Accumulate accepted trajectory rows across chunks.
    times = [evaluation_times[0]]
    densities = [initial_densities]
    derivatives = [
        glv_derivatives(
            evaluation_times[0],
            initial_densities,
            growth_rates,
            interaction_matrix,
        )
    ]
    current_index = 0
    current_densities = initial_densities

    for check_index in check_indices:
        segment_times = evaluation_times[current_index:check_index + 1]
        # RK45 handles adaptive internal steps; t_eval controls recorded rows.
        solution = solve_ivp(
            glv_derivatives, # equation
            t_span=(segment_times[0], segment_times[-1]), # time span
            y0=current_densities, # initial densities
            t_eval=segment_times, # times to evaluate
            args=(growth_rates, interaction_matrix), # pass growth and interaction matrix to glv_derivatives
            method="RK45", # Runge-Kutta method, adaptive step size
            events=[
                make_extinction_event(extinction_threshold, target_index),
                make_blow_up_event(blow_up_threshold),
            ],
        )

        # Transpose time[species] -> species[time]
        segment_densities = solution.y.T
        segment_times = list(solution.t[1:])
        segment_densities = list(segment_densities[1:])
        event_indices = [
            index
            for index, event_times in enumerate(solution.t_events)
            if len(event_times)
        ]
        terminal_event = None

        if event_indices:
            # Append the exact event state so classification sees the terminal row.
            event_index = event_indices[0]
            terminal_event = "extinction" if event_index == 0 else "blow_up"
            event_time = solution.t_events[event_index][0]
            event_density = solution.y_events[event_index][0]
            terminal_species = None
            if terminal_event == "extinction":
                if target_index is None:
                    terminal_species = list(species_ids)[np.argmin(event_density)]
                else:
                    terminal_species = target_species
            segment_times.append(event_time)
            segment_densities.append(event_density)

        segment_derivatives = [
            glv_derivatives(time, density, growth_rates, interaction_matrix)
            for time, density in zip(segment_times, segment_densities)
        ]
        times.extend(segment_times)
        densities.extend(segment_densities)
        derivatives.extend(segment_derivatives)

        simulation_output = build_simulation_output(
            species_ids,
            np.array(times),
            np.array(densities),
            np.array(derivatives),
        )

        if event_indices:
            simulation_output.attrs["terminal_event"] = terminal_event
            simulation_output.attrs["terminal_species"] = terminal_species
            return simulation_output

        # Fixed points and repeating orbits are checked only at configured intervals.
        stable = simulation_output["derivative_norm"].iloc[-1] <= fixed_point_threshold
        orbit = detect_repeating_orbit(
            simulation_output[list(species_ids)].to_numpy(dtype=float),
            simulation_output["time"].to_numpy(dtype=float),
            time_step,
            len(species_ids),
            orbit_threshold,
            cycle_amplitude_tolerance,
            cycle_window,
            min_cycle_time,
        )

        if stable or orbit:
            return simulation_output

        current_index = check_index
        current_densities = segment_densities[-1]

    return simulation_output


def long_format(simulation_output):
    """Return simulation output as time, species, density rows."""
    species_columns = [
        col
        for col in simulation_output.columns
        if col not in {"time", "derivative_norm"} and not col.startswith("d_")
    ]
    return simulation_output.melt(
        id_vars=["time"],
        value_vars=species_columns,
        var_name="species",
        value_name="density",
    )


def plot_simulation_output(simulation_output, title, output_path=None):
    """Plot species densities and optionally save the figure."""
    long_output = long_format(simulation_output)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Get species columns, omitting the time column
    species_columns = [
        col
        for col in simulation_output.columns
        if col not in {"time", "derivative_norm"} and not col.startswith("d_")
    ]

    # Plot each species density over time
    for species in species_columns:
        species_output = long_output[long_output["species"] == species]
        ax.plot(
            species_output["time"],
            species_output["density"],
            label=species,
            linewidth=2.0,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Density / Population", fontsize=11, fontweight="semibold")
    ax.legend(title="Species", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    # Save figure if output path is provided, otherwise show plot
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

    return long_output


def plot_eigenvalue_output(eigenvalue_output, output_path):
    """Plot community eigenvalues in the complex plane."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    stable_points = eigenvalue_output[eigenvalue_output["eigenvalue_stable"]]
    unstable_points = eigenvalue_output[~eigenvalue_output["eigenvalue_stable"]]

    ax.scatter(
        stable_points["real"],
        stable_points["imaginary"],
        label="negative real part",
        color="tab:green",
        alpha=0.75,
    )
    ax.scatter(
        unstable_points["real"],
        unstable_points["imaginary"],
        label="nonnegative real part",
        color="tab:red",
        alpha=0.75,
    )

    # Real parts left of zero indicate local stability at the coexistence equilibrium.
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)

    ax.set_title("Community Jacobian Eigenvalues", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Real part", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Imaginary part", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def equilibrium_densities(growth_rates, interaction_matrix):
    """Return coexistence equilibrium densities."""
    # solve for equilibrium densities
    return solve(interaction_matrix, -growth_rates)


def jacobian_eigenvalues(equilibrium, interaction_matrix):
    """Return Jacobian eigenvalues at a feasible coexistence equilibrium."""
    # calculate jacobian
    jacobian = np.diag(equilibrium) @ interaction_matrix
    return np.linalg.eigvals(jacobian)


def equilibrium_stats(species_ids, growth_rates, interaction_matrix, extinction_threshold):
    """Estimate feasibility and local stability of the coexistence equilibrium."""
    stats = {
        "feasible_equilibrium": False,
        "equilibrium_stable": False,
        "max_equilibrium_eigenvalue_real": np.nan,
    }

    equilibrium = equilibrium_densities(growth_rates, interaction_matrix)

    # add equilibrium species densities to stats
    for species, density in zip(species_ids, equilibrium):
        stats[f"equilibrium_{species}"] = density

    # check if equilibrium is feasible
    if not np.all(np.isfinite(equilibrium)):
        return stats

    # check if equilibrium is stable
    stats["feasible_equilibrium"] = bool(np.all(equilibrium > extinction_threshold))
    if not stats["feasible_equilibrium"]:
        return stats

    eigenvalues = jacobian_eigenvalues(equilibrium, interaction_matrix)
    # calculate max real eigenvalue
    max_real_eigenvalue = np.max(eigenvalues.real)

    stats["max_equilibrium_eigenvalue_real"] = max_real_eigenvalue
    stats["equilibrium_stable"] = bool(max_real_eigenvalue < 0)

    return stats


def detect_repeating_orbit(
    densities,
    times,
    time_step,
    species_count,
    orbit_threshold,
    cycle_amplitude_tolerance,
    cycle_window,
    min_cycle_time,
):
    """Detect a numerically repeating orbit in the trailing trajectory."""
    if species_count < 3 or len(times) < 4:
        return None

    # window size in units of time steps
    window_size = int(round(cycle_window / time_step))
    # minimum lag in units of time steps (avoids comparing overlapping windows)
    min_lag = max(window_size, int(round(min_cycle_time / time_step)))
    # maximum lag in units of time steps (should be less than half the trajectory to allow comparison of two different points)
    max_lag = len(times) - window_size

    # if the time window is too small compared to the minimum lag, return None
    if max_lag < min_lag:
        return None

    # get the tail of the trajectory
    tail = densities[-window_size:]
    tail_amplitude = np.ptp(tail, axis=0)
    # divide magnitude of tail by sqrt of size -> RMS magnitude, avoid division by zero
    scale = max(np.linalg.norm(tail) / np.sqrt(tail.size), 1e-12)
    # get best matching earlier segment
    best_error = np.inf
    best_lag = None
    best_amplitude_error = np.nan

    # for lag in range error between tail and earlier segment is compared
    for lag in range(min_lag, max_lag + 1):
        # get preceeding window of same size as tail
        previous = densities[-window_size - lag:-lag]
        previous_amplitude = np.ptp(previous, axis=0)
        # total difference between tail and previous, normalized to per value error, relative to magnitude
        error = np.linalg.norm(tail - previous) / np.sqrt(tail.size) / scale
        amplitude_error = np.linalg.norm(tail_amplitude - previous_amplitude) / max(
            np.linalg.norm(previous_amplitude),
            1e-12,
        )
        # keep best error and corresponding lag
        if error < best_error and amplitude_error <= cycle_amplitude_tolerance:
            best_error = error
            best_lag = lag
            best_amplitude_error = amplitude_error
    # if best error is close enough to orbit threshold, classify as periodic orbit
    if best_error <= orbit_threshold:
        return {
            # convert back to time units
            "cycle_period": best_lag * time_step,
            "cycle_error": best_error,
            "cycle_amplitude_error": best_amplitude_error,
        }

    # if no repeating orbit found return none
    return None


def classify_stop_condition(
    simulation_output,
    species_ids,
    growth_rates,
    interaction_matrix,
    time_step,
    fixed_point_threshold,
    extinction_threshold,
    blow_up_threshold,
    orbit_threshold,
    cycle_amplitude_tolerance,
    cycle_window,
    min_cycle_time,
    target_species=None,
):
    """Classify the trajectory's terminal behavior."""
    species_ids = list(species_ids)
    target_index = species_ids.index(target_species) if target_species else None

    times = simulation_output["time"].to_numpy(dtype=float)
    densities = simulation_output[species_ids].to_numpy(dtype=float)
    derivative_columns = [f"d_{species}" for species in species_ids]
    derivatives = simulation_output[derivative_columns].to_numpy(dtype=float)
    derivative_norms = simulation_output["derivative_norm"].to_numpy(dtype=float)

    # Collect candidate terminal states and choose whichever happened first.
    candidates = []
    terminal_event = simulation_output.attrs.get("terminal_event")
    terminal_species = simulation_output.attrs.get("terminal_species")
    if terminal_event:
        candidates.append((terminal_event, len(times) - 1))

    # make list of blow up indices (densities are greater than threshold)
    blow_up_indices = np.where(np.max(np.abs(densities), axis=1) >= blow_up_threshold)[0]
    # make list of extinction indices (target density in target mode, otherwise any density)
    if target_index is None:
        extinction_indices = np.where(np.any(densities <= extinction_threshold, axis=1))[0]
    else:
        extinction_indices = np.where(densities[:, target_index] <= extinction_threshold)[0]
    # make list of fixed point indices (RMS derivative is smaller than fixed point threshold)
    fixed_point_indices = np.where(derivative_norms <= fixed_point_threshold)[0]

    # add candidates to list
    if len(blow_up_indices):
        candidates.append(("blow_up", blow_up_indices[0]))
    if len(extinction_indices):
        candidates.append(("extinction", extinction_indices[0]))
    if len(fixed_point_indices):
        candidates.append(("fixed_point", fixed_point_indices[0]))

    # detect repeating orbit
    orbit = detect_repeating_orbit(
        densities,
        times,
        time_step,
        len(species_ids),
        orbit_threshold,
        cycle_amplitude_tolerance,
        cycle_window,
        min_cycle_time,
    )

    # if threshold met, set status and stop index
    if candidates:
        status, stop_index = min(candidates, key=lambda item: item[1])
    # else if limit cycle detected
    elif orbit:
        status, stop_index = "limit_cycle", len(times) - 1
    # else max time
    else:
        status, stop_index = "max_time", len(times) - 1

    # get species densities at stop
    stop_densities = densities[stop_index]
    # get list of extinct species
    extinct_species = [
        species
        for species, density in zip(species_ids, stop_densities)
        if density <= extinction_threshold
    ]
    if terminal_event == "extinction" and terminal_species not in extinct_species:
        extinct_species.append(terminal_species)

    # Summary stats are the stable contract consumed by ML and diagnostics.
    stats = {
        "status": status,
        "stable_at_stop": status in {"fixed_point", "limit_cycle"},
        "stop_time": times[stop_index],
        "derivative_norm": derivative_norms[stop_index],
        "max_abs_derivative": np.max(np.abs(derivatives[stop_index])),
        "extinct_species": ";".join(extinct_species),
        "target_species": target_species or "",
        "final_target_biomass": (
            stop_densities[target_index]
            if target_index is not None
            else np.nan
        ),
        "target_extinct": (
            bool(stop_densities[target_index] <= extinction_threshold)
            if target_index is not None
            else False
        ),
        "cycle_period": np.nan,
        "cycle_error": np.nan,
        "cycle_amplitude_error": np.nan,
    }

    # add cycle period and error if limit cycle detected
    if orbit and status == "limit_cycle":
        stats.update(orbit)

    # add final species densities
    for species, density in zip(species_ids, stop_densities):
        stats[f"final_{species}"] = density

    # add equilibrium statistics
    stats.update(
        equilibrium_stats(
            species_ids,
            growth_rates,
            interaction_matrix,
            extinction_threshold,
        )
    )

    return stats


def safe_filename(parts):
    """Build a deterministic filename from species ids."""
    joined = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", joined)


def enumerate_species_groups(species_ids, community_size, target_species=None):
    """Return community tuples, always including target_species when provided."""
    # Without a target, community_size is the total number of species.
    if target_species is None:
        return list(itertools.combinations(species_ids, community_size))

    if target_species not in species_ids:
        raise ValueError("target_species must be present in the CSV species_id column")

    # With a target, community_size is partner count around the fixed target.
    partner_species = [species for species in species_ids if species != target_species]
    if not 0 <= community_size <= len(partner_species):
        raise ValueError(
            "community_size must be between 0 and the number of non-target partner species"
        )

    return [
        tuple(sorted((*partner_group, target_species)))
        for partner_group in itertools.combinations(partner_species, community_size)
    ]


def run_communities(
    csv_path,
    community_size,
    output_dir,
    initial_density,
    max_time,
    time_step,
    fixed_point_threshold,
    extinction_threshold,
    blow_up_threshold,
    orbit_threshold,
    cycle_amplitude_tolerance,
    cycle_window,
    min_cycle_time,
    start_check_step,
    check_interval_steps,
    target_species=None,
    save_trajectories=False,
    plot_communities=True,
):
    """Simulate every species combination and write deterministic outputs."""
    community_data = load_community_data(csv_path)

    if target_species is None and not 1 <= community_size <= len(community_data.species_ids):
        raise ValueError(
            "community_size must be between 1 and the number of species in the CSV"
        )

    output_name = (
        f"partner_count_{community_size}"
        if target_species
        else f"community_size_{community_size}"
    )
    output_dir = Path(output_dir) / output_name
    plots_dir = output_dir / "plots"
    trajectories_dir = output_dir / "trajectories"

    # Each community size gets isolated output directories.
    plots_dir.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    eigenvalue_rows = []
    species_groups = enumerate_species_groups(
        community_data.species_ids,
        community_size,
        target_species,
    )
    index_by_species = {
        species: index for index, species in enumerate(community_data.species_ids)
    }

    for species_group in species_groups:
        # Slice the full GLV system down to this community's species.
        indices = [index_by_species[species] for species in species_group]
        growth_rates = community_data.growth_rates[indices]
        interaction_matrix = community_data.interaction_matrix[np.ix_(indices, indices)]
        initial_densities = np.full(len(species_group), initial_density)
        community_name = safe_filename(species_group)
        plot_path = plots_dir / f"{community_name}.png"

        # Integrate first, then classify from the recorded trajectory.
        simulation_output = integrate_glv(
            species_group,
            growth_rates,
            interaction_matrix,
            initial_densities,
            max_time=max_time,
            time_step=time_step,
            fixed_point_threshold=fixed_point_threshold,
            extinction_threshold=extinction_threshold,
            blow_up_threshold=blow_up_threshold,
            orbit_threshold=orbit_threshold,
            cycle_amplitude_tolerance=cycle_amplitude_tolerance,
            cycle_window=cycle_window,
            min_cycle_time=min_cycle_time,
            start_check_step=start_check_step,
            check_interval_steps=check_interval_steps,
            target_species=target_species,
        )
        if plot_communities:
            plot_simulation_output(
                simulation_output,
                title=f"GLV Community: {', '.join(species_group)}",
                output_path=plot_path,
            )
        else:
            plot_path = ""

        if save_trajectories:
            trajectory_path = trajectories_dir / f"{community_name}.csv"
            long_format(simulation_output).to_csv(trajectory_path, index=False)
        else:
            trajectory_path = ""

        # Classification adds stop status, final densities, and equilibrium stats.
        stats = classify_stop_condition(
            simulation_output,
            species_group,
            growth_rates,
            interaction_matrix,
            time_step,
            fixed_point_threshold,
            extinction_threshold,
            blow_up_threshold,
            orbit_threshold,
            cycle_amplitude_tolerance,
            cycle_window,
            min_cycle_time,
            target_species=target_species,
        )

        equilibrium = equilibrium_densities(growth_rates, interaction_matrix)
        feasible_equilibrium = bool(np.all(equilibrium > extinction_threshold))

        if feasible_equilibrium:
            # Eigenvalues are only meaningful for feasible coexistence equilibria.
            eigenvalues = jacobian_eigenvalues(equilibrium, interaction_matrix)
            community_stable = bool(np.max(eigenvalues.real) < 0)

            for eigenvalue_index, eigenvalue in enumerate(eigenvalues, start=1):
                eigenvalue_rows.append({
                    "community": ";".join(species_group),
                    "community_size": len(species_group),
                    "partner_count": (
                        community_size if target_species else max(len(species_group) - 1, 0)
                    ),
                    "target_species": target_species or "",
                    "eigenvalue_index": eigenvalue_index,
                    "real": eigenvalue.real,
                    "imaginary": eigenvalue.imag,
                    "eigenvalue_stable": bool(eigenvalue.real < 0),
                    "community_stable": community_stable,
                })

        summary_rows.append({
            "community": ";".join(species_group),
            "community_size": len(species_group),
            "partner_count": (
                community_size if target_species else max(len(species_group) - 1, 0)
            ),
            "target_species": target_species or "",
            "plot_path": str(plot_path),
            "trajectory_path": str(trajectory_path),
            **stats,
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    # Eigenvalue output supports stability plots across feasible communities.
    eigenvalue_columns = [
        "community",
        "community_size",
        "partner_count",
        "target_species",
        "eigenvalue_index",
        "real",
        "imaginary",
        "eigenvalue_stable",
        "community_stable",
    ]
    eigenvalue_output = pd.DataFrame(eigenvalue_rows, columns=eigenvalue_columns)
    eigenvalue_output["eigenvalue_stable"] = eigenvalue_output["eigenvalue_stable"].astype(bool)
    eigenvalue_output["community_stable"] = eigenvalue_output["community_stable"].astype(bool)
    eigenvalue_path = output_dir / "eigenvalues.csv"
    eigenvalue_plot_path = output_dir / "eigenvalues.png"
    eigenvalue_output.to_csv(eigenvalue_path, index=False)
    plot_eigenvalue_output(eigenvalue_output, eigenvalue_plot_path)

    return summary_path, summary, eigenvalue_path, eigenvalue_plot_path


def run_community_size_range(
    csv_path,
    min_community_size,
    max_community_size,
    output_dir,
    initial_density,
    max_time,
    time_step,
    fixed_point_threshold,
    extinction_threshold,
    blow_up_threshold,
    orbit_threshold,
    cycle_amplitude_tolerance,
    cycle_window,
    min_cycle_time,
    start_check_step,
    check_interval_steps,
    target_species=None,
    save_trajectories=False,
    plot_communities=True,
):
    """Simulate a range of community sizes and write combined summaries."""
    summary_frames = []
    eigenvalue_frames = []
    output_dir = Path(output_dir)

    # Run each size separately, then publish combined tables for ML.
    for community_size in range(min_community_size, max_community_size + 1):
        summary_path, summary, eigenvalue_path, _eigenvalue_plot_path = run_communities(
            csv_path=csv_path,
            community_size=community_size,
            output_dir=output_dir,
            initial_density=initial_density,
            max_time=max_time,
            time_step=time_step,
            fixed_point_threshold=fixed_point_threshold,
            extinction_threshold=extinction_threshold,
            blow_up_threshold=blow_up_threshold,
            orbit_threshold=orbit_threshold,
            cycle_amplitude_tolerance=cycle_amplitude_tolerance,
            cycle_window=cycle_window,
            min_cycle_time=min_cycle_time,
            start_check_step=start_check_step,
            check_interval_steps=check_interval_steps,
            target_species=target_species,
            save_trajectories=save_trajectories,
            plot_communities=plot_communities,
        )
        summary_frames.append(summary)
        eigenvalue_frames.append(pd.read_csv(eigenvalue_path))
        print(f"Wrote {len(summary)} community result(s) to {summary_path}")

    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_eigenvalues = pd.concat(eigenvalue_frames, ignore_index=True)
    all_summary_path = output_dir / "all_summary_stats.csv"
    all_eigenvalue_path = output_dir / "all_eigenvalues.csv"
    all_summary.to_csv(all_summary_path, index=False)
    all_eigenvalues.to_csv(all_eigenvalue_path, index=False)

    return all_summary_path, all_summary, all_eigenvalue_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GLV simulations for species combinations from a CSV."
    )
    parser.add_argument("csv_path", nargs="?", help="CSV with species_id, growth_rate, and species columns.")
    parser.add_argument("--community-size", type=int)
    parser.add_argument("--min-community-size", type=int)
    parser.add_argument("--max-community-size", type=int)
    parser.add_argument(
        "--target-species",
        help="Species included in every community; size arguments then mean partner count.",
    )
    parser.add_argument("--output-dir", default="GLV_ML/outputs/simulation/exhaustive")
    parser.add_argument("--initial-density", type=float, default=0.5)
    parser.add_argument("--max-time", type=float, default=2000)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--fixed-point-threshold", type=float, default=1e-6)
    parser.add_argument("--extinction-threshold", type=float, default=1e-8)
    parser.add_argument("--blow-up-threshold", type=float, default=1e6)
    parser.add_argument("--orbit-threshold", type=float, default=1e-3)
    parser.add_argument("--cycle-amplitude-tolerance", type=float, default=0.05)
    parser.add_argument("--cycle-window", type=float, default=10)
    parser.add_argument("--min-cycle-time", type=float, default=1)
    parser.add_argument("--start-check-step", type=int, default=50)
    parser.add_argument("--check-interval-steps", type=int, default=50)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--skip-community-plots", action="store_true")
    parser.add_argument("--generate-csv", help="Write synthetic GLV input data to this CSV path and exit.")
    parser.add_argument("--species-count", type=int, default=10)
    parser.add_argument("--interaction-range", type=float, default=1.0)
    # Explicit off-diagonal bounds replace the symmetric interaction range.
    parser.add_argument("--off-diagonal-min", type=float)
    parser.add_argument("--off-diagonal-max", type=float)
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
    # Lets the target be stabilized independently when it is the prediction label.
    parser.add_argument("--target-self-interaction", type=float)
    parser.add_argument(
        "--effect-prior-csv",
        help="Optional interaction_effect_prior.csv used to bias generated target effects.",
    )
    parser.add_argument(
        "--target-effect-scale",
        type=float,
        default=0.25,
        help="Scale mapping empirical main effects onto target-row GLV interactions.",
    )
    parser.add_argument(
        "--pair-effect-scale",
        type=float,
        default=0.25,
        help="Scale mapping empirical pair output effects onto partner-partner GLV interactions.",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate_csv:
        output_path, data = write_generated_data(
            output_path=args.generate_csv,
            species_count=args.species_count,
            interaction_range=args.interaction_range,
            off_diagonal_min=args.off_diagonal_min,
            off_diagonal_max=args.off_diagonal_max,
            growth_rate=args.growth_rate,
            self_interaction=args.self_interaction,
            target_species=args.target_species,
            target_self_interaction=args.target_self_interaction,
            effect_prior_csv=args.effect_prior_csv,
            target_effect_scale=args.target_effect_scale,
            pair_effect_scale=args.pair_effect_scale,
            seed=args.seed,
            interaction_generator=args.interaction_generator,
            carrying_capacity_min=args.carrying_capacity_min,
            carrying_capacity_max=args.carrying_capacity_max,
            hierarchy_strength=args.hierarchy_strength,
            hierarchy_noise=args.hierarchy_noise,
            target_interaction_scale=args.target_interaction_scale,
        )
        print(f"Wrote {len(data)} generated species to {output_path}")
        return

    if args.csv_path is None:
        raise ValueError("csv_path is required unless --generate-csv is used")

    if args.min_community_size is not None or args.max_community_size is not None:
        if args.min_community_size is None or args.max_community_size is None:
            raise ValueError("--min-community-size and --max-community-size must be provided together")

        summary_path, summary, eigenvalue_path = run_community_size_range(
            csv_path=args.csv_path,
            min_community_size=args.min_community_size,
            max_community_size=args.max_community_size,
            output_dir=args.output_dir,
            initial_density=args.initial_density,
            max_time=args.max_time,
            time_step=args.time_step,
            fixed_point_threshold=args.fixed_point_threshold,
            extinction_threshold=args.extinction_threshold,
            blow_up_threshold=args.blow_up_threshold,
            orbit_threshold=args.orbit_threshold,
            cycle_amplitude_tolerance=args.cycle_amplitude_tolerance,
            cycle_window=args.cycle_window,
            min_cycle_time=args.min_cycle_time,
            start_check_step=args.start_check_step,
            check_interval_steps=args.check_interval_steps,
            target_species=args.target_species,
            save_trajectories=args.save_trajectories,
            plot_communities=not args.skip_community_plots,
        )
        print(f"Wrote {len(summary)} combined community result(s) to {summary_path}")
        print(f"Wrote combined eigenvalues to {eigenvalue_path}")
        return

    if args.community_size is None:
        raise ValueError("--community-size is required unless using a community size range")

    summary_path, summary, eigenvalue_path, eigenvalue_plot_path = run_communities(
        csv_path=args.csv_path,
        community_size=args.community_size,
        output_dir=args.output_dir,
        initial_density=args.initial_density,
        max_time=args.max_time,
        time_step=args.time_step,
        fixed_point_threshold=args.fixed_point_threshold,
        extinction_threshold=args.extinction_threshold,
        blow_up_threshold=args.blow_up_threshold,
        orbit_threshold=args.orbit_threshold,
        cycle_amplitude_tolerance=args.cycle_amplitude_tolerance,
        cycle_window=args.cycle_window,
        min_cycle_time=args.min_cycle_time,
        start_check_step=args.start_check_step,
        check_interval_steps=args.check_interval_steps,
        target_species=args.target_species,
        save_trajectories=args.save_trajectories,
        plot_communities=not args.skip_community_plots,
    )

    print(f"Wrote {len(summary)} community result(s) to {summary_path}")
    print(f"Wrote eigenvalues to {eigenvalue_path}")
    print(f"Wrote eigenvalue plot to {eigenvalue_plot_path}")


if __name__ == "__main__":
    main()
