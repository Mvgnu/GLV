import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import simulated_landscape_scaling as scaling
from active_learning import select_acquisitions, surrogate_recommendation_metrics
from calibrate_simulation_rates import (
    fixed_quantile_map,
    landscape_structure,
    partner_count_adjustment,
    validate_assay_mapping,
)
from lotka_volterra import generate_interaction_data, saturating_endpoint
from ml_benchmark import build_regressor
from simulation_assay_noise import matched_context_effects


class FakeEvaluator:
    def summary_for(self, groups):
        return pd.DataFrame([
            {
                "community": ";".join(sorted((*group, "sp_006"))),
                "partner_count": len(group),
                "target_species": "sp_006",
                "final_target_biomass": float(len(group)),
                "pathogen_signal_std": 0.0,
                "replicate_count": 1,
            }
            for group in groups
        ])


class SimulatedLandscapeScalingTests(unittest.TestCase):
    def test_random_acquisition_is_reproducible_without_replacement(self):
        dataset = SimpleNamespace(presence=np.zeros((6, 3), dtype=int))
        arguments = (
            "random",
            dataset,
            np.array([0, 1]),
            np.array([2, 3, 4, 5]),
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            3,
            0.05,
        )

        first = select_acquisitions(*arguments, np.random.default_rng(42))
        second = select_acquisitions(*arguments, np.random.default_rng(42))

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first)), 3)

    def test_max_diversity_uses_the_accumulated_measured_set(self):
        dataset = SimpleNamespace(presence=np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]))
        selected = select_acquisitions(
            "max_diversity",
            dataset,
            np.array([0]),
            np.array([1, 2, 3]),
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            1,
            0.05,
            np.random.default_rng(42),
        )

        np.testing.assert_array_equal(selected, np.array([2]))

    def test_surrogate_recommendations_are_validated_with_true_biomass(self):
        dataset = SimpleNamespace(target_biomass=np.array([3.0, 1.0, 2.0]))
        model = SimpleNamespace(predict=lambda _features: np.array([0.0, 2.0, 1.0]))
        metrics, recommended, _predictions = surrogate_recommendation_metrics(
            dataset,
            model,
            np.zeros((3, 1)),
            top_k=2,
        )

        np.testing.assert_array_equal(recommended, np.array([0, 2]))
        self.assertEqual(metrics["surrogate_best_validated_biomass"], 2.0)
        self.assertEqual(metrics["surrogate_global_best_gap"], 1.0)

    def test_random_forest_does_not_spawn_parallel_workers(self):
        model = build_regressor("random_forest", seed=42)
        self.assertEqual(model.n_jobs, 1)

    def test_random_sampling_is_weighted_by_available_communities(self):
        partners = [f"sp_{index}" for index in range(5)]
        selected_sizes = []
        for seed in range(1000):
            group = scaling.sample_partner_groups(
                partners,
                [1, 2],
                np.random.default_rng(seed),
                count=1,
            )[0]
            selected_sizes.append(len(group))

        size_two_rate = np.mean(np.array(selected_sizes) == 2)
        self.assertGreater(size_two_rate, 0.60)
        self.assertLess(size_two_rate, 0.72)

    def test_max_diversity_returns_only_the_requested_budget(self):
        partners = [f"sp_{index}" for index in range(8)]
        groups = scaling.max_diversity_explore_groups(
            partners,
            [2, 3, 4],
            np.random.default_rng(42),
            budget=7,
            excluded=set(),
            proposal_candidate_size=80,
        )
        self.assertEqual(len(groups), 7)
        self.assertEqual(len(set(groups)), 7)

    def test_bayesian_batches_fit_all_measurements_accumulated_so_far(self):
        train_sizes = []

        def record_training_size(dataset, train_indices, seed):
            train_sizes.append(len(train_indices))
            row_count = len(dataset.target_biomass)
            return (
                np.zeros(row_count, dtype=float),
                np.ones(row_count, dtype=float),
                np.arange(row_count, dtype=float),
            )

        partners = [f"sp_{index:03d}" for index in range(1, 6)]
        with patch.object(
            scaling,
            "bayesian_optimization_statistics",
            side_effect=record_training_size,
        ):
            groups = scaling.bayesian_iterative_groups(
                partners,
                [1, 2],
                FakeEvaluator(),
                "sp_006",
                [*partners, "sp_006"],
                seed=42,
                initial_size=2,
                batch_size=2,
                budget=6,
                excluded=set(),
                proposal_candidate_size=10,
            )

        self.assertEqual(len(groups), 6)
        self.assertEqual(train_sizes, [2, 4])

    def test_phase2_search_does_not_add_random_pool_to_walk_optimizers(self):
        partners = [f"sp_{index}" for index in range(8)]

        def score(groups):
            return -np.array([len(group) for group in groups], dtype=float)

        recommendations, _scores, evaluated_count = scaling.run_phase2_optimizer(
            "greedy_forward",
            score,
            partners,
            [3, 4, 5],
            seed=42,
            top_k=3,
            proposal_candidate_size=10_000,
        )
        self.assertTrue(all(len(group) == 5 for group in recommendations))
        self.assertLess(evaluated_count, 1000)

    def test_direct_phase2_optimizers_spend_the_exact_measurement_budget(self):
        partners = [f"sp_{index}" for index in range(8)]
        measurement_budget = 25

        for optimizer in [
            "predicted_best",
            "greedy_forward",
            "simulated_annealing",
            "genetic_algorithm",
        ]:
            measured_groups = []

            def score(groups):
                measured_groups.extend(groups)
                return np.array([len(group) + 0.01 * index for index, group in enumerate(groups)])

            recommendations, scores, evaluated_count = scaling.run_phase2_optimizer(
                optimizer,
                score,
                partners,
                [2, 3, 4],
                seed=42,
                top_k=3,
                proposal_candidate_size=40,
                measurement_budget=measurement_budget,
            )

            with self.subTest(optimizer=optimizer):
                self.assertEqual(evaluated_count, measurement_budget)
                self.assertEqual(len(measured_groups), measurement_budget)
                self.assertEqual(len(set(measured_groups)), measurement_budget)
                self.assertEqual(len(recommendations), 3)
                self.assertTrue(np.isfinite(scores).all())

    def test_partner_count_adjustment_is_independent_of_batch_shape(self):
        batch = partner_count_adjustment(np.array([3, 5]), 2.0, 4.0, 2.0)
        singles = np.array([
            partner_count_adjustment(np.array([count]), 2.0, 4.0, 2.0)[0]
            for count in [3, 5]
        ])
        np.testing.assert_allclose(batch, singles)

    def test_fixed_quantile_mapping_interpolates_between_reference_points(self):
        mapped = fixed_quantile_map(
            np.array([-0.25, 0.25, 0.75, 1.25]),
            np.array([0.0, 1.0]),
            np.array([10.0, 20.0]),
        )
        np.testing.assert_allclose(mapped, np.array([7.5, 12.5, 17.5, 22.5]))

    def test_real_assay_noise_requires_real_scale_mapping(self):
        with self.assertRaisesRegex(ValueError, "cannot use real-assay noise"):
            validate_assay_mapping(True, "latent", 1.0)
        validate_assay_mapping(True, "quantile", 1.0)
        validate_assay_mapping(True, "latent", 0.0)

    def test_hierarchical_traits_are_balanced_across_species_order(self):
        interactions = generate_interaction_data(
            species_count=28,
            interaction_range=1.0,
            off_diagonal_min=-0.5,
            off_diagonal_max=0.2,
            growth_rate=1.0,
            self_interaction=-1.0,
            target_species="sp_012",
            target_self_interaction=-1.0,
            effect_prior_csv=None,
            target_effect_scale=0.25,
            pair_effect_scale=-0.5,
            seed=42,
            interaction_generator="hierarchical",
            carrying_capacity_min=0.5,
            carrying_capacity_max=2.0,
            hierarchy_strength=0.15,
            hierarchy_noise=0.0,
        )
        species_ids = interactions["species_id"].tolist()
        matrix = interactions[species_ids].to_numpy(dtype=float)
        carrying_capacity = -1.0 / np.diag(matrix)

        self.assertLess(abs(np.corrcoef(np.arange(28), carrying_capacity)[0, 1]), 0.25)
        self.assertLess(float(np.min(carrying_capacity[:12])), 0.7)
        self.assertGreater(float(np.max(carrying_capacity[:12])), 1.8)

    def test_empirical_main_effects_set_the_target_row(self):
        priors = pd.DataFrame([
            {
                "effect_scope": "target_partner_main",
                "species_a": "sp_001",
                "species_b": "",
                "coefficient": -1.0,
            },
            {
                "effect_scope": "target_partner_main",
                "species_a": "sp_002",
                "species_b": "",
                "coefficient": 0.5,
            },
        ])
        with patch("lotka_volterra.pd.read_csv", return_value=priors):
            interactions = generate_interaction_data(
                species_count=3,
                interaction_range=1.0,
                off_diagonal_min=-0.5,
                off_diagonal_max=0.2,
                growth_rate=1.0,
                self_interaction=-1.0,
                target_species="sp_003",
                target_self_interaction=-1.0,
                effect_prior_csv="effects.csv",
                target_effect_scale=0.25,
                pair_effect_scale=-0.5,
                seed=42,
            )

        target = interactions.set_index("species_id").loc["sp_003"]
        self.assertEqual(target["sp_001"], -0.25)
        self.assertEqual(target["sp_002"], 0.125)

    def test_landscape_structure_detects_pair_epistasis(self):
        rows = []
        for mask in range(8):
            partners = [f"sp_{index + 1:03d}" for index in range(3) if mask & (1 << index)]
            value = sum([1.0, -0.5, 0.25][index] for index in range(3) if mask & (1 << index))
            if mask & 1 and mask & 2:
                value += 0.75
            rows.append({
                "community": ";".join(partners),
                "target_species": "pathogen",
                "final_target_biomass": value,
            })

        metrics, species_rows = landscape_structure(pd.DataFrame(rows), "test")
        self.assertEqual(metrics["species_count"], 3)
        self.assertGreater(metrics["pair_epistasis_sd"], 0)
        self.assertEqual(len(species_rows), 3)

    def test_effect_priors_use_matched_context_changes(self):
        presence = np.array([
            [(mask & (1 << index)) > 0 for index in range(3)]
            for mask in range(8)
        ])
        values = presence @ np.array([-1.0, 0.5, 0.25])
        values[presence[:, 0] & presence[:, 1]] += 0.75
        dataset = SimpleNamespace(
            presence=presence,
            target_biomass=values,
            partner_ids=["sp_001", "sp_002", "sp_003"],
        )

        main, pair = matched_context_effects(dataset)
        main = main.set_index("species_a")
        self.assertLess(main.loc["sp_001", "coefficient"], 0)
        self.assertGreater(main.loc["sp_002", "coefficient"], 0)
        interaction = pair[
            pair["species_a"].eq("sp_001") & pair["species_b"].eq("sp_002")
        ].iloc[0]
        self.assertAlmostEqual(float(interaction["coefficient"]), 0.75)

    def test_saturating_endpoint_reaches_single_species_fixed_point(self):
        final = saturating_endpoint(
            np.array([1.0]),
            np.array([[-1.0]]),
            initial_density=0.5,
            max_time=500.0,
            saturation_pressure=1.0,
        )
        self.assertAlmostEqual(float(final[0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
