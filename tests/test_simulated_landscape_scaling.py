import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import simulated_landscape_scaling as scaling
from calibrate_simulation_rates import fixed_quantile_map, partner_count_adjustment
from lotka_volterra import saturating_endpoint


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
