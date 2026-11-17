import contextlib
import csv
import io
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import radar_triangulation_v5 as radar
from test_radar_triangulation_v4 import rotating_beam_points


def randomize_phases(points, seed=42):
    result = points.copy()
    result[:, 0] += np.random.default_rng(seed).uniform(-2.5, 2.5, len(points))
    return result


def assess(points, site=(51.0, 6.0, 5.0)):
    training, validation = radar.split_search_validation(points)
    return radar.evaluate_candidate(
        points, site, "II1", 1.0, 1, training_points=training, validation_points=validation
    )


class ValidationTests(unittest.TestCase):
    def test_split_keeps_aircraft_time_blocks_disjoint_and_is_order_independent(self):
        points = rotating_beam_points(cycles=100)
        training, validation = radar.split_search_validation(points)

        def groups(rows):
            return {
                (int(row[4]), math.floor((row[0] - radar.T0_EPOCH) / 300.0)) for row in rows
            }

        self.assertEqual(len(training) + len(validation), len(points))
        self.assertFalse(groups(training) & groups(validation))
        self.assertGreater(len(training), len(points) // 3)
        self.assertGreater(len(validation), len(points) // 3)
        shuffled = points[np.random.default_rng(42).permutation(len(points))]
        train_shuffled, val_shuffled = radar.split_search_validation(shuffled)
        self.assertEqual(groups(training), groups(train_shuffled))
        self.assertEqual(groups(validation), groups(val_shuffled))

    def test_random_phases_do_not_become_temporally_stable_after_inlier_selection(self):
        points = randomize_phases(rotating_beam_points(cycles=100))
        rbar, _, mask, _, _, _, stability = radar.evaluate_site_diagnostics(
            points, 51.0, 6.0, 5.0
        )
        bearing, _ = radar.site_geometry(points, 51.0, 6.0)
        phase = bearing - np.remainder(points[:, 0] - radar.T0_EPOCH, 5.0) * (radar.PI2 / 5.0)
        # This is the v4 failure: even noise looks coherent after a phase cut.
        self.assertGreater(abs(np.mean(np.exp(1j * phase[mask]))), 0.9)
        self.assertLess(rbar, 0.05)
        self.assertEqual(stability, 0.0)
        self.assertIsNone(assess(points)[0])

    def test_training_perfection_cannot_turn_holdout_noise_into_a_station(self):
        training = rotating_beam_points(cycles=100)
        for seed in (1, 42, 123):
            with self.subTest(seed=seed):
                validation = randomize_phases(training, seed)
                row, mask, train_rbar = radar.evaluate_candidate(
                    np.vstack((training, validation)), (51.0, 6.0, 5.0), "II1", 1.0, 1,
                    training_points=training, validation_points=validation,
                )
                self.assertAlmostEqual(train_rbar, 1.0, places=10)
                self.assertTrue(mask.all())
                self.assertIsNone(row)

    def test_single_coherent_segment_cannot_pass_as_a_stable_station(self):
        points = rotating_beam_points(cycles=100)
        points[len(points) // 4:] = randomize_phases(points[len(points) // 4:])
        diagnostics = radar.evaluate_site_diagnostics(points, 51.0, 6.0, 5.0)
        self.assertAlmostEqual(diagnostics[6], 0.25, delta=0.01)
        self.assertIsNone(assess(points)[0])

    def test_three_coherent_quartiles_are_counted_without_a_random_phase_angle(self):
        phase = np.concatenate((np.zeros(300), np.tile(np.arange(4) * math.pi / 2.0, 25)))
        self.assertAlmostEqual(
            radar.compute_temporal_stability(np.arange(400), phase, np.ones(400)), 0.75
        )
        phase = np.repeat(np.arange(4) * math.pi / 2.0, 100)
        self.assertLess(
            radar.compute_temporal_stability(np.arange(400), phase, np.ones(400)), 1e-12
        )

    def test_coherent_but_phase_shifted_holdout_is_rejected(self):
        training = rotating_beam_points(cycles=100)
        validation = rotating_beam_points(cycles=100, phase_offset=math.pi / 2.0)
        self.assertAlmostEqual(
            radar.evaluate_site_diagnostics(validation, 51.0, 6.0, 5.0)[0], 1.0
        )
        row, _, _ = radar.evaluate_candidate(
            np.vstack((training, validation)), (51.0, 6.0, 5.0), "II1", 1.0, 1,
            training_points=training, validation_points=validation,
        )
        self.assertIsNone(row)

    def test_validation_requires_two_hundred_unfiltered_bursts(self):
        training = rotating_beam_points(cycles=100)
        for count in (199, 200):
            with self.subTest(count=count):
                validation = training[np.linspace(0, len(training) - 1, count).astype(int)].copy()
                validation[:, 0] += 10.0
                validation[:, 4] += 1000
                row, _, _ = radar.evaluate_candidate(
                    np.vstack((training, validation)), (51.0, 6.0, 5.0), "II1", 1.0, 1,
                    training_points=training, validation_points=validation,
                )
                self.assertEqual(row is not None, count == 200)

    def test_coherent_radar_is_localized_and_verified_without_refitting_holdout(self):
        points = rotating_beam_points(cycles=100)
        training, validation = radar.split_search_validation(points)
        site = radar.search_candidate(training, 50000)
        self.assertIsNotNone(site)
        row, _, _ = radar.evaluate_candidate(
            points, site, "II1", 1.0, 1, training_points=training, validation_points=validation
        )
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["confidence_score"], 70.0)
        self.assertLess(radar.haversine_km(51.0, 6.0, row["lat_r"], row["lon_r"]), 0.1)
        self.assertAlmostEqual(row["period_s"], 5.0, delta=0.00002)
        self.assertEqual(row["n_validation_bursts"], len(validation))
        self.assertEqual(row["n_validation_inliers"], len(validation))
        self.assertEqual(row["n_inliers"], len(points))
        self.assertGreater(row["temporal_stability"], 0.999)
        for key in ("range_p05_km", "range_median_km", "range_p95_km"):
            self.assertAlmostEqual(row[key], 80.0, delta=0.1)

    def test_locally_dominant_cochannel_radars_survive_global_dilution(self):
        sites = [(50.0, 4.0, 5.0), (51.4, 6.6, 5.0), (53.5, 9.0, 5.0)]
        sources = []
        for i, site in enumerate(sites):
            points = rotating_beam_points(
                40, site[0], site[1], cycles=60, phase_offset=i * 0.8
            )
            points[:, 4] += i * 1000
            sources.append(points)
        points = np.vstack(sources)
        training, validation = radar.split_search_validation(points)
        self.assertLess(
            radar.evaluate_site_diagnostics(training, *sites[0])[0], radar.TILE_TRIGGER_RBAR
        )
        candidates = radar.search_spatial_tiles(
            points, "II1", 1.0, training_points=training, validation_points=validation
        )
        verified = radar.deduplicate_and_filter(candidates)
        for site in sites:
            with self.subTest(site=site):
                nearby = [
                    row for row in verified
                    if radar.haversine_km(site[0], site[1], row["lat_r"], row["lon_r"]) < 1.0
                ]
                self.assertEqual(len(nearby), 1)
                self.assertGreater(nearby[0]["R_bar_max"], 0.8)
                self.assertGreaterEqual(nearby[0]["n_validation_bursts"], 200)

    def test_peeling_changes_only_training_data_and_never_the_holdout(self):
        points = rotating_beam_points(cycles=100)
        events = []
        for row in points:
            ticks = round((row[0] - radar.T0_EPOCH) * 1e9)
            for offset in (0, 10_000_000):
                mono = ticks + offset
                events.append((f"{int(row[4]):06X}", int(radar.T0_EPOCH * 1e9) + mono,
                               mono, mono, row[1], row[2]))
        events.sort(key=lambda event: event[2])

        def evaluate(points, site, code, ratio, peel_pass, **kwargs):
            training = kwargs["training_points"]
            mask = np.arange(len(training)) < 300
            return None, mask, 0.6

        with (
            patch.object(radar, "search_candidate", return_value=(51.0, 6.0, 5.0)) as search,
            patch.object(radar, "evaluate_candidate", side_effect=evaluate) as evaluation,
        ):
            radar.process_one_code((("II", 1), [], events, (0, 0, 1e9, 0)))
        self.assertEqual(search.call_count, 3)
        holdout = evaluation.call_args_list[0].kwargs["validation_points"]
        for i, call in enumerate(evaluation.call_args_list):
            self.assertIs(call.kwargs["validation_points"], holdout)
            train = call.kwargs["training_points"]
            self.assertEqual(len(train), len(search.call_args_list[0].args[0]) - i * 300)
            self.assertEqual(len(radar.split_search_validation(train)[1]), 0)
        self.assertEqual(len(radar.split_search_validation(holdout)[0]), 0)

    def test_tile_trigger_counts_full_pool_not_only_the_training_half(self):
        events = []
        for index in range(40000):
            for offset in (0, 10_000_000):
                ticks = index * 2_000_000_000 + offset
                events.append(("000001", int(radar.T0_EPOCH * 1e9) + ticks,
                               ticks, ticks, 51.0, 6.0))
        for count, rbar, expected_calls in ((39999, 0.34, 0), (40000, 0.35, 0), (40000, 0.349, 1)):
            with self.subTest(count=count, rbar=rbar):
                def evaluate(points, site, code, ratio, peel_pass, **kwargs):
                    return None, np.zeros(len(kwargs["training_points"]), dtype=bool), rbar

                with (
                    patch.object(radar, "search_candidate", return_value=(51.0, 6.0, 5.0)),
                    patch.object(radar, "evaluate_candidate", side_effect=evaluate),
                    patch.object(radar, "search_spatial_tiles", return_value=[]) as tiles,
                ):
                    radar.process_one_code((
                        ("II", 1), [], events[:count * 2], (0, 0, 1e9, 0),
                    ))
                self.assertEqual(tiles.call_count, expected_calls)
                if expected_calls:
                    self.assertEqual(len(tiles.call_args.args[0]), 40000)
                    training = tiles.call_args.kwargs["training_points"]
                    validation = tiles.call_args.kwargs["validation_points"]
                    self.assertLess(len(training), 40000)
                    self.assertEqual(len(training) + len(validation), 40000)


class OutputTests(unittest.TestCase):
    def test_v5_outputs_include_validation_metrics_and_complete_stdout(self):
        row = assess(rotating_beam_points(cycles=100))[0]
        self.assertIsNotNone(row)
        future = Mock()
        future.result.return_value = ([row, dict(row, code="II2", confidence_score=69.9)], 0, 0, [])
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(radar, "OUTPUT", output),
                patch.object(radar, "archived_segments", return_value=[]),
                patch.object(radar, "read_active_segment", return_value=({}, None)),
                patch.object(radar, "ProcessPoolExecutor"),
                patch.object(radar, "as_completed", return_value=[future]),
                contextlib.redirect_stdout(stdout),
            ):
                radar.main()
            rows = list(csv.DictReader(io.StringIO(
                (output / "radar-triangulation-v5-verified.csv").read_text()
            )))
            self.assertEqual(len(rows), 1)
            for key in ("R_bar_train", "n_validation_bursts", "n_validation_inliers",
                        "validation_phase_delta_deg", "temporal_stability", "n_coverage_cells"):
                self.assertIn(key, rows[0])
            self.assertEqual(
                list(csv.DictReader(io.StringIO(stdout.getvalue().split("\nVERIFIED RADARS\n")[1]))),
                rows,
            )
            text = "".join(ET.parse(
                output / "radar-triangulation-v5-verified.svg"
            ).getroot().itertext())
            self.assertIn("II1", text)
            self.assertNotIn("II2", text)


if __name__ == "__main__":
    unittest.main()
