import contextlib
import csv
import io
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

import radar_triangulation_v4 as radar


def rotating_beam_points(distance_km=80.0, lat_r=51.0, lon_r=6.0,
                         bearings=72, cycles=10, phase_offset=0.0):
    angles = np.tile(np.arange(bearings) * radar.PI2 / bearings, cycles)
    lat0, lon0 = np.deg2rad([lat_r, lon_r])
    distance = np.asarray(distance_km) / 6371.0
    lat = np.arcsin(
        np.sin(lat0) * np.cos(distance)
        + np.cos(lat0) * np.sin(distance) * np.cos(angles)
    )
    lon = lon0 + np.arctan2(
        np.sin(angles) * np.sin(distance) * np.cos(lat0),
        np.cos(distance) - np.sin(lat0) * np.sin(lat),
    )
    rotations = np.arange(cycles) // 2 * 100 + np.arange(cycles) % 2
    times = radar.T0_EPOCH + 5.0 * (
        np.repeat(rotations, bearings) + (angles - phase_offset) / radar.PI2
    )
    return np.column_stack((
        times, np.rad2deg(lat), np.rad2deg(lon),
        np.full(len(angles), 2.0), np.tile(np.arange(bearings), cycles),
    ))


class RadarDiagnosticsTests(unittest.TestCase):
    def test_eight_geodesic_controls_use_percentile_not_max_or_mean(self):
        points = rotating_beam_points(bearings=8, cycles=4)
        for ring_rbars in (
            [1.0, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79],
            [1.0, 1.0, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79],
        ):
            with self.subTest(ring_rbars=ring_rbars):
                points[:, 0] = radar.T0_EPOCH
                vectors = [np.zeros(len(points))]
                vectors.extend(
                    np.tile([math.acos(r), -math.acos(r)], len(points) // 2)
                    for r in ring_rbars
                )
                with patch.object(radar, "site_geometry", side_effect=[
                    (vector, np.full(len(points), 200.0)) for vector in vectors
                ]) as geometry:
                    diagnostics = radar.evaluate_site_diagnostics(points, 51.0, 6.0, 5.0)
                expected_drop = 1.0 - math.cos(math.atan2(120.0, 200.0))
                expected = min(1.5, (1.0 - np.percentile(ring_rbars, 80)) / expected_drop)
                self.assertAlmostEqual(diagnostics[4], expected, places=12)
                self.assertEqual(geometry.call_count, 9)
                for k, call in enumerate(geometry.call_args_list[1:]):
                    _, lat, lon = call.args
                    self.assertAlmostEqual(radar.haversine_km(51.0, 6.0, lat, lon), 120.0, places=8)
                    dlon = math.radians(lon - 6.0)
                    azimuth = math.degrees(math.atan2(
                        math.sin(dlon) * math.cos(math.radians(lat)),
                        math.cos(math.radians(51.0)) * math.sin(math.radians(lat))
                        - math.sin(math.radians(51.0)) * math.cos(math.radians(lat)) * math.cos(dlon),
                    )) % 360.0
                    self.assertAlmostEqual(azimuth, k * 45.0, places=8)

    def test_coherent_near_and_far_peaks_remain_distance_normalized(self):
        for distance in (80, 320, 400):
            with self.subTest(distance=distance):
                points = rotating_beam_points(distance)
                rbar, _, mask, spread, sharpness, contrast, stability = (
                    radar.evaluate_site_diagnostics(points, 51.0, 6.0, 5.0)
                )
                self.assertEqual(int(mask.sum()), len(points))
                self.assertAlmostEqual(stability, 1.0, places=10)
                self.assertGreaterEqual(sharpness, 0.49)
                self.assertGreaterEqual(contrast, 0.25)
                score = radar.compute_confidence_score(
                    rbar, sharpness, contrast, spread, 1.0, len(points),
                    51.0, 6.0, 1, stability,
                )
                self.assertGreaterEqual(score, 60.0)
                if distance == 80:
                    self.assertGreaterEqual(score, radar.MIN_CONFIDENCE_SCORE)

    def test_temporal_stability_sorts_chronologically_and_wraps_phase(self):
        times = np.arange(16)
        phases = np.deg2rad(np.repeat([359.0, 1.0, 359.0, 1.0], 4))
        order = np.random.default_rng(42).permutation(len(times))
        result = radar.compute_temporal_stability(
            times[order], phases[order], np.ones(len(times))
        )
        self.assertAlmostEqual(result, math.cos(math.radians(1.0)), places=12)

    def test_temporal_stability_rejects_drifting_or_single_segment_phases(self):
        times = np.arange(16)
        weights = np.ones(16)
        drifting = np.repeat(np.arange(4) * math.pi / 2.0, 4)
        self.assertLess(radar.compute_temporal_stability(times, drifting, weights), 1e-12)
        single_segment = np.concatenate((np.zeros(4), np.tile(np.arange(4) * math.pi / 2.0, 3)))
        self.assertLessEqual(
            radar.compute_temporal_stability(times, single_segment, weights), 0.25
        )
        self.assertEqual(radar.compute_temporal_stability(times[:3], drifting[:3], weights[:3]), 0.0)

    def test_temporal_quartile_vectors_are_weighted(self):
        phases = np.tile([0.0, 0.0, math.pi, math.pi], 4)
        weights = np.tile([12.0, 12.0, 1.0, 1.0], 4)
        self.assertAlmostEqual(
            radar.compute_temporal_stability(np.arange(16), phases, weights), 1.0
        )

    def test_confidence_uses_square_root_of_stability(self):
        args = (0.95, 1.0, 0.5, 1.0, 1.0, 10000, 51.0, 6.0, 1)
        stable = radar.compute_confidence_score(*args, 1.0)
        self.assertEqual(radar.compute_confidence_score(*args, 0.25), stable / 2.0)
        self.assertEqual(radar.compute_confidence_score(*args, 0.0), 0.0)

    def test_inlier_ranges_and_cell_counts_exclude_unmatched_or_distant_bursts(self):
        inliers = rotating_beam_points()
        points = np.vstack((
            inliers, rotating_beam_points(600, bearings=10, cycles=2),
            rotating_beam_points(80, bearings=10, cycles=2, phase_offset=math.pi),
        ))
        row, mask, _ = radar.evaluate_candidate(points, (51.0, 6.0, 5.0), "II1", 1.0, 1)
        self.assertIsNotNone(row)
        self.assertEqual(int(mask.sum()), len(inliers))
        self.assertEqual(row["n_inliers"], len(inliers))
        for key in ("range_p05_km", "range_median_km", "range_p95_km"):
            self.assertEqual(row[key], 80.0)
        cells = {
            (math.floor((lat - 42.0) / 0.0833), math.floor((lon + 15.0) / 0.1253))
            for lat, lon in inliers[:, 1:3]
        }
        self.assertEqual(row["n_coverage_cells"], len(cells))

    def test_range_percentiles_and_floor_cells_include_negative_indices(self):
        points = rotating_beam_points(np.array([100, 200, 300, 400]), bearings=4, cycles=1)
        stats = radar.inlier_coverage_statistics(points, 51.0, 6.0)
        self.assertEqual([stats[k] for k in (
            "range_p05_km", "range_median_km", "range_p95_km"
        )], [115.0, 250.0, 385.0])
        points[:, 1:3] = [
            [42.01, -14.99], [42.02, -14.98], [41.99, -15.01], [42.09, -14.99],
        ]
        self.assertEqual(
            radar.inlier_coverage_statistics(points, 51.0, 6.0)["n_coverage_cells"], 3
        )


class SpatialSearchTests(unittest.TestCase):
    def test_search_runs_real_coarse_fine_and_four_stage_refinement(self):
        points = rotating_beam_points(bearings=24, cycles=8)
        bounds = (49.0, 51.6, 3.0, 6.6)
        site = radar.search_candidate(points, 25000, bounds=bounds)
        self.assertIsNotNone(site)
        self.assertLess(radar.haversine_km(51.0, 6.0, site[0], site[1]), 1.0)
        self.assertAlmostEqual(site[2], 5.0, delta=0.00002)

    def test_local_subsample_limit_and_bounds_reach_all_search_stages(self):
        points = np.resize(rotating_beam_points(), (30001, 5))
        site = (51.0, 6.0, 5.0)
        bounds = (50.2, 52.8, 4.7, 8.3)
        with (
            patch.object(radar, "grid_search_weighted", return_value=(1.0, *site)) as grid,
            patch.object(radar, "refine_continuous", return_value=site) as refine,
        ):
            self.assertEqual(radar.search_candidate(points, 25000, bounds=bounds), site)
        self.assertEqual(grid.call_count, 2)
        for call in [*grid.call_args_list, refine.call_args]:
            self.assertEqual(len(call.args[0]), 25000)
            self.assertEqual(call.kwargs["bounds"], bounds)
        self.assertGreater(grid.call_args.args[0][-1, 0], points[0, 0])

    def test_tiles_select_full_pool_inliers_using_local_phase_and_range(self):
        local = rotating_beam_points(40, 50.0, 4.0, bearings=192, cycles=8)
        outside_tile = rotating_beam_points(200, 50.0, 4.0, bearings=1, cycles=720)
        too_far = rotating_beam_points(600, 50.0, 4.0, bearings=1, cycles=720)
        wrong_phase = rotating_beam_points(40, 50.0, 4.0, bearings=10, cycles=10,
                                          phase_offset=math.pi)
        points = np.vstack((local, outside_tile, too_far, wrong_phase))
        with patch.object(radar, "search_candidate", wraps=radar.search_candidate) as search:
            candidates = radar.search_spatial_tiles(points, "II1", 1.0)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(len(search.call_args.args[0]), len(local) + len(wrong_phase))
        self.assertEqual(search.call_args.args[1], 25000)
        self.assertEqual(len(candidates), 1)
        row = candidates[0]
        self.assertEqual(row["code"], "II1@T11")
        self.assertEqual(row["n_inliers"], len(local) + len(outside_tile))
        self.assertAlmostEqual(row["range_p05_km"], 40.0, delta=0.1)
        self.assertAlmostEqual(row["range_p95_km"], 200.0, delta=0.1)
        self.assertGreater(row["temporal_stability"], 0.999)
        self.assertLess(radar.haversine_km(50.0, 4.0, row["lat_r"], row["lon_r"]), 0.1)

    def test_nine_overlapping_tiles_cover_domain_and_skip_small_subsets(self):
        lat, lon = np.meshgrid(np.linspace(48.5, 54.5, 61), np.linspace(2.5, 10.5, 81))
        points = np.column_stack((
            np.zeros(lat.size), lat.ravel(), lon.ravel(), np.ones(lat.size), np.zeros(lat.size),
        ))
        # Five copies put every tile above the actual 1,500-burst threshold.
        points = np.tile(points, (5, 1))
        with patch.object(radar, "search_candidate", return_value=None) as search:
            self.assertEqual(radar.search_spatial_tiles(points, "II1", 1.0), [])
        self.assertEqual(search.call_count, 9)
        covered = np.zeros(len(points), dtype=bool)
        bounds_list = [call.kwargs["bounds"] for call in search.call_args_list]
        for call, (la0, la1, lo0, lo1) in zip(search.call_args_list, bounds_list):
            self.assertAlmostEqual(la1 - la0, 2.6)
            self.assertAlmostEqual(lo1 - lo0, 3.6)
            mask = ((points[:, 1] >= la0) & (points[:, 1] <= la1)
                    & (points[:, 2] >= lo0) & (points[:, 2] <= lo1))
            covered |= mask
            np.testing.assert_array_equal(call.args[0], points[mask])
        self.assertTrue(covered.all())
        self.assertLess(bounds_list[3][0], bounds_list[0][1])
        self.assertLess(bounds_list[1][2], bounds_list[0][3])
        with patch.object(radar, "search_candidate") as search:
            radar.search_spatial_tiles(points[:1499], "II1", 1.0)
        search.assert_not_called()

    def test_tile_trigger_thresholds_on_first_global_pass(self):
        events = []
        for index in range(40000):
            for frame in range(2):
                ticks = index * 2_000_000_000 + frame * 100_000_000
                events.append(("000001", int(radar.T0_EPOCH * 1e9) + ticks,
                               ticks, ticks, 51.0, 6.0))
        for count, rbar, expected_calls in ((39999, 0.34, 0), (40000, 0.35, 0), (40000, 0.349, 1)):
            with self.subTest(count=count, rbar=rbar):
                tile_row = {"code": "II1@T11"}
                with (
                    patch.object(radar, "search_candidate", return_value=(51.0, 6.0, 5.0)),
                    patch.object(radar, "evaluate_candidate",
                                 return_value=(None, np.zeros(count, dtype=bool), rbar)),
                    patch.object(radar, "search_spatial_tiles", return_value=[tile_row]) as tiles,
                ):
                    result = radar.process_one_code((
                        ("II", 1), [], events[:count * 2], (0, 0, 1e9, 0),
                    ))
                self.assertEqual(tiles.call_count, expected_calls)
                self.assertEqual(result[0], [tile_row] if expected_calls else [])


class OutputTests(unittest.TestCase):
    def test_csv_svg_and_complete_stdout_share_verified_candidates(self):
        row, _, _ = radar.evaluate_candidate(
            rotating_beam_points(bearings=72, cycles=100), (51.0, 6.0, 5.0), "II1", 1.0, 1
        )
        self.assertGreaterEqual(row["confidence_score"], 70.0)
        duplicates = [
            dict(row, code="II1@T11", confidence_score=row["confidence_score"] - 1),
            dict(row, code="SI1", base_code="SI1", n_inliers=300, confidence_score=71.0),
            dict(row, code="II1#2", peel_pass=2, period_s=2.5, confidence_score=70.0),
            dict(row, code="II2", base_code="II2", lat_r=53.0, confidence_score=69.9),
        ]

        class FakeExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, task):
                return task

        class FakeFuture:
            def result(self):
                return [row, *duplicates], 0, 0, []

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stdout = io.StringIO()
            with (
                patch.object(radar, "OUTPUT", output),
                patch.object(radar, "archived_segments", return_value=[]),
                patch.object(radar, "read_active_segment", return_value=({}, None)),
                patch.object(radar, "ProcessPoolExecutor", FakeExecutor),
                patch.object(radar, "as_completed", return_value=[FakeFuture()]),
                contextlib.redirect_stdout(stdout),
            ):
                radar.main()
            csv_path = output / "radar-triangulation-v4-verified.csv"
            csv_text = csv_path.read_text()
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["code"], "II1")
            for key in ("temporal_stability", "range_p05_km", "range_median_km",
                        "range_p95_km", "n_coverage_cells"):
                self.assertIn(key, rows[0])
            printed_csv = stdout.getvalue().split("\nVERIFIED RADARS\n", 1)[1]
            self.assertEqual(list(csv.DictReader(io.StringIO(printed_csv))), rows)
            self.assertTrue(stdout.getvalue().endswith(printed_csv))
            svg = ET.parse(output / "radar-triangulation-v4-verified.svg")
            text = "".join(svg.getroot().itertext())
            self.assertIn("II1", text)
            self.assertNotIn("II1@T11", text)


if __name__ == "__main__":
    unittest.main()
