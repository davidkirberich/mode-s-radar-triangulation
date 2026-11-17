import math
import unittest

import numpy as np

import radar_triangulation_v3 as radar


def rotating_beam_points(distance_km):
    count = 720
    bearing = np.arange(count) * (2 * math.pi / count)
    lat0, lon0 = np.deg2rad([51.0, 6.0])
    distance = distance_km / 6371.0
    lat = np.arcsin(
        np.sin(lat0) * np.cos(distance)
        + np.cos(lat0) * np.sin(distance) * np.cos(bearing)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(distance) * np.cos(lat0),
        np.cos(distance) - np.sin(lat0) * np.sin(lat),
    )
    times = radar.T0_EPOCH + 5.0 * (np.arange(count) * 50 + bearing / (2 * math.pi))
    return np.column_stack((times, np.rad2deg(lat), np.rad2deg(lon), np.full(count, 2)))


class RadarDiagnosticsTests(unittest.TestCase):
    def test_coherent_near_and_far_peaks_get_normalized_scores(self):
        for distance in (80, 320, 400):
            with self.subTest(distance=distance):
                points = rotating_beam_points(distance)
                rbar, _, inliers, spread, sharpness, period_contrast = (
                    radar.evaluate_site_diagnostics(points, 51.0, 6.0, 5.0)
                )
                score = radar.compute_confidence_score(
                    rbar, sharpness, period_contrast, spread, 1.0,
                    int(np.sum(inliers)), 51.0, 6.0, 1,
                )
                self.assertGreaterEqual(sharpness, 0.49)
                self.assertGreaterEqual(period_contrast, 0.25)
                self.assertGreaterEqual(score, 60.0)
                if distance == 80:
                    self.assertGreaterEqual(score, radar.MIN_CONFIDENCE_SCORE)

    def test_flat_period_peak_does_not_pass_score_threshold(self):
        points = rotating_beam_points(320)
        points[:, 0] = radar.T0_EPOCH + np.remainder(points[:, 0] - radar.T0_EPOCH, 5.0)
        rbar, _, inliers, spread, sharpness, period_contrast = (
            radar.evaluate_site_diagnostics(points, 51.0, 6.0, 5.0)
        )
        score = radar.compute_confidence_score(
            rbar, sharpness, period_contrast, spread, 1.0,
            int(np.sum(inliers)), 51.0, 6.0, 1,
        )
        self.assertLess(period_contrast, 0.01)
        self.assertLess(score, radar.MIN_CONFIDENCE_SCORE)


if __name__ == "__main__":
    unittest.main()
