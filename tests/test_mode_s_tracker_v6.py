import contextlib
import csv
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mode_s_tracker_v6 as tracker


def radars():
    return {
        "SI10": tracker.Radar("SI10", 50.8, 5.8, 5, 180, 1, 420, 0.9, math.radians(1)),
        "SI44": tracker.Radar("SI44", 51.6, 6.8, 5, 180, 1, 420, 0.9, math.radians(1)),
    }


def moving_aircraft(noise=0.003, altitude=True):
    sites = radars()
    rng = np.random.default_rng(42)
    bearings, obs = [], []
    for index in range(121):
        delta = index * 2.5
        utc = tracker.T0_EPOCH + 1000 + delta
        east, north = 70000 + 140 * delta, -20000 + 70 * delta
        lat, lon = tracker.horizontal_position(east, north)
        code = ("SI10", "SI44")[index % 2]
        radar = sites[code]
        angle = tracker.true_bearing(radar.lat, radar.lon, lat, lon) + rng.normal(0, noise)
        bearings.append(tracker.Bearing(utc, code, angle, 3))
        if altitude:
            obs.append(tracker.Observation(utc, "A", (3000 + 2 * delta,)))
    return sites, bearings, obs


def parity_message(prefix, address=0):
    parity = tracker.modes_crc(prefix + b"\0\0\0") ^ address
    return prefix + parity.to_bytes(3, "big")


def ac13(feet):
    n = (feet + 1000) // 25
    return ((n & 0x7E0) << 2) | ((n & 0x10) << 1) | (n & 15) | 0x10


def beast(payload, ticks):
    body = ticks.to_bytes(6, "big") + b"\x40" + payload
    return b"\x1a" + bytes((0x33 if len(payload) == 14 else 0x32,)) + body.replace(b"\x1a", b"\x1a\x1a")


class GeometryTests(unittest.TestCase):
    def test_standalone_entrypoint_starts_without_unittest_import_side_effects(self):
        result = subprocess.run([sys.executable, tracker.__file__, "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--radars", result.stdout)

    def test_wgs84_round_trip_and_ground_footpoint(self):
        for lat, lon, alt in ((51.145, 6.355, 0), (51.4, 6.5, 12000),
                              (48.5, 2.5, 2000), (53.8, 9.5, 10000)):
            with self.subTest(lat=lat):
                enu = tracker.geodetic_to_enu(lat, lon, alt)
                recovered = tracker.enu_to_geodetic(*enu)
                np.testing.assert_allclose(recovered, [lat, lon, alt], atol=1e-5)
                surface = tracker.geodetic_to_enu(lat, lon)
                p = tracker.horizontal_position(*surface[:2])
                np.testing.assert_allclose(p, [lat, lon], atol=1e-8)

    def test_phase_epoch_and_angle_wrap_match_radar_calibration(self):
        radar = radars()["SI10"]
        self.assertAlmostEqual(radar.angle(tracker.T0_EPOCH + 1.25), math.radians(270))
        self.assertEqual(tracker.wrap(-math.pi), math.pi)
        self.assertAlmostEqual(tracker.wrap(math.radians(359)), math.radians(-1))

    def test_ground_velocity_round_trip_includes_surface_curvature(self):
        for lat, lon in ((51.145, 6.355), (48.5, 2.5), (54.5, 10.5)):
            for track in (0, 45, 180, 270):
                east, north = tracker.velocity_enu(lat, lon, track, 230)
                recovered = tracker.velocity_true(lat, lon, east, north)
                angle = math.radians(track)
                np.testing.assert_allclose(
                    recovered, [230 * math.sin(angle), 230 * math.cos(angle)], atol=1e-10)

    def test_same_radar_or_parallel_rays_cannot_initialize(self):
        sites, rows, _ = moving_aircraft()
        self.assertIsNone(tracker.initial_position([rows[0], rows[2]], sites))
        same = dict(sites, SI44=tracker.replace(sites["SI44"], lat=50.8, lon=5.8))
        rays = [tracker.Bearing(10, "SI10", 1.0, 2), tracker.Bearing(11, "SI44", 1.0, 2)]
        self.assertIsNone(tracker.initial_position(rays, same))


class DecoderTests(unittest.TestCase):
    def test_crc_and_address_parity_match_real_df17_and_known_address(self):
        payload = bytes.fromhex("8D40621D58C382D690C8AC2863A7")
        self.assertEqual(tracker.modes_crc(payload), 0)
        ac = ac13(25000)
        prefix = bytes((0x20, 0, ac >> 8, ac & 255))
        message = parity_message(prefix, 0xABC123)
        self.assertEqual(tracker.modes_crc(message), 0xABC123)
        self.assertEqual(tracker.decode_altitude(int.from_bytes(message[2:4], "big") & 8191), 25000)
        self.assertIsNone(tracker.decode_altitude(0))
        self.assertIsNone(tracker.decode_altitude(0x40))

    def test_global_cpr_known_pair(self):
        def fractions(hexadecimal, utc):
            payload = bytes.fromhex(hexadecimal)
            return (utc, (((payload[6] & 3) << 15) | payload[7] << 7 | payload[8] >> 1) / 131072,
                    (((payload[8] & 1) << 16) | payload[9] << 8 | payload[10]) / 131072)
        even = fractions("8D40621D58C382D690C8AC2863A7", 1)
        odd = fractions("8D40621D58C386435CC412692AD6", 0)
        lat, lon = tracker.cpr_position(even, odd)
        self.assertAlmostEqual(lat, 52.2572021484375, places=8)
        self.assertAlmostEqual(lon, 3.91937255859375, places=8)
        self.assertIsNone(tracker.cpr_position(even, (20, odd[1], odd[2])))

    def test_bds50_ground_velocity_and_ambiguous_register_rejection(self):
        def message(sign, track):
            fields = [(1, 1, 1), (12, 12, 1), (24, 24, 1), (35, 35, 1), (46, 46, 1),
                      (13, 13, sign), (14, 23, track), (25, 34, 200), (47, 56, 205)]
            mb = sum(value << (56 - end) for _, end, value in fields)
            return b"\xa0\0\0\0" + mb.to_bytes(7, "big") + b"\0\0\0"
        velocity, status = tracker.decode_bds50(message(0, 512))
        self.assertEqual(status, "bds50_velocity")
        self.assertEqual(velocity, (90.0, 400 * tracker.KT_MS))
        velocity, status = tracker.decode_bds50(message(1, 250))
        self.assertIsNone(velocity)
        self.assertEqual(status, "commb_ambiguous_50_60")

    def test_sqlite_bursts_include_null_positions_and_use_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.sqlite"
            with contextlib.closing(sqlite3.connect(path)) as db:
                db.executescript(
                    "CREATE TABLE df11_events (icao24 TEXT,recv_ns INTEGER,monotonic_ns INTEGER,"
                    "mlat_ticks INTEGER,code_type TEXT,code_val INTEGER,lat REAL,lon REAL);"
                    "CREATE INDEX df11_code_time ON df11_events(code_type,code_val,monotonic_ns);")
                for index in range(8):
                    code = 10 if index % 2 == 0 else 44
                    for offset in (0, 100_000_000, 400_000_000):
                        ticks = index * 2_000_000_000 + offset
                        db.execute("INSERT INTO df11_events VALUES (?,?,?,?,?,?,NULL,NULL)",
                                   ("ABC123", int(tracker.T0_EPOCH * 1e9) + ticks, ticks,
                                    ticks, "SI", code))
                db.commit()
            fragments = tracker.load_burst_fragments((dict(db=path, clock=(0, 0, 1e9, 0)), radars()))
            bursts = tracker.merge_bursts(fragments, radars())["ABC123"]
            self.assertEqual(len(bursts), 8)
            self.assertEqual(bursts[0].count, 3)
            self.assertAlmostEqual(bursts[0].utc - tracker.T0_EPOCH, 0.1, places=6)

    def test_raw_cache_classifies_unpaired_df17_and_extracts_pressure_height(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "capture.raw"
            payload = bytes.fromhex("8D40621D58C382D690C8AC2863A7")
            ac = ac13(30000)
            altitude = parity_message(bytes((0x20, 0, ac >> 8, ac & 255)), 0x40621D)
            data = beast(payload, 1_000_000_000) + beast(altitude, 2_000_000_000)
            raw.write_bytes(data)
            segment = dict(raw=raw, clock=(0, 0, 1e9, 0), ends=[len(data)],
                           chunks=[dict(realtime_ns=int(tracker.T0_EPOCH * 1e9), monotonic_ns=0)])
            path, counts = tracker.decode_raw_segment((segment, {"40621D"}, root))
            self.assertEqual(counts["altitude_updates"], 1)
            with tracker.connect_readonly(path) as db:
                self.assertEqual(db.execute("SELECT icao FROM df17_position").fetchone()[0], "40621D")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM reference").fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT a FROM obs WHERE kind='A'").fetchone()[0],
                                 30000 * tracker.FT_M)
            self.assertEqual(tracker.decode_raw_segment((segment, {"40621D"}, root))[0], path)


class FilterTests(unittest.TestCase):
    def test_reinitializes_after_prolonged_rejected_bearings(self):
        sites, bearings, obs = moving_aircraft()
        original = tracker.bearing_update
        start = bearings[0].utc

        def outage(state, covariance, row, radar):
            if 50 <= row.utc - start <= 212.5:
                return state, covariance, False
            return original(state, covariance, row, radar)

        with patch.object(tracker, "load_aircraft", return_value=obs), patch.object(
                tracker, "bearing_update", side_effect=outage):
            _, tracks, _, _, counts, _ = tracker.run_aircraft(
                ("ABC123", bearings, [], sites, False))
        recovered = [points for _, points in tracks if points]
        self.assertEqual(len(recovered), 2)
        self.assertEqual(counts["track_lost"], 1)
        self.assertNotIn("_resume_at", counts)
        self.assertEqual(recovered[-1][-1]["utc"], bearings[-1].utc)
        self.assertLess(recovered[0][-1]["utc"], recovered[1][0]["utc"])

    def test_blind_ekf_rts_tracks_motion_height_and_rejects_bearing_outlier(self):
        sites, bearings, obs = moving_aircraft()
        bad = bearings[40]
        bearings[40] = tracker.replace(bad, angle=bad.angle + math.pi / 2)
        points, counts = tracker.track_flight(bearings, obs, sites)
        self.assertGreater(len(points), 100)
        self.assertGreaterEqual(counts["B_rejected"], 1)
        errors = []
        for point in points:
            delta = point["utc"] - tracker.T0_EPOCH - 1000
            actual = tracker.geodetic_to_enu(point["lat"], point["lon"])[:2]
            errors.append(np.linalg.norm(actual - [70000 + 140 * delta, -20000 + 70 * delta]))
            self.assertLess(abs(point["alt_ft"] * tracker.FT_M - (3000 + 2 * delta)), 10)
            self.assertGreater(point["pos_sigma_km"], 0)
        self.assertLess(float(np.sqrt(np.mean(np.asarray(errors) ** 2))), 1500)
        self.assertGreaterEqual(points[-1]["n_updates"], 6)

    def test_rts_improves_noisy_bearing_only_solution_without_inventing_altitude(self):
        sites, bearings, obs = moving_aircraft(noise=0.01, altitude=False)
        smooth, _ = tracker.track_flight(bearings, obs, sites)
        with patch.object(tracker, "rts_smooth", side_effect=lambda filtered, *_: filtered):
            filtered, _ = tracker.track_flight(bearings, obs, sites)

        def error(points):
            return np.mean([
                np.sum((tracker.geodetic_to_enu(p["lat"], p["lon"])[:2] - [
                    70000 + 140 * (p["utc"] - tracker.T0_EPOCH - 1000),
                    -20000 + 70 * (p["utc"] - tracker.T0_EPOCH - 1000)]) ** 2)
                for p in points
            ])
        self.assertTrue(smooth)
        self.assertLess(error(smooth), error(filtered))
        self.assertTrue(all(point["alt_ft"] is None for point in smooth))

    def test_reference_positions_are_not_tracking_observations(self):
        sites, bearings, _ = moving_aircraft()
        with self.assertRaisesRegex(ValueError, "Only DF4"):
            tracker.track_flight(bearings, [
                tracker.Observation(bearings[0].utc, "R", (51.0, 6.0))
            ], sites)

    def test_calibration_and_validation_aircraft_are_disjoint(self):
        sites, rows, _ = moving_aircraft()
        bursts = {f"{i:06X}": rows for i in range(40)}
        known = set(list(bursts)[:30])
        calibration, validation, mode_s = tracker.select_cohorts(bursts, known, known)
        self.assertFalse(set(calibration) & set(validation))
        self.assertEqual(set(calibration) | set(validation), known)
        self.assertEqual(set(mode_s), set(bursts) - known)

    def test_noise_calibration_uses_only_requested_aircraft(self):
        sites = radars()
        reference = [(tracker.T0_EPOCH + 1000 + i, 51.2, 6.4) for i in range(201)]
        bursts = {}
        for icao in ("AAAAAA", "BBBBBB"):
            rows = []
            for index in range(200):
                for code, radar in sites.items():
                    angle = tracker.true_bearing(radar.lat, radar.lon, 51.2, 6.4)
                    rows.append(tracker.Bearing(reference[index][0], code,
                                                 angle + math.radians((-1) ** index), 2))
            bursts[icao] = rows
        with patch.object(tracker, "load_aircraft", return_value=reference) as load:
            calibrated, report = tracker.calibrate_noise(sites, bursts, [], list(bursts))
        self.assertEqual({call.args[1] for call in load.call_args_list}, set(bursts))
        for code in sites:
            self.assertGreater(calibrated[code].sigma, math.radians(1))
            self.assertEqual(report[code]["aircraft"], 2)

    def test_radars_prefer_v5_and_reject_ambiguous_duplicate_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "radar-triangulation-v5-verified.csv"
            columns = ["code", "confidence_score", "lat_r", "lon_r", "period_s", "phi0_deg",
                       "range_p05_km", "range_p95_km", "R_bar_max"]
            rows = [["SI10", 90, 51, 6, 5, 180, 1, 420, 0.9],
                    ["SI44", 90, 52, 7, 5, 180, 1, 420, 0.9]]
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
            loaded, chosen = tracker.load_radars(root)
            self.assertEqual(chosen, path)
            self.assertEqual(len(loaded), 2)
            with path.open("a", newline="") as stream:
                csv.writer(stream).writerow(rows[0])
            with self.assertRaisesRegex(ValueError, "Multiple physical radars"):
                tracker.load_radars(root)

    def test_statistics_are_pooled_by_positions_not_averaged_flight_medians(self):
        result = tracker.error_statistics([1, 2, 3, 10])
        self.assertEqual(result["median_km"], 2.5)
        self.assertEqual(result["mean_km"], 4)
        self.assertAlmostEqual(result["rmse_km"], math.sqrt(28.5))
        self.assertAlmostEqual(result["p90_km"], 7.9)
        self.assertIsNone(tracker.error_statistics([])["rmse_km"])

    def test_cli_coordinates_update_receiver_geometry_and_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            with (
                patch.dict(os.environ, {"ADSB_RX_LAT": "1", "ADSB_RX_LON": "2"}),
                patch.object(tracker, "RX_LAT", tracker.RX_LAT),
                patch.object(tracker, "RX_LON", tracker.RX_LON),
                patch.object(tracker, "RX_ECEF", tracker.RX_ECEF),
                patch.object(tracker, "RX_ROT", tracker.RX_ROT),
                patch.object(tracker, "create_run_directory", return_value=run) as create,
                patch.object(tracker, "execute") as execute,
            ):
                tracker.main(["--archive", str(root / "input"), "--output", str(root / "output"),
                              "--rx-lat", "52", "--rx-lon", "7"])
                args = execute.call_args.args[0]
                self.assertEqual(args.archive, root / "input")
                self.assertEqual(args.output_root, root / "output")
                self.assertEqual(args.output, run)
                self.assertEqual(create.call_args.args[2], root / "output")
                np.testing.assert_allclose(tracker.RX_ECEF, tracker.ecef(52, 7))
                np.testing.assert_allclose(tracker.RX_ROT, tracker.enu_rotation(52, 7))
                self.assertEqual(os.environ["ADSB_RX_LAT"], "52.0")
                self.assertEqual(os.environ["ADSB_RX_LON"], "7.0")


if __name__ == "__main__":
    unittest.main()
