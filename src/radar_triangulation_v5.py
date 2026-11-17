#!/usr/bin/env python3
"""
Passive Mode-S (DF11) Radar-Triangulation (v5 — Held-Out Candidate Validation)
Hardware-Profil: 56 Kerne (2x Intel Xeon Gold), 24 GB RAM (Debian VM)
"""
import bisect
import collections
import csv
import datetime as dt
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append("/opt/adsb")
from beast_inspect import frames, interrogator
from experiment_runs import create_run_directory, parse_common_options


ARCHIVE = Path(os.environ.get("ADSB_ARCHIVE", "/var/lib/adsb/beast-experiment"))
OUTPUT = Path(os.environ.get("ADSB_OUTPUT", "/tmp/radar-triangulation/v5"))
RAW_LIMIT = 5000
BURST_GAP_S = 0.5
BACKGROUND_LIMIT = 10000
MAX_GRID_BURSTS = 50000
MAX_WORKERS = min(56, os.cpu_count() or 56)
TILE_TRIGGER_RBAR = 0.35
TILE_TRIGGER_BURSTS = 40000
TILE_MIN_BURSTS = 1500
TILE_MAX_GRID_BURSTS = 25000
TILE_LAT_SPAN, TILE_LON_SPAN = 2.6, 3.6
SHARPNESS_RADIUS_KM = 120.0
COVERAGE_LAT_ORIGIN, COVERAGE_LON_ORIGIN = 42.0, -15.0
COVERAGE_D_LAT, COVERAGE_D_LON = 0.0833, 0.1253
VALIDATION_BLOCK_S = 300.0
MIN_VALIDATION_BURSTS = 200
MIN_QUARTILE_BURSTS = 25
MIN_VALIDATION_RBAR = 0.45
MIN_TEMPORAL_STABILITY = 0.65
MAX_VALIDATION_PHASE_DELTA_DEG = 20.0

# Filter- & Konfidenz-Parameter
MIN_MAINLOBE_FRAMES = 2
MIN_MAINLOBE_RATIO = 0.12        # Unterdrückt 1-Bit-CRC-Schattencodes
MAX_BURST_WEIGHT = 12.0
MAX_PEEL_PASSES = 3
PEEL_MAX_RANGE_KM = 420.0
MIN_CONFIDENCE_SCORE = 70.0      # Schwelle für finale CSV & SVG (0..100)
T0_EPOCH = 1763078400.0          # 2025-11-14T00:00:00Z

LAT_MIN, LAT_MAX = 48.5, 54.5
LON_MIN, LON_MAX = 2.5, 10.5
RX_LAT, RX_LON = float(os.environ.get("ADSB_RX_LAT", "51.145")), float(os.environ.get("ADSB_RX_LON", "6.355"))
PI2 = 2.0 * math.pi

EUROPE_POLYS = [[[6.19,49.46],[6.66,49.2],[8.1,49.02],[7.59,48.33],[7.47,47.62],[7.19,47.45],[6.74,47.54],[6.77,47.29],[6.04,46.73],[6.02,46.27],[6.5,46.43],[6.84,45.99],[6.8,45.71],[7.1,45.33],[6.75,45.03],[7.01,44.25],[7.55,44.13],[7.44,43.69],[6.53,43.13],[4.56,43.4],[3.1,43.08],[2.99,42.47],[1.83,42.34],[0.7,42.8],[0.34,42.58],[-1.5,43.03],[-1.9,43.42],[-1.38,44.02],[-1.19,46.01],[-2.23,47.06],[-2.96,47.57],[-4.49,47.95],[-4.59,48.68],[-3.3,48.9],[-1.62,48.64],[-1.93,49.78],[-0.99,49.35],[1.34,50.13],[1.64,50.95],[2.51,51.15],[2.66,50.8],[3.12,50.78],[3.59,50.38],[4.29,49.91],[4.8,49.99],[5.67,49.53],[5.9,49.44],[6.19,49.46]],[[16.98,48.12],[16.9,47.71],[16.34,47.71],[16.53,47.5],[16.2,46.85],[16.01,46.68],[15.14,46.66],[14.63,46.43],[13.81,46.51],[12.38,46.77],[12.15,47.12],[11.16,46.94],[11.05,46.75],[10.44,46.89],[9.93,46.92],[9.48,47.1],[9.63,47.35],[9.59,47.53],[9.9,47.58],[10.4,47.3],[10.54,47.57],[11.43,47.52],[12.14,47.7],[12.62,47.67],[12.93,47.47],[13.03,47.64],[12.88,48.29],[13.24,48.42],[13.6,48.88],[14.34,48.56],[14.9,48.96],[15.25,49.04],[16.03,48.73],[16.5,48.79],[16.96,48.6],[16.88,48.47],[16.98,48.12]],[[14.12,53.76],[14.35,53.25],[14.07,52.98],[14.44,52.62],[14.69,52.09],[14.61,51.75],[15.02,51.11],[14.57,51.0],[14.31,51.12],[14.06,50.93],[13.34,50.73],[12.97,50.48],[12.24,50.27],[12.42,49.97],[12.52,49.55],[13.03,49.31],[13.6,48.88],[13.24,48.42],[12.88,48.29],[13.03,47.64],[12.93,47.47],[12.62,47.67],[12.14,47.7],[11.43,47.52],[10.54,47.57],[10.4,47.3],[9.9,47.58],[9.59,47.53],[8.52,47.83],[8.32,47.61],[7.47,47.62],[7.59,48.33],[8.1,49.02],[6.66,49.2],[6.19,49.46],[6.24,49.9],[6.04,50.13],[6.16,50.8],[5.99,51.85],[6.59,51.85],[6.84,52.23],[7.09,53.14],[6.91,53.48],[7.1,53.69],[7.94,53.75],[8.12,53.53],[8.8,54.02],[8.57,54.4],[8.53,54.96],[9.28,54.83],[9.92,54.98],[9.94,54.6],[10.95,54.36],[10.94,54.01],[11.96,54.2],[12.52,54.47],[13.65,54.08],[14.12,53.76]],[[9.59,47.53],[9.63,47.35],[9.48,47.1],[9.93,46.92],[10.44,46.89],[10.36,46.48],[9.92,46.31],[9.18,46.44],[8.97,46.04],[8.49,46.01],[8.32,46.16],[7.76,45.82],[7.27,45.78],[6.84,45.99],[6.5,46.43],[6.02,46.27],[6.04,46.73],[6.77,47.29],[6.74,47.54],[7.19,47.45],[7.47,47.62],[8.32,47.61],[8.52,47.83],[9.59,47.53]],[[6.04,50.13],[6.24,49.9],[6.19,49.46],[5.9,49.44],[5.67,49.53],[5.78,50.09],[6.04,50.13]],[[6.16,50.8],[6.04,50.13],[5.78,50.09],[5.67,49.53],[4.8,49.99],[4.29,49.91],[3.59,50.38],[3.12,50.78],[2.66,50.8],[2.51,51.15],[3.31,51.35],[4.05,51.27],[4.97,51.48],[5.61,51.04],[6.16,50.8]],[[6.91,53.48],[7.09,53.14],[6.84,52.23],[6.59,51.85],[5.99,51.85],[6.16,50.8],[5.61,51.04],[4.97,51.48],[4.05,51.27],[3.31,51.35],[3.83,51.62],[4.71,53.09],[6.07,53.51],[6.91,53.48]],[[9.92,54.98],[9.28,54.83],[8.53,54.96],[8.12,55.52],[8.09,56.54],[8.26,56.81],[8.54,57.11],[9.42,57.17],[9.78,57.45],[10.58,57.73],[10.55,57.22],[10.25,56.89],[10.37,56.61],[10.91,56.46],[10.67,56.08],[10.37,56.19],[9.65,55.47],[9.92,54.98]],[[12.37,56.11],[12.69,55.61],[12.09,54.8],[11.04,55.36],[10.9,55.78],[12.37,56.11]],[[-3.09,53.4],[-2.95,53.98],[-3.61,54.6],[-4.84,54.79],[-5.08,55.06],[-4.72,55.51],[-5.05,55.78],[-5.59,55.31],[-5.64,56.28],[-6.15,56.79],[-5.79,57.82],[-5.01,58.63],[-4.21,58.55],[-3.01,58.64],[-4.07,57.55],[-3.06,57.69],[-1.96,57.68],[-2.22,56.87],[-3.12,55.97],[-2.09,55.91],[-2.01,55.8],[-1.11,54.62],[-0.43,54.46],[0.18,53.33],[0.47,52.93],[1.68,52.74],[1.56,52.1],[1.05,51.81],[1.45,51.29],[0.55,50.77],[-0.79,50.77],[-2.49,50.5],[-2.96,50.7],[-3.62,50.23],[-4.54,50.34],[-5.25,49.96],[-5.78,50.16],[-4.31,51.21],[-3.41,51.43],[-4.98,51.59],[-5.27,51.99],[-4.22,52.3],[-4.77,52.84],[-4.58,53.5],[-3.09,53.4]]]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 12742.0 * math.asin(min(1.0, math.sqrt(a)))


def fit_clock(anchors):
    if len(anchors) < 2:
        raise ValueError(f"Only {len(anchors)} clock anchors")
    mono0, tick0 = anchors[0]
    x = [(mono - mono0) / 1e9 for mono, _ in anchors]
    y = [tick - tick0 for _, tick in anchors]
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    denom = sum((value - xm) ** 2 for value in x)
    if denom == 0:
        raise ValueError("Clock anchors have no monotonic span")
    rate = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denom
    if rate <= 0:
        raise ValueError(f"Invalid clock rate {rate}")
    intercept = ym - rate * xm
    return mono0, tick0, rate, intercept


def archived_segments():
    result = []
    for path in sorted(ARCHIVE.glob("capture-*.sqlite")):
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            metadata_json, status, boundary = db.execute(
                "SELECT metadata_json,status,boundary_complete FROM run"
            ).fetchone()
            start = json.loads(metadata_json)
            anchors = db.execute(
                "SELECT monotonic_ns,mlat_ticks FROM clock_anchors ORDER BY monotonic_ns"
            ).fetchall()
            discontinuities, = db.execute(
                "SELECT discontinuities FROM clock_fit"
            ).fetchone()
            if discontinuities:
                raise ValueError(f"Clock discontinuity in {path.name}")
            clock = fit_clock(anchors)
            result.append({
                "path": path,
                "start_ns": start["realtime_ns"],
                "status": status,
                "boundary": boundary,
                "clock": clock,
            })
        finally:
            db.close()
    return sorted(result, key=lambda row: row["start_ns"])


class LimitedReader:
    def __init__(self, stream, limit):
        self.stream = stream
        self.remaining = limit

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data


def read_active_segment():
    metadata_paths = sorted(ARCHIVE.glob("capture-*.jsonl"))
    raw_paths = sorted(ARCHIVE.glob("capture-*.raw"))
    active = []
    raw_by_stem = {path.stem: path for path in raw_paths}
    for meta_path in metadata_paths:
        if meta_path.stem not in raw_by_stem:
            continue
        start = None
        chunks, snapshots = [], []
        with meta_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["type"] == "start":
                    start = row
                elif row["type"] == "chunk":
                    chunks.append(row)
                elif row["type"] == "snapshot":
                    snapshots.append(row)
                elif row["type"] == "end":
                    start = None
                    break
        if start is not None and chunks:
            active.append((meta_path, raw_by_stem[meta_path.stem], start, chunks, snapshots))
    if not active:
        return {}, None
    if len(active) != 1:
        raise ValueError(f"Expected one active capture, found {len(active)}")
    meta_path, raw_path, start, chunks, snapshots = active[0]
    ends = []
    for row in chunks:
        if row["offset"] != (ends[-1] if ends else 0) or row["size"] <= 0:
            raise ValueError(f"Noncontiguous active chunk index: {meta_path}")
        ends.append(row["offset"] + row["size"])
    cutoff = ends[-1]
    if raw_path.stat().st_size < cutoff:
        raise ValueError("Active raw file is shorter than its published chunk index")
    snap_times = [row["monotonic_ns"] for row in snapshots]
    position_maps = [
        {str(a.get("hex", "")).lstrip("~").upper(): a for a in row["aircraft"]}
        for row in snapshots
    ]
    anchors = []
    last_anchor = None
    found = collections.defaultdict(list)
    with raw_path.open("rb") as stream:
        limited = LimitedReader(stream, cutoff)
        for _, frame_end, kind, body in frames(limited):
            chunk_index = bisect.bisect_left(ends, frame_end)
            if chunk_index >= len(chunks):
                raise ValueError("Frame extends beyond active chunk index")
            chunk = chunks[chunk_index]
            mono, recv = chunk["monotonic_ns"], chunk["realtime_ns"]
            ticks = int.from_bytes(body[:6], "big")
            if kind in (0x32, 0x33) and ticks:
                if (mono - start["monotonic_ns"] >= 10_000_000_000
                        and (last_anchor is None or mono - last_anchor >= 60_000_000_000)):
                    anchors.append((mono, ticks))
                    last_anchor = mono
            if kind != 0x32:
                continue
            payload = body[7:]
            code = interrogator(payload)
            if code is None or code[1] == 0:
                continue
            address = payload[1:4].hex().upper()
            index = bisect.bisect_right(snap_times, mono) - 1
            if index < 0:
                continue
            aircraft = position_maps[index].get(address)
            if not aircraft or not isinstance(aircraft.get("seen_pos"), (int, float)):
                continue
            age = round((recv / 1e9 - snapshots[index]["snapshot_utc"] + aircraft["seen_pos"]) * 1000)
            lat, lon = aircraft.get("lat"), aircraft.get("lon")
            if age > RAW_LIMIT or lat is None or lon is None:
                continue
            found[code].append((address, recv, mono, ticks, float(lat), float(lon)))
    if not anchors:
        raise ValueError(f"No clock anchors in active capture: {raw_path}")
    clock = fit_clock(anchors)
    return found, {
        "path": raw_path.name,
        "cutoff_bytes": cutoff,
        "cutoff_utc": dt.datetime.fromtimestamp(
            chunks[-1]["realtime_ns"] / 1e9, dt.timezone.utc
        ).isoformat(),
        "status": "ACTIVE_PARTIAL",
        "events": sum(map(len, found.values())),
        "clock": clock,
        "start_ns": start["realtime_ns"],
    }


def compute_interval_hist(points):
    hist = np.zeros(135, dtype=np.int64)
    if len(points) < 2:
        return hist
    order = np.lexsort((points[:, 0], points[:, 4]))
    sorted_pts = points[order]
    same_icao = sorted_pts[1:, 4] == sorted_pts[:-1, 4]
    deltas = sorted_pts[1:, 0] - sorted_pts[:-1, 0]
    valid = same_icao & (deltas >= 1.5) & (deltas <= 15.0)
    bins = np.minimum(134, ((deltas[valid] - 1.5) * 10.0).astype(np.int64))
    np.add.at(hist, bins, 1)
    return hist


def candidate_periods(hist):
    smooth = np.convolve(np.asarray(hist, dtype=np.float64),
                         np.array([0.25, 0.5, 0.25]), mode="same")
    peaks = [
        i for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0
    ]
    if not peaks:
        return []
    peaks.sort(key=lambda i: smooth[i], reverse=True)
    threshold = max(3.0, float(smooth[peaks[0]]) * 0.05)
    chosen = []
    for index in peaks:
        if smooth[index] < threshold:
            break
        if all(abs(index - other) >= 4 for other in chosen):
            chosen.append(index)
        if len(chosen) == 5:
            break
    return [1.5 + (index + 0.5) * 0.1 for index in chosen]


def period_grid(candidates):
    values = set()
    for center in candidates:
        values.update(round(center - 0.25 + offset * 0.02, 6) for offset in range(26))
    return np.array(sorted(values), dtype=np.float64)


def grid_search_weighted(points, periods, fine_center=None, bounds=None):
    lat_min, lat_max, lon_min, lon_max = bounds or (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    dt_s = np.asarray(points[:, 0] - T0_EPOCH, dtype=np.float64)
    target_lat = np.deg2rad(np.asarray(points[:, 1], dtype=np.float32))
    target_lon = np.deg2rad(np.asarray(points[:, 2], dtype=np.float32))
    weights = np.minimum(points[:, 3], MAX_BURST_WEIGHT).astype(np.float32)
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        return (-1.0, None, None, None)

    cos_target = np.cos(target_lat)
    sin_target = np.sin(target_lat)
    cos_target_lon = np.cos(target_lon)
    sin_target_lon = np.sin(target_lon)

    if fine_center is None:
        site_lats = np.linspace(lat_min, lat_max, 25, dtype=np.float32)
        site_lons = np.linspace(lon_min, lon_max, 33, dtype=np.float32)
        mesh_lon, mesh_lat = np.meshgrid(site_lons, site_lats)
        candidate_lat = mesh_lat.ravel()
        candidate_lon = mesh_lon.ravel()
    else:
        lat0, lon0, period0 = fine_center
        fine_lats = np.unique(np.round(np.clip(
            lat0 + np.arange(-15, 16) * 0.02, lat_min, lat_max), 8)).astype(np.float32)
        fine_lons = np.unique(np.round(np.clip(
            lon0 + np.arange(-15, 16) * 0.02, lon_min, lon_max), 8)).astype(np.float32)
        mesh_lon, mesh_lat = np.meshgrid(fine_lons, fine_lats)
        candidate_lat = mesh_lat.ravel()
        candidate_lon = mesh_lon.ravel()
        periods = period0 + np.arange(-15, 16) * 0.002

    periods = np.asarray(periods, dtype=np.float64)
    weighted_phases = np.empty((len(periods), len(dt_s)), dtype=np.complex64)
    for idx, period in enumerate(periods):
        phase = (np.remainder(dt_s, period) * (PI2 / period)).astype(np.float32)
        weighted_phases[idx] = weights * (np.cos(phase) - 1j * np.sin(phase))

    cos_sites = np.cos(np.deg2rad(candidate_lat))
    sin_sites = np.sin(np.deg2rad(candidate_lat))
    site_lon_rad = np.deg2rad(candidate_lon)
    cos_site_lon = np.cos(site_lon_rad)
    sin_site_lon = np.sin(site_lon_rad)

    best = (-1.0, None, None, None)
    lat_chunks = 64
    for start in range(0, len(candidate_lat), lat_chunks):
        stop = min(start + lat_chunks, len(candidate_lat))
        sin_delta_lon = (
            sin_target_lon[None, :] * cos_site_lon[start:stop, None]
            - cos_target_lon[None, :] * sin_site_lon[start:stop, None]
        )
        cos_delta_lon = (
            cos_target_lon[None, :] * cos_site_lon[start:stop, None]
            + sin_target_lon[None, :] * sin_site_lon[start:stop, None]
        )
        y = sin_delta_lon * cos_target[None, :]
        x = (cos_sites[start:stop, None] * sin_target[None, :]
             - sin_sites[start:stop, None] * cos_target[None, :] * cos_delta_lon)
        inv_len = 1.0 / np.maximum(np.hypot(x, y), 1e-12)
        unit_bearing = (x * inv_len) + 1j * (y * inv_len)
        values = unit_bearing @ weighted_phases.T
        magnitudes = np.abs(values)
        site_index, period_index = np.unravel_index(
            int(np.argmax(magnitudes)), magnitudes.shape
        )
        rbar = float(magnitudes[site_index, period_index] / w_sum)
        if rbar > best[0]:
            row = start + site_index
            best = (rbar, float(candidate_lat[row]),
                    float(candidate_lon[row]), float(periods[period_index]))
    return best


def refine_continuous(points, lat0, lon0, period0, bounds=None):
    """
    4-stufige Sub-Grid-Nachoptimierung in float64 (+-0.08° Reichweite bis auf
    0,0002° / ~20 m und 0,00001 s / 10 µs Periodenpräzision).
    """
    lat_min, lat_max, lon_min, lon_max = bounds or (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    dt_s = points[:, 0] - T0_EPOCH
    t_lat = np.deg2rad(points[:, 1])
    t_lon = np.deg2rad(points[:, 2])
    weights = np.minimum(points[:, 3], MAX_BURST_WEIGHT)
    w_sum = float(np.sum(weights))
    cos_tlat = np.cos(t_lat)
    sin_tlat = np.sin(t_lat)

    lat_c, lon_c, per_c = float(lat0), float(lon0), float(period0)
    for d_deg, d_per in ((0.015, 0.0005), (0.004, 0.00015), (0.001, 0.00004), (0.0002, 0.00001)):
        lats = np.clip(lat_c + np.arange(-4, 5) * d_deg, lat_min, lat_max)
        lons = np.clip(lon_c + np.arange(-4, 5) * d_deg, lon_min, lon_max)
        pers = per_c + np.arange(-4, 5) * d_per

        phase_t = np.remainder(dt_s, per_c) * (PI2 / per_c)
        wt_vec = weights * (np.cos(phase_t) - 1j * np.sin(phase_t))
        mesh_lon, mesh_lat = np.meshgrid(lons, lats)
        clat = mesh_lat.ravel()
        clon = mesh_lon.ravel()
        lat_r = np.deg2rad(clat)[:, None]
        lon_r = np.deg2rad(clon)[:, None]
        dlon = t_lon[None, :] - lon_r
        y = np.sin(dlon) * cos_tlat[None, :]
        x = np.cos(lat_r) * sin_tlat[None, :] - np.sin(lat_r) * cos_tlat[None, :] * np.cos(dlon)
        inv_l = 1.0 / np.maximum(np.hypot(x, y), 1e-12)
        ub = (x * inv_l) + 1j * (y * inv_l)
        mags = np.abs(ub @ wt_vec)
        best_s = int(np.argmax(mags))
        lat_c, lon_c = float(clat[best_s]), float(clon[best_s])

        lat_rad = math.radians(lat_c)
        lon_rad = math.radians(lon_c)
        dlon_1 = t_lon - lon_rad
        y1 = np.sin(dlon_1) * cos_tlat
        x1 = math.cos(lat_rad) * sin_tlat - math.sin(lat_rad) * cos_tlat * np.cos(dlon_1)
        bearing = np.arctan2(y1, x1)
        ub_1 = weights * (np.cos(bearing) + 1j * np.sin(bearing))

        best_p_val = -1.0
        for p_cand in pers:
            ph = np.remainder(dt_s, p_cand) * (PI2 / p_cand)
            val = abs(np.sum(ub_1 * (np.cos(ph) - 1j * np.sin(ph)))) / w_sum
            if val > best_p_val:
                best_p_val = float(val)
                per_c = float(p_cand)

    return lat_c, lon_c, per_c


def site_geometry(points, lat_r, lon_r):
    """Peilwinkel und Grosskreisentfernung vom Kandidaten zu allen Bursts."""
    lat0 = math.radians(lat_r)
    lon0 = math.radians(lon_r)
    cos_lat0 = math.cos(lat0)
    sin_lat0 = math.sin(lat0)

    t_lat = np.deg2rad(points[:, 1])
    t_lon = np.deg2rad(points[:, 2])

    dlon = t_lon - lon0
    cos_tlat = np.cos(t_lat)
    sin_tlat = np.sin(t_lat)
    y = np.sin(dlon) * cos_tlat
    x = cos_lat0 * sin_tlat - sin_lat0 * cos_tlat * np.cos(dlon)
    bearing = np.arctan2(y, x)

    cos_c = np.clip(sin_lat0 * sin_tlat + cos_lat0 * cos_tlat * np.cos(dlon), -1.0, 1.0)
    dist_km = 6371.0 * np.arccos(cos_c)
    return bearing, dist_km


def mainlobe_inlier_mask(phase, dist_km, rbar_w, phi0_rad):
    sigma_deg = math.degrees(math.sqrt(max(0.01, -2.0 * math.log(max(0.15, min(0.995, rbar_w))))))
    adaptive_cut_deg = max(16.0, min(36.0, 2.2 * sigma_deg))
    delta_deg = np.abs((np.rad2deg(phase - phi0_rad) + 180.0) % 360.0 - 180.0)
    return (delta_deg <= adaptive_cut_deg) & (dist_km <= PEEL_MAX_RANGE_KM)


def split_search_validation(points):
    """Fester 50/50-Split nach Flugzeug und 5-Minuten-Block, vor jedem Fit."""
    aircraft = points[:, 4].astype(np.uint64)
    blocks = np.floor((points[:, 0] - T0_EPOCH) / VALIDATION_BLOCK_S).astype(np.int64)
    keys = (aircraft * 0x9E3779B1) ^ (blocks.astype(np.uint64) * 0x85EBCA77)
    keys = ((keys ^ (keys >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    keys = ((keys ^ (keys >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    validation = ((keys ^ (keys >> 16)) & 1) != 0
    return points[~validation], points[validation]


def compute_temporal_stability(times, phase, weights):
    """Quartals-Konsistenz auf ungefilterten Validierungsphasen, nicht auf Inliern."""
    if len(times) < 4 * MIN_QUARTILE_BURSTS:
        return 0.0
    quartile_vectors = []
    for indices in np.array_split(np.argsort(times), 4):
        vector = np.sum(weights[indices] * np.exp(1j * phase[indices]))
        r_q = abs(vector) / float(np.sum(weights[indices]))
        if r_q >= MIN_VALIDATION_RBAR:
            quartile_vectors.append(np.exp(1j * np.angle(vector)))
    if not quartile_vectors:
        return 0.0
    consistency = float(abs(np.mean(quartile_vectors)))
    return max(0.0, min(1.0, consistency * len(quartile_vectors) / 4.0))


def inlier_coverage_statistics(points, lat_r, lon_r):
    """Empirische Reichweiten und belegte Zellen der zugeordneten Inlier-Bursts."""
    if len(points) == 0:
        raise ValueError("Coverage statistics require inlier bursts")
    _, distances = site_geometry(points, lat_r, lon_r)
    p05, median, p95 = np.percentile(distances, [5, 50, 95])
    cells = np.column_stack((
        np.floor((points[:, 1] - COVERAGE_LAT_ORIGIN) / COVERAGE_D_LAT),
        np.floor((points[:, 2] - COVERAGE_LON_ORIGIN) / COVERAGE_D_LON),
    )).astype(np.int64)
    return {
        "range_p05_km": round(float(p05), 2),
        "range_median_km": round(float(median), 2),
        "range_p95_km": round(float(p95), 2),
        "n_coverage_cells": len(np.unique(cells, axis=0)),
    }


def evaluate_site_diagnostics(points, lat_r, lon_r, period_s):
    """
    Bewertet einen eingefrorenen Fit auf einem rein geometrisch bestimmten Pool.
    R_bar, Ring-/Perioden-Kontrast und Zeitstabilitaet verwenden alle Bursts.
    """
    bearing, dist_km = site_geometry(points, lat_r, lon_r)
    dt_s = points[:, 0] - T0_EPOCH
    weights = np.minimum(points[:, 3], MAX_BURST_WEIGHT)
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        raise ValueError("Site diagnostics require positive burst weights")
    time_phase = np.remainder(dt_s, period_s) * (PI2 / period_s)
    phase = bearing - time_phase
    w_re = float(np.sum(weights * np.cos(phase)))
    w_im = float(np.sum(weights * np.sin(phase)))
    rbar_w = math.hypot(w_re, w_im) / w_sum
    phi0_rad = math.atan2(w_im, w_re)
    phi0_deg = math.degrees(phi0_rad) % 360.0
    inlier_mask = mainlobe_inlier_mask(phase, dist_km, rbar_w, phi0_rad)
    temporal_stability = compute_temporal_stability(
        points[:, 0], phase, weights
    )

    inlier_bearings = bearing[inlier_mask]
    if len(inlier_bearings) >= 20:
        b_re = float(np.mean(np.cos(inlier_bearings)))
        b_im = float(np.mean(np.sin(inlier_bearings)))
        r_bearing = math.hypot(b_re, b_im)
        azimuth_spread_score = max(0.0, min(1.0, (1.0 - r_bearing) / 0.55))
        median_dist_km = float(np.median(dist_km[inlier_mask]))
    else:
        azimuth_spread_score = 0.0
        median_dist_km = float(np.median(dist_km)) if len(dist_km) else 250.0

    # Acht geodaetische Kontrollpunkte; der Kreisschlusspunkt zaehlt nicht doppelt.
    ring_rbars = []
    for lon_c, lat_c in geodesic_circle(lat_r, lon_r, SHARPNESS_RADIUS_KM, n_pts=8)[:-1]:
        bc, _ = site_geometry(points, lat_c, lon_c)
        ph_c = bc - time_phase
        rc = math.hypot(float(np.sum(weights * np.cos(ph_c))),
                        float(np.sum(weights * np.sin(ph_c)))) / w_sum
        ring_rbars.append(rc)
    directional_drop = max(0.0, rbar_w - np.percentile(ring_rbars, 80))
    expected_drop = max(0.03, rbar_w * (
        1.0 - math.cos(math.atan2(SHARPNESS_RADIUS_KM, max(60.0, median_dist_km)))
    ))
    spatial_sharpness_ratio = min(1.5, float(directional_drop) / expected_drop)

    # Perioden-Schärfe (Phasenabfall bei +-0.008 s Verstimmung)
    per_rbars = []
    for dp in (-0.008, 0.008):
        p_test = period_s + dp
        ph_p = bearing - np.remainder(dt_s, p_test) * (PI2 / p_test)
        rp = math.hypot(float(np.sum(weights * np.cos(ph_p))),
                        float(np.sum(weights * np.sin(ph_p)))) / w_sum
        per_rbars.append(rp)
    period_contrast = max(0.0, rbar_w - max(per_rbars))

    return (rbar_w, phi0_deg, inlier_mask, azimuth_spread_score,
            spatial_sharpness_ratio, period_contrast, temporal_stability)


def compute_confidence_score(rbar_w, spatial_sharpness, period_contrast, azimuth_spread,
                             mainlobe_ratio, n_inliers, lat_r, lon_r, peel_pass,
                             temporal_stability):
    """Berechnet den entfernungsnormierten Konfidenz-Score in [0.0, 100.0]."""
    d_edge = min(abs(lat_r - LAT_MIN), abs(LAT_MAX - lat_r),
                 abs(lon_r - LON_MIN), abs(LON_MAX - lon_r))
    if d_edge < 0.05 or mainlobe_ratio < MIN_MAINLOBE_RATIO:
        return 0.0
    s_edge = min(1.0, d_edge / 0.18)

    s_R = max(0.0, min(1.0, (rbar_w - 0.35) / 0.52))
    s_sharp = max(0.0, min(1.0, spatial_sharpness / 0.75))
    s_per = max(0.0, min(1.0, period_contrast / 0.25))
    s_gdop = 0.65 + 0.35 * azimuth_spread
    s_ml = max(0.0, min(1.0, mainlobe_ratio / 0.25))
    s_n = max(0.0, min(1.0, (math.log10(max(1, n_inliers)) - 2.0) / 1.8))
    s_pass = 1.0 if peel_pass == 1 else (0.94 if peel_pass == 2 else 0.84)

    raw_score = (100.0 * (s_R ** 1.1) * (s_sharp ** 0.45) * (s_per ** 0.35)
                 * (s_gdop ** 0.45) * (s_ml ** 0.45) * (s_n ** 0.35) * s_edge * s_pass
                 * (temporal_stability ** 0.5))
    return round(max(0.0, min(100.0, raw_score)), 1)


def search_candidate(points, max_grid_bursts, bounds=None):
    periods = candidate_periods(compute_interval_hist(points))
    if not periods:
        return None
    if len(points) > max_grid_bursts:
        step = len(points) / max_grid_bursts
        indices = np.floor(np.arange(max_grid_bursts) * step).astype(np.int64)
        grid_pts = points[indices]
    else:
        grid_pts = points
    coarse = grid_search_weighted(grid_pts, period_grid(periods), bounds=bounds)
    _, lat_f, lon_f, per_f = grid_search_weighted(
        grid_pts, None, fine_center=coarse[1:], bounds=bounds
    )
    return refine_continuous(grid_pts, lat_f, lon_f, per_f, bounds=bounds)


def evaluate_candidate(points, site, code_label, mainlobe_ratio, peel_pass, *,
                       training_points, validation_points, bounds=None):
    """Validiert ohne Phasenvorauswahl; Inlier dienen erst danach der Zuordnung."""
    lat_opt, lon_opt, per_opt = site
    train_bearing, train_dist = site_geometry(training_points, lat_opt, lon_opt)
    train_weights = np.minimum(training_points[:, 3], MAX_BURST_WEIGHT)
    train_phase = train_bearing - np.remainder(
        training_points[:, 0] - T0_EPOCH, per_opt
    ) * (PI2 / per_opt)
    inlier_mask = np.zeros(len(training_points), dtype=bool)
    if len(training_points) == 0:
        return None, inlier_mask, 0.0
    train_vector = np.sum(train_weights * np.exp(1j * train_phase))
    train_rbar = float(abs(train_vector) / np.sum(train_weights))
    train_phi = float(np.angle(train_vector))
    inlier_mask = mainlobe_inlier_mask(train_phase, train_dist, train_rbar, train_phi)

    _, validation_dist = site_geometry(validation_points, lat_opt, lon_opt)
    validation_mask = validation_dist <= PEEL_MAX_RANGE_KM
    if bounds is not None:
        validation_mask &= (
            (validation_points[:, 1] >= bounds[0]) & (validation_points[:, 1] <= bounds[1])
            & (validation_points[:, 2] >= bounds[2]) & (validation_points[:, 2] <= bounds[3])
        )
    validation_pool = validation_points[validation_mask]
    if len(validation_pool) < MIN_VALIDATION_BURSTS:
        return None, inlier_mask, train_rbar

    rbar_w, phi0_deg, _, az_spread, sp_sharp, per_contrast, stability = (
        evaluate_site_diagnostics(validation_pool, lat_opt, lon_opt, per_opt)
    )
    phase_delta = abs((phi0_deg - math.degrees(train_phi) + 180.0) % 360.0 - 180.0)
    if (rbar_w < MIN_VALIDATION_RBAR or stability < MIN_TEMPORAL_STABILITY
            or phase_delta > MAX_VALIDATION_PHASE_DELTA_DEG):
        return None, inlier_mask, train_rbar

    validation_bearing, validation_dist = site_geometry(validation_pool, lat_opt, lon_opt)
    validation_phase = validation_bearing - np.remainder(
        validation_pool[:, 0] - T0_EPOCH, per_opt
    ) * (PI2 / per_opt)
    n_validation_inliers = int(np.sum(mainlobe_inlier_mask(
        validation_phase, validation_dist, train_rbar, train_phi
    )))
    bearing, distances = site_geometry(points, lat_opt, lon_opt)
    phase = bearing - np.remainder(points[:, 0] - T0_EPOCH, per_opt) * (PI2 / per_opt)
    all_inliers = mainlobe_inlier_mask(phase, distances, train_rbar, train_phi)
    n_inliers = int(np.sum(all_inliers))
    conf = compute_confidence_score(
        rbar_w, sp_sharp, per_contrast, az_spread, mainlobe_ratio,
        n_validation_inliers, lat_opt, lon_opt, peel_pass, stability
    )
    row = None
    if conf >= 40.0 and n_inliers >= 300:
        row = {
            "code": code_label if peel_pass == 1 else f"{code_label}#{peel_pass}",
            "base_code": code_label,
            "peel_pass": peel_pass,
            "confidence_score": conf,
            "lat_r": round(lat_opt, 4),
            "lon_r": round(lon_opt, 4),
            "period_s": round(per_opt, 5),
            "rpm": round(60.0 / per_opt, 4),
            "R_bar_max": round(rbar_w, 4),
            "R_bar_train": round(train_rbar, 4),
            "n_validation_bursts": len(validation_pool),
            "n_validation_inliers": n_validation_inliers,
            "validation_phase_delta_deg": round(phase_delta, 2),
            "spatial_sharpness": round(sp_sharp, 4),
            "period_contrast": round(per_contrast, 4),
            "azimuth_spread": round(az_spread, 4),
            "mainlobe_ratio": round(mainlobe_ratio, 4),
            "temporal_stability": round(stability, 4),
            "n_inliers": n_inliers,
            "phi0_deg": round(math.degrees(train_phi) % 360.0, 2),
            **inlier_coverage_statistics(points[all_inliers], lat_opt, lon_opt),
        }
    return row, inlier_mask, train_rbar


def search_spatial_tiles(points, code_label, mainlobe_ratio, *,
                         training_points, validation_points):
    """Kachel-Fits auf Trainingsdaten, Pruefung auf ungefilterten Kachel-Holdouts."""
    candidates = []
    for i, lat_min in enumerate(np.linspace(LAT_MIN, LAT_MAX - TILE_LAT_SPAN, 3)):
        for j, lon_min in enumerate(np.linspace(LON_MIN, LON_MAX - TILE_LON_SPAN, 3)):
            bounds = (lat_min, lat_min + TILE_LAT_SPAN, lon_min, lon_min + TILE_LON_SPAN)
            mask = ((points[:, 1] >= bounds[0]) & (points[:, 1] <= bounds[1])
                    & (points[:, 2] >= bounds[2]) & (points[:, 2] <= bounds[3]))
            if np.sum(mask) < TILE_MIN_BURSTS:
                continue
            train_mask = (
                (training_points[:, 1] >= bounds[0]) & (training_points[:, 1] <= bounds[1])
                & (training_points[:, 2] >= bounds[2]) & (training_points[:, 2] <= bounds[3])
            )
            local_training = training_points[train_mask]
            if len(local_training) < 250:
                continue
            site = search_candidate(local_training, TILE_MAX_GRID_BURSTS, bounds=bounds)
            if site is None:
                continue
            row, _, _ = evaluate_candidate(
                points, site, code_label, mainlobe_ratio, 1,
                training_points=local_training, validation_points=validation_points, bounds=bounds,
            )
            if row is not None:
                row["code"] = f"{code_label}@T{i + 1}{j + 1}"
                candidates.append(row)
    return candidates


def process_one_code(args):
    code, segments, active_list, active_clock = args
    code_label = f"{code[0]}{code[1]}"
    raw_frames = 0

    buf_t = array("d")
    buf_lat = array("d")
    buf_lon = array("d")
    buf_cnt = array("d")
    buf_icao = array("d")

    background = []
    sample_state = [0, random.Random(20261116 + code[1] * 31 + (0 if code[0] == "II" else 1000))]
    states = {}

    def flush_state(icao_int, st):
        inv_n = 1.0 / st[4]
        t_m = st[0] * inv_n
        lat_m = st[1] * inv_n
        lon_m = st[2] * inv_n
        buf_t.append(t_m)
        buf_lat.append(lat_m)
        buf_lon.append(lon_m)
        buf_cnt.append(float(st[4]))
        buf_icao.append(float(icao_int))

        seen = sample_state[0]
        sample_state[0] += 1
        if seen < 160:
            background.append((lat_m, lon_m))
        else:
            idx = sample_state[1].randrange(sample_state[0])
            if idx < 160:
                background[idx] = (lat_m, lon_m)

    for segment in segments:
        db = sqlite3.connect(f"file:{segment['path']}?mode=ro", uri=True)
        try:
            db.execute("PRAGMA mmap_size=268435456;")
            db.execute("PRAGMA temp_store=MEMORY;")
            mono0, tick0, rate, intercept = segment["clock"]
            inv_rate_ns = 1e9 / rate
            query = (
                "SELECT icao24,recv_ns,monotonic_ns,mlat_ticks,lat,lon "
                "FROM df11_events INDEXED BY df11_code_time "
                "WHERE code_type=? AND code_val=? AND pos_age_ms<=? "
                "AND lat IS NOT NULL AND lon IS NOT NULL "
                "ORDER BY monotonic_ns,mlat_ticks"
            )
            for icao, recv, mono, ticks, lat, lon in db.execute(
                query, (code[0], code[1], RAW_LIMIT)
            ):
                raw_frames += 1
                t = (mono0 + (ticks - tick0 - intercept) * inv_rate_ns + recv - mono) * 1e-9
                icao_int = int(icao, 16)
                st = states.get(icao_int)
                if st is None:
                    states[icao_int] = (t, float(lat), float(lon), t, 1)
                else:
                    delta = t - st[3]
                    if 0.0 <= delta <= BURST_GAP_S:
                        states[icao_int] = (
                            st[0] + t, st[1] + float(lat), st[2] + float(lon), t, st[4] + 1
                        )
                    else:
                        flush_state(icao_int, st)
                        states[icao_int] = (t, float(lat), float(lon), t, 1)
        finally:
            db.close()

    if active_list and active_clock is not None:
        mono0, tick0, rate, intercept = active_clock
        inv_rate_ns = 1e9 / rate
        for icao, recv, mono, ticks, lat, lon in active_list:
            raw_frames += 1
            t = (mono0 + (ticks - tick0 - intercept) * inv_rate_ns + recv - mono) * 1e-9
            icao_int = int(icao, 16)
            st = states.get(icao_int)
            if st is None:
                states[icao_int] = (t, float(lat), float(lon), t, 1)
            else:
                delta = t - st[3]
                if 0.0 <= delta <= BURST_GAP_S:
                    states[icao_int] = (
                        st[0] + t, st[1] + float(lat), st[2] + float(lon), t, st[4] + 1
                    )
                else:
                    flush_state(icao_int, st)
                    states[icao_int] = (t, float(lat), float(lon), t, 1)

    for icao_int, st in states.items():
        flush_state(icao_int, st)
    states.clear()

    if raw_frames < 150 or len(buf_t) == 0:
        return None

    n_total_bursts = len(buf_t)
    points_all = np.column_stack((
        np.frombuffer(buf_t, dtype=np.float64),
        np.frombuffer(buf_lat, dtype=np.float64),
        np.frombuffer(buf_lon, dtype=np.float64),
        np.frombuffer(buf_cnt, dtype=np.float64),
        np.frombuffer(buf_icao, dtype=np.float64),
    ))
    del buf_t, buf_lat, buf_lon, buf_cnt, buf_icao

    mainlobe_ratio = float(np.mean(points_all[:, 3] >= MIN_MAINLOBE_FRAMES))
    if mainlobe_ratio < MIN_MAINLOBE_RATIO:
        return [], raw_frames, n_total_bursts, background

    ml_mask = points_all[:, 3] >= MIN_MAINLOBE_FRAMES
    support_pool = points_all[ml_mask] if np.sum(ml_mask) >= 400 else points_all
    del points_all
    working_pool, validation_pool = split_search_validation(support_pool)

    detected_rows = []
    for peel_pass in range(1, MAX_PEEL_PASSES + 1):
        n_pool = len(working_pool)
        if n_pool < 250:
            break

        site = search_candidate(working_pool, MAX_GRID_BURSTS)
        if site is None:
            break

        row, inlier_mask, rbar_w = evaluate_candidate(
            support_pool, site, code_label, mainlobe_ratio, peel_pass,
            training_points=working_pool, validation_points=validation_pool,
        )
        n_inliers = int(np.sum(inlier_mask))
        if row is not None:
            detected_rows.append(row)
        if (peel_pass == 1 and rbar_w < TILE_TRIGGER_RBAR
                and len(support_pool) >= TILE_TRIGGER_BURSTS):
            detected_rows.extend(search_spatial_tiles(
                support_pool, code_label, mainlobe_ratio,
                training_points=working_pool, validation_points=validation_pool,
            ))

        if n_inliers < 250 or (n_pool - n_inliers) < 400:
            break
        working_pool = working_pool[~inlier_mask]

    return detected_rows, raw_frames, n_total_bursts, background


def deduplicate_and_filter(candidates):
    """
    Entfernt verbleibende Self-Splits, Sub-Harmonische und Cross-Code 1-Bit-Schatten
    und filtert strikt auf confidence_score >= MIN_CONFIDENCE_SCORE.
    """
    candidates.sort(key=lambda r: r["confidence_score"], reverse=True)
    accepted = []
    for cand in candidates:
        is_artifact = False
        for prev in accepted:
            dist = haversine_km(cand["lat_r"], cand["lon_r"], prev["lat_r"], prev["lon_r"])
            dt = abs(cand["period_s"] - prev["period_s"])
            dphi = abs((cand["phi0_deg"] - prev["phi0_deg"] + 180.0) % 360.0 - 180.0)

            if cand["base_code"] == prev["base_code"]:
                if dt < 0.02 and dist < 95.0:
                    is_artifact = True
                    break
                ratio = prev["period_s"] / cand["period_s"] if cand["period_s"] > 0 else 0.0
                if dist < 80.0 and abs(ratio - round(ratio)) < 0.01 and round(ratio) >= 2:
                    is_artifact = True
                    break
            else:
                if (dist < 60.0 and dt < 0.01 and dphi < 15.0
                        and prev["n_inliers"] > 5 * cand["n_inliers"]):
                    is_artifact = True
                    break

        if not is_artifact and cand["confidence_score"] >= MIN_CONFIDENCE_SCORE:
            accepted.append(cand)
    return accepted


def geodesic_circle(lat_c, lon_c, radius_km, n_pts=120):
    ang_dist = radius_km / 6371.0
    lat1 = math.radians(lat_c)
    lon1 = math.radians(lon_c)
    pts = []
    for k in range(n_pts + 1):
        brng = PI2 * k / n_pts
        lat2 = math.asin(
            math.sin(lat1) * math.cos(ang_dist)
            + math.cos(lat1) * math.sin(ang_dist) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(ang_dist) * math.cos(lat1),
            math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
        )
        pts.append((math.degrees(lon2), math.degrees(lat2)))
    return pts


def write_svg(path, background, radars):
    width, height = 1380, 860
    left, top, right, bottom = 74, 68, 956, 796
    lon0, lon1 = 1.2, 11.4
    lat0, lat1 = 48.0, 54.8
    mid_lat = 0.5 * (lat0 + lat1)
    cos_mid = math.cos(math.radians(mid_lat))
    xspan = (lon1 - lon0) * cos_mid
    yspan = lat1 - lat0
    scale = min((right - left) / xspan, (bottom - top) / yspan)
    plot_width, plot_height = xspan * scale, yspan * scale
    right = left + plot_width
    bottom = top + plot_height

    def project(lon, lat):
        x = left + (lon - lon0) * cos_mid * scale
        y = top + (lat1 - lat) * scale
        return x, y

    def esc(text):
        return (str(text).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    font = "Noto Sans, DejaVu Sans, Arial, sans-serif"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{font}">',
        f'<defs><clipPath id="map-clip"><rect x="{left:.1f}" y="{top:.1f}" '
        f'width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="18" font-weight="bold" fill="#0f172a">'
        f'Verifizierte Mode-S Radarstationen (v5, Confidence ≥ {MIN_CONFIDENCE_SCORE:.0f} / 100)</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#475569">'
        f'{len(radars)} bereinigte Stationen · RX: ({RX_LAT:.2f}°, {RX_LON:.2f}°) · '
        '100–400 km Ringe</text>',
        '<g clip-path="url(#map-clip)">',
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" '
        f'height="{plot_height:.1f}" fill="#d8e6f2"/>',
    ]

    for ring in EUROPE_POLYS:
        pts_str = " ".join(f"{project(lo, la)[0]:.1f},{project(lo, la)[1]:.1f}" for lo, la in ring)
        lines.append(
            f'<polygon points="{pts_str}" fill="#f6f4ee" stroke="#7c8b99" '
            'stroke-width="1.05" stroke-linejoin="round"/>'
        )

    for lon in range(2, 12):
        x, _ = project(lon, mid_lat)
        lines.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                     'stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3" stroke-opacity="0.55"/>')
    for lat in range(49, 55):
        _, y = project(6.0, lat)
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                     'stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3" stroke-opacity="0.55"/>')

    rx1, ry1 = project(LON_MIN, LAT_MAX)
    rx2, ry2 = project(LON_MAX, LAT_MIN)
    lines.append(f'<rect x="{rx1:.1f}" y="{ry1:.1f}" width="{rx2-rx1:.1f}" '
                 f'height="{ry2-ry1:.1f}" fill="none" stroke="#475569" '
                 'stroke-dasharray="6 4" stroke-width="1.3"/>')

    for r_km in (100, 200, 300, 400):
        c_pts = geodesic_circle(RX_LAT, RX_LON, r_km, n_pts=120)
        p_str = " ".join(f"{project(lo, la)[0]:.1f},{project(lo, la)[1]:.1f}" for lo, la in c_pts)
        lines.append(f'<polyline points="{p_str}" fill="none" stroke="#64748b" '
                     'stroke-width="0.9" stroke-dasharray="4 4" stroke-opacity="0.7"/>')
        lx, ly = project(*c_pts[105])
        lines.append(f'<text x="{lx-4:.1f}" y="{ly-3:.1f}" font-size="9.5" '
                     f'font-weight="bold" fill="#475569" text-anchor="end">{r_km} km</text>')

    lines.append('<g fill="#2b5c8f" fill-opacity="0.20">')
    for lat, lon in background[:BACKGROUND_LIMIT]:
        x, y = project(lon, lat)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3"/>')
    lines.append('</g>')

    ref_cities = [
        ("Amsterdam", 4.90, 52.37, "end", -6, 3),
        ("Brüssel", 4.35, 50.85, "end", -6, 3),
        ("Köln", 6.96, 50.94, "start", 6, -4),
        ("Frankfurt", 8.68, 50.11, "start", 6, 3),
        ("Hannover", 9.73, 52.37, "start", 6, 3),
        ("Hamburg", 9.99, 53.55, "start", 6, 3),
        ("Stuttgart", 9.18, 48.78, "end", -6, 4),
    ]
    for cname, clo, cla, anchor, dx, dy in ref_cities:
        cx, cy = project(clo, cla)
        lines.append(f'<rect x="{cx-2:.1f}" y="{cy-2:.1f}" width="4" height="4" '
                     'fill="#475569" stroke="#fff" stroke-width="0.6"/>')
        lines.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" font-size="10.5" '
                     f'fill="#334155" text-anchor="{anchor}">{esc(cname)}</text>')

    rxx, rxy = project(RX_LON, RX_LAT)
    lines.append(f'<circle cx="{rxx:.1f}" cy="{rxy:.1f}" r="6" fill="#facc15" '
                 'stroke="#0f172a" stroke-width="1.8"/>')
    lines.append(f'<text x="{rxx+8:.1f}" y="{rxy-6:.1f}" font-size="10.5" '
                 'font-weight="bold" fill="#0f172a">RX</text>')

    colors = ["#b91c1c", "#1d4ed8", "#047857", "#d97706", "#6d28d9",
              "#0e7490", "#be185d", "#15803d", "#c2410c", "#4338ca"]
    placed_labels = []
    for i, row in enumerate(radars):
        x, y = project(row["lon_r"], row["lat_r"])
        color = colors[i % len(colors)]
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="{color}" fill-opacity="0.18"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" '
                     'stroke="#ffffff" stroke-width="1.6"/>')
        lx, ly = x + 8.0, y - 5.0
        for px_old, py_old in placed_labels:
            if abs(lx - px_old) < 125.0 and abs(ly - py_old) < 14.0:
                ly = py_old + 14.0
        placed_labels.append((lx, ly))
        label = f'{row["code"]} [{row["confidence_score"]:.0f}%] ({row["period_s"]:.2f}s)'
        lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10.5" '
                     f'font-weight="bold" fill="{color}">{esc(label)}</text>')

    lines.append('</g>')
    lines.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" '
                 f'height="{plot_height:.1f}" fill="none" stroke="#64748b" stroke-width="1.2"/>')

    for lon in range(2, 12):
        x, _ = project(lon, mid_lat)
        lines.append(f'<text x="{x:.1f}" y="{bottom+18:.1f}" font-size="11" '
                     f'fill="#475569" text-anchor="middle">{lon}°E</text>')
    for lat in range(49, 55):
        _, y = project(6.0, lat)
        lines.append(f'<text x="{left-9:.1f}" y="{y+4:.1f}" font-size="11" '
                     f'fill="#475569" text-anchor="end">{lat}°N</text>')

    px = right + 20
    pw = width - px - 18
    lines.append(f'<rect x="{px:.1f}" y="{top:.1f}" width="{pw:.1f}" '
                 f'height="{plot_height:.1f}" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>')
    lines.append(f'<text x="{px+14:.1f}" y="{top+24:.1f}" font-size="13" '
                 f'font-weight="bold" fill="#0f172a">Verifizierte Radare ({len(radars)})</text>')
    ly = top + 48
    for i, row in enumerate(radars):
        if ly > bottom - 20:
            break
        color = colors[i % len(colors)]
        lines.append(f'<circle cx="{px+18:.1f}" cy="{ly-4:.1f}" r="4.5" fill="{color}"/>')
        hdr = (f'{row["code"]}  —  Conf: {row["confidence_score"]:.1f}%  '
               f'({row["period_s"]:.3f}s / {row["rpm"]:.1f} RPM)')
        sub = (f'{row["lat_r"]:.4f}° N, {row["lon_r"]:.4f}° E · '
               f'R̄={row["R_bar_max"]:.3f} · φ0={row["phi0_deg"]:.1f}° · n={row["n_inliers"]:,}')
        lines.append(f'<text x="{px+28:.1f}" y="{ly:.1f}" font-size="10.5" '
                     f'font-weight="bold" fill="#0f172a">{esc(hdr)}</text>')
        lines.append(f'<text x="{px+28:.1f}" y="{ly+13:.1f}" font-size="9.5" '
                     f'fill="#475569">{esc(sub)}</text>')
        ly += 33

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t_start = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    segments = archived_segments()
    active_rows, active_meta = read_active_segment()
    active_clock = active_meta["clock"] if active_meta else None
    codes = ([("II", value) for value in range(1, 16)]
             + [("SI", value) for value in range(1, 64)])

    tasks = [(code, segments, active_rows.get(code, []), active_clock) for code in codes]
    raw_candidates = []
    all_bg = []

    print(f"START workers={MAX_WORKERS} segments={len(segments)} min_conf={MIN_CONFIDENCE_SCORE}", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_one_code, task) for task in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            detected_rows, _, _, bg = res
            raw_candidates.extend(detected_rows)
            all_bg.extend(bg)

    verified_radars = deduplicate_and_filter(raw_candidates)

    rng = random.Random(20261116)
    if len(all_bg) > BACKGROUND_LIMIT:
        all_bg = rng.sample(all_bg, BACKGROUND_LIMIT)

    csv_path = OUTPUT / "radar-triangulation-v5-verified.csv"
    columns = [
        "code", "confidence_score", "lat_r", "lon_r", "period_s", "rpm",
        "R_bar_max", "spatial_sharpness", "period_contrast", "azimuth_spread", "mainlobe_ratio",
        "temporal_stability", "n_inliers", "phi0_deg", "peel_pass",
        "range_p05_km", "range_median_km", "range_p95_km", "n_coverage_cells",
        "R_bar_train", "n_validation_bursts", "n_validation_inliers",
        "validation_phase_delta_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in verified_radars:
            writer.writerow(row)

    svg_path = OUTPUT / "radar-triangulation-v5-verified.svg"
    write_svg(svg_path, all_bg, verified_radars)

    elapsed_s = round(time.monotonic() - t_start, 2)
    print("\nSUMMARY", json.dumps({
        "elapsed_s": elapsed_s,
        "verified_radars": len(verified_radars),
        "min_confidence": MIN_CONFIDENCE_SCORE,
        "csv": str(csv_path),
        "svg": str(svg_path),
    }, separators=(",", ":")), flush=True)
    print("\nVERIFIED RADARS", flush=True)
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(verified_radars)


if __name__ == "__main__":
    args = parse_common_options(ARCHIVE, OUTPUT, RX_LAT, RX_LON)
    ARCHIVE, OUTPUT, RX_LAT, RX_LON = args.archive, args.output, args.rx_lat, args.rx_lon
    OUTPUT = create_run_directory("v5", __file__, OUTPUT)
    main()