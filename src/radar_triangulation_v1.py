#!/usr/bin/env python3
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
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append("/opt/adsb")
from beast_inspect import frames, interrogator
from experiment_runs import create_run_directory, parse_common_options


ARCHIVE = Path(os.environ.get("ADSB_ARCHIVE", "/var/lib/adsb/beast-experiment"))
OUTPUT = Path(os.environ.get("ADSB_OUTPUT", "/tmp/radar-triangulation/v1"))
RX_LAT, RX_LON = float(os.environ.get("ADSB_RX_LAT", "51.145")), float(os.environ.get("ADSB_RX_LON", "6.355"))
RAW_LIMIT = 5000
BURST_GAP_S = 0.5
BACKGROUND_LIMIT = 12000
LAT_MIN, LAT_MAX = 48.5, 54.5
LON_MIN, LON_MAX = 2.5, 10.5
PI2 = 2.0 * math.pi


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


def event_epoch(recv_ns, mono_ns, ticks, clock):
    mono0, tick0, rate, intercept = clock
    estimated_mono = mono0 + ((ticks - tick0 - intercept) / rate) * 1e9
    return (estimated_mono + recv_ns - mono_ns) / 1e9


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
                "SELECT monotonic_ns,mlat_ticks FROM clock_anchors "
                "ORDER BY monotonic_ns"
            ).fetchall()
            discontinuities, = db.execute(
                "SELECT discontinuities FROM clock_fit"
            ).fetchone()
            if discontinuities or boundary != 1:
                # Boundary-incomplete rows remain useful; they are still parsed records.
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
            active.append((meta_path, raw_by_stem[meta_path.stem],
                           start, chunks, snapshots))
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
    position_maps = []
    for row in snapshots:
        position_maps.append({
            str(a.get("hex", "")).lstrip("~").upper(): a
            for a in row["aircraft"]
        })
    anchors = []
    first_mono = None
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
                if first_mono is None:
                    first_mono = mono
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
            age = round((recv / 1e9 - snapshots[index]["snapshot_utc"]
                         + aircraft["seen_pos"]) * 1000)
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


def candidate_periods(hist):
    smooth = np.convolve(np.asarray(hist, dtype=np.float64),
                         np.array([0.25, 0.5, 0.25]), mode="same")
    peaks = [
        i for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
        and smooth[i] > 0
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
        values.update(round(center - 0.25 + offset * 0.02, 6)
                      for offset in range(26))
    return np.array(sorted(values), dtype=np.float64)


def rayleigh_p(n, rbar):
    z = n * rbar * rbar
    p = math.exp(-z) if z < 745 else 0.0
    p *= 1.0 + (2.0 * z - z * z) / (4.0 * n)
    p -= math.exp(-z) * (
        24.0 * z - 132.0 * z * z + 76.0 * z**3 - 9.0 * z**4
    ) / (288.0 * n * n) if z < 745 else 0.0
    return max(0.0, min(1.0, p))


def grid_search(points, periods, fine_center=None):
    times = np.asarray(points[:, 0], dtype=np.float64)
    target_lat = np.deg2rad(np.asarray(points[:, 1], dtype=np.float64))
    target_lon = np.deg2rad(np.asarray(points[:, 2], dtype=np.float64))
    cos_target = np.cos(target_lat)
    sin_target = np.sin(target_lat)
    cos_target_lon = np.cos(target_lon)
    sin_target_lon = np.sin(target_lon)
    if fine_center is None:
        site_lats = np.linspace(LAT_MIN, LAT_MAX, 25)
        site_lons = np.linspace(LON_MIN, LON_MAX, 33)
        mesh_lon, mesh_lat = np.meshgrid(site_lons, site_lats)
        candidate_lat = mesh_lat.ravel()
        candidate_lon = mesh_lon.ravel()
    else:
        lat0, lon0, period0 = fine_center
        fine_lats = np.unique(np.round(np.clip(
            lat0 + np.arange(-15, 16) * 0.02, LAT_MIN, LAT_MAX), 8))
        fine_lons = np.unique(np.round(np.clip(
            lon0 + np.arange(-15, 16) * 0.02, LON_MIN, LON_MAX), 8))
        mesh_lon, mesh_lat = np.meshgrid(fine_lons, fine_lats)
        candidate_lat = mesh_lat.ravel()
        candidate_lon = mesh_lon.ravel()
        periods = period0 + np.arange(-15, 16) * 0.002
    best = (-1.0, None, None, None)
    cos_sites = np.cos(np.deg2rad(candidate_lat))
    sin_sites = np.sin(np.deg2rad(candidate_lat))
    site_lon_rad = np.deg2rad(candidate_lon)
    cos_site_lon = np.cos(site_lon_rad)
    sin_site_lon = np.sin(site_lon_rad)
    periods = np.asarray(periods, dtype=np.float64)
    weighted_phases = np.empty((len(periods), len(times)), dtype=np.complex128)
    for index, period in enumerate(periods):
        phase = np.remainder(times, period) * (PI2 / period)
        weighted_phases[index] = np.cos(phase) - 1j * np.sin(phase)
    lat_chunks = 32
    for start in range(0, len(candidate_lat), lat_chunks):
        stop = min(start + lat_chunks, len(candidate_lat))
        # Candidate sites are row-major; each chunk may cross a latitude row.
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
             - sin_sites[start:stop, None] * cos_target[None, :]
             * cos_delta_lon)
        length = np.hypot(x, y)
        np.divide(x, length, out=x, where=length > 0)
        np.divide(y, length, out=y, where=length > 0)
        unit_bearing = x + 1j * y
        values = unit_bearing @ weighted_phases.T
        magnitudes = np.abs(values)
        site_index, period_index = np.unravel_index(
            int(np.argmax(magnitudes)), magnitudes.shape
        )
        rbar = float(magnitudes[site_index, period_index] / len(times))
        if rbar > best[0]:
            row = start + site_index
            best = (rbar, float(candidate_lat[row]),
                    float(candidate_lon[row]), float(periods[period_index]))
    return best


def add_burst(output, hist, previous, icao, times, lats, lons,
              background, sample_state, extrema):
    t = float(statistics.median(times))
    lat = float(statistics.median(lats))
    lon = float(statistics.median(lons))
    output.extend((t, lat, lon))
    if icao in previous:
        delta = t - previous[icao]
        if 1.5 <= delta <= 15.0:
            bin_index = min(134, int((delta - 1.5) / 0.1))
            hist[bin_index] += 1
    previous[icao] = t
    seen = sample_state[0]
    sample_state[0] += 1
    if seen < BACKGROUND_LIMIT:
        background.append((lat, lon))
    else:
        index = sample_state[1].randrange(sample_state[0])
        if index < BACKGROUND_LIMIT:
            background[index] = (lat, lon)
    extrema[0] = min(extrema[0], lat)
    extrema[1] = max(extrema[1], lat)
    extrema[2] = min(extrema[2], lon)
    extrema[3] = max(extrema[3], lon)


def reduce_code(code, segments, active_rows, active_clock, background,
                sample_state, extrema):
    output = array("d")
    hist = [0] * 135
    previous = {}
    states = {}
    raw_frames = 0

    def consume(row, clock):
        nonlocal raw_frames
        icao, recv, mono, ticks, lat, lon = row
        t = event_epoch(recv, mono, ticks, clock)
        raw_frames += 1
        state = states.get(icao)
        if state is None:
            states[icao] = ([t], [lat], [lon], t)
            return
        delta = t - state[3]
        if 0.0 <= delta <= BURST_GAP_S:
            state[0].append(t)
            state[1].append(lat)
            state[2].append(lon)
            state = (state[0], state[1], state[2], t)
            states[icao] = state
            return
        add_burst(output, hist, previous, icao, *state[:3],
                  background, sample_state, extrema)
        states[icao] = ([t], [lat], [lon], t)

    for segment in segments:
        db = sqlite3.connect(f"file:{segment['path']}?mode=ro", uri=True)
        try:
            query = (
                "SELECT icao24,recv_ns,monotonic_ns,mlat_ticks,lat,lon "
                "FROM df11_events INDEXED BY df11_code_time "
                "WHERE code_type=? AND code_val=? AND pos_age_ms<=? "
                "AND lat IS NOT NULL AND lon IS NOT NULL "
                "ORDER BY monotonic_ns,mlat_ticks"
            )
            for row in db.execute(query, (code[0], code[1], RAW_LIMIT)):
                consume(row, segment["clock"])
        finally:
            db.close()
    for row in active_rows.get(code, ()):
        icao, recv, mono, ticks, lat, lon = row
        consume((icao, recv, mono, ticks, lat, lon), active_clock)
    for icao, state in states.items():
        add_burst(output, hist, previous, icao, *state[:3],
                  background, sample_state, extrema)
    return output, hist, raw_frames


def write_svg(path, background, rows, extrema):
    width, height = 1200, 780
    left, top, right, bottom = 82, 50, 930, 704
    lat0 = min(LAT_MIN, extrema[0])
    lat1 = max(LAT_MAX, extrema[1])
    lon0 = min(LON_MIN, extrema[2])
    lon1 = max(LON_MAX, extrema[3])
    mid_lat = (lat0 + lat1) / 2
    cos_mid = math.cos(math.radians(mid_lat))
    xspan = (lon1 - lon0) * cos_mid
    yspan = lat1 - lat0
    scale = min((right - left) / xspan, (bottom - top) / yspan)
    plot_width, plot_height = xspan * scale, yspan * scale
    xoff = left + ((right - left) - plot_width) / 2
    yoff = top + ((bottom - top) - plot_height) / 2

    def project(lon, lat):
        x = xoff + (lon - lon0) * cos_mid * scale
        y = yoff + (lat1 - lat) * scale
        return x, y

    def esc(text):
        return (str(text).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}'
        '.label{font-size:12px;font-weight:bold}.grid{stroke:#ddd;stroke-width:1}'
        '.range{fill:none;stroke:#8b9aa8;stroke-dasharray:6 4;stroke-width:1.5}</style>',
        '<text x="82" y="27" font-size="18" font-weight="bold">'
        'DF11 positive II/SI: Burst-Empfangsbereich und Phasenraster-Maxima</text>',
    ]
    for lat in np.arange(math.ceil(lat0), math.floor(lat1) + 1, 1):
        x1, y = project(lon0, lat)
        x2, _ = project(lon1, lat)
        lines.append(f'<line class="grid" x1="{x1:.1f}" y1="{y:.1f}" '
                     f'x2="{x2:.1f}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" '
                     f'text-anchor="end">{lat}°N</text>')
    for lon in np.arange(math.ceil(lon0), math.floor(lon1) + 1, 1):
        x, y1 = project(lon, lat0)
        _, y2 = project(lon, lat1)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{y1:.1f}" '
                     f'x2="{x:.1f}" y2="{y2:.1f}"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{bottom+20}" '
                     f'text-anchor="middle">{lon}°E</text>')
    rx1, ry1 = project(LON_MIN, LAT_MAX)
    rx2, ry2 = project(LON_MAX, LAT_MIN)
    lines.append(f'<rect class="range" x="{rx1:.1f}" y="{ry1:.1f}" '
                 f'width="{rx2-rx1:.1f}" height="{ry2-ry1:.1f}"/>')
    for lat, lon in background:
        x, y = project(lon, lat)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.25" '
                     'fill="#536b7a" fill-opacity=".25"/>')
    converged = [row for row in rows if row["status"] == "converged"]
    colors = ["#c0392b", "#2874a6", "#7d3c98", "#117864", "#ca6f1e",
              "#2e86c1", "#884ea0", "#1e8449", "#b03a2e", "#6e2c00"]
    for i, row in enumerate(converged):
        x, y = project(row["lon_r"], row["lat_r"])
        color = colors[i % len(colors)]
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
                     'stroke="#fff" stroke-width="1.5"/>')
        offset_y = -8 if i % 2 == 0 else 15
        label = (f'{row["code"]} ({row["period_s"]:.3f}s / '
                 f'{row["rpm"]:.2f} RPM)')
        lines.append(f'<text class="label" x="{x+7:.1f}" y="{y+offset_y:.1f}" '
                     f'fill="{color}">{esc(label)}</text>')
    lines.extend([
        f'<text class="small" x="{left}" y="{height-18}">'
        f'Empfangspunkte: {len(background):,} zufällig aus allen Burst-Positionen; '
        f'gestricheltes Rechteck: Suchgebiet</text>',
        f'<text class="small" x="{right+22}" y="{top+5}" font-weight="bold">'
        f'Konvergiert: {len(converged)}</text>',
    ])
    legend_y = top + 28
    for row in converged:
        if legend_y > bottom:
            break
        lines.append(f'<text class="small" x="{right+22}" y="{legend_y}">'
                     f'{esc(row["code"])}  R={row["R_bar_max"]:.3f}</text>')
        legend_y += 16
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    segments = archived_segments()
    active_rows, active_meta = read_active_segment()
    active_clock = active_meta["clock"] if active_meta else None
    codes = ([("II", value) for value in range(1, 16)]
             + [("SI", value) for value in range(1, 64)])
    background = []
    sample_state = [0, random.Random(20261116)]
    extrema = [float("inf"), float("-inf"), float("inf"), float("-inf")]
    results = []
    sample_counts = []
    for code in codes:
        events, hist, raw_frames = reduce_code(
            code, segments, active_rows, active_clock, background,
            sample_state, extrema
        )
        if raw_frames < 150:
            continue
        n_bursts = len(events) // 3
        periods = candidate_periods(hist)
        print(f"BURSTS {code[0]}{code[1]} raw={raw_frames} bursts={n_bursts} "
              f"intervals={sum(hist)} T0={','.join(f'{v:.2f}' for v in periods)}",
              flush=True)
        base = {
            "code": f"{code[0]}{code[1]}",
            "n_bursts": n_bursts,
            "lat_r": None,
            "lon_r": None,
            "period_s": None,
            "rpm": None,
            "R_bar_max": None,
            "rayleigh_p": None,
            "status": "no_period_candidate",
        }
        if periods and n_bursts > 0:
            data_path = OUTPUT / f'{code[0]}{code[1]}.f64'
            with data_path.open("wb") as stream:
                events.tofile(stream)
            points = np.memmap(data_path, dtype=np.float64, mode="r",
                               shape=(n_bursts, 3))
            print(f"GRID {base['code']} n={n_bursts} "
                  f"coarse_periods={len(period_grid(periods))}", flush=True)
            coarse = grid_search(points, period_grid(periods))
            fine = grid_search(points, None, fine_center=coarse[1:])
            rbar, lat, lon, period = fine
            base.update({
                "lat_r": lat,
                "lon_r": lon,
                "period_s": period,
                "rpm": 60.0 / period,
                "R_bar_max": rbar,
                "rayleigh_p": rayleigh_p(n_bursts, rbar),
                "status": "converged" if rbar >= 0.35 else "not_converged",
            })
            print(f"RESULT {base['code']} R={rbar:.4f} "
                  f"site={lat:.3f},{lon:.3f} T={period:.3f}s "
                  f"status={base['status']}", flush=True)
            del points
            data_path.unlink()
        results.append(base)
        sample_counts.append((base["code"], raw_frames, n_bursts))
    csv_path = OUTPUT / "radar-triangulation-v1.csv"
    columns = ["code", "n_bursts", "lat_r", "lon_r", "period_s", "rpm",
               "R_bar_max", "rayleigh_p", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    write_svg(OUTPUT / "radar-triangulation-v1.svg", background, results, extrema)
    print("COVERAGE", json.dumps({
        "archived_segments": len(segments),
        "statuses": dict(collections.Counter(s["status"] for s in segments)),
        "archive_start_utc": dt.datetime.fromtimestamp(
            segments[0]["start_ns"] / 1e9, dt.timezone.utc
        ).isoformat(),
        "active": active_meta,
        "fresh_positive_frames": sum(row[1] for row in sample_counts),
        "burst_count": sum(row[2] for row in sample_counts),
        "codes": len(results),
        "converged": sum(row["status"] == "converged" for row in results),
        "background_points": len(background),
        "csv": str(csv_path),
        "svg": str(OUTPUT / "radar-triangulation-v1.svg"),
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    args = parse_common_options(ARCHIVE, OUTPUT, RX_LAT, RX_LON)
    ARCHIVE, OUTPUT, RX_LAT, RX_LON = args.archive, args.output, args.rx_lat, args.rx_lon
    OUTPUT = create_run_directory("v1", __file__, OUTPUT)
    main()
