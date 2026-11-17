#!/usr/bin/env python3
"""Passive Mode-S tracking with aircraft-disjoint calibration, EKF and RTS.

Horizontal coordinates are WGS84/receiver-ENU ground-footpoint coordinates.
The third state is pressure altitude, NOT geometric ENU Up.
"""
import argparse
import bisect
import collections
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from xml.sax.saxutils import escape

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append("/opt/adsb")
from beast_inspect import frames, interrogator
from experiment_runs import add_common_options, configure_receiver, create_run_directory

SOURCE_NAME = "radar_triangulation_v4"
source = importlib.import_module(SOURCE_NAME)

ARCHIVE = Path(os.environ.get("ADSB_ARCHIVE", "/var/lib/adsb/beast-experiment"))
OUTPUT = Path(os.environ.get("ADSB_OUTPUT", "/tmp/radar-triangulation/v6"))
MAX_WORKERS = min(56, os.cpu_count() or 56)
T0_EPOCH = 1763078400.0
RX_LAT, RX_LON = float(os.environ.get("ADSB_RX_LAT", "51.145")), float(os.environ.get("ADSB_RX_LON", "6.355"))
BURST_GAP_S = 0.5
MAX_INIT_DT = 15.0
MAX_BEARING_GAP = 120.0
REFERENCE_MAX_GAP = 10.0
FT_M, KT_MS = 0.3048, 1852.0 / 3600.0
WGS_A, WGS_F = 6378137.0, 1.0 / 298.257223563
WGS_E2 = WGS_F * (2.0 - WGS_F)
CACHE_VERSION = 1


@dataclass(frozen=True)
class Radar:
    code: str
    lat: float
    lon: float
    period: float
    phase: float
    range_min: float
    range_max: float
    rbar: float
    sigma: float = math.nan

    def angle(self, utc):
        return math.radians((self.phase + 360.0 / self.period
                             * ((utc - T0_EPOCH) % self.period)) % 360.0)


@dataclass(frozen=True)
class Bearing:
    utc: float
    code: str
    angle: float
    count: int


@dataclass(frozen=True)
class Observation:
    utc: float
    kind: str
    values: tuple
    code: str = ""


def wrap(angle):
    value = (angle + math.pi) % (2 * math.pi) - math.pi
    return math.pi if value == -math.pi else value


def ecef(lat, lon, height=0.0):
    p, l = math.radians(lat), math.radians(lon)
    n = WGS_A / math.sqrt(1 - WGS_E2 * math.sin(p) ** 2)
    return np.array(((n + height) * math.cos(p) * math.cos(l),
                     (n + height) * math.cos(p) * math.sin(l),
                     (n * (1 - WGS_E2) + height) * math.sin(p)))


def enu_rotation(lat, lon):
    p, l = math.radians(lat), math.radians(lon)
    return np.array(((-math.sin(l), math.cos(l), 0.0),
                     (-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)),
                     (math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p))))


RX_ECEF = ecef(RX_LAT, RX_LON)
RX_ROT = enu_rotation(RX_LAT, RX_LON)


def geodetic_to_enu(lat, lon, height=0.0):
    return RX_ROT @ (ecef(lat, lon, height) - RX_ECEF)


def enu_to_geodetic(east, north, up):
    x, y, z = RX_ECEF + RX_ROT.T @ np.array((east, north, up))
    radius = math.hypot(x, y)
    lat = math.atan2(z, radius * (1 - WGS_E2))
    for _ in range(12):
        n = WGS_A / math.sqrt(1 - WGS_E2 * math.sin(lat) ** 2)
        updated = math.atan2(z + WGS_E2 * n * math.sin(lat), radius)
        if abs(updated - lat) < 1e-13:
            lat = updated
            break
        lat = updated
    n = WGS_A / math.sqrt(1 - WGS_E2 * math.sin(lat) ** 2)
    height = radius / math.cos(lat) - n
    return math.degrees(lat), math.degrees(math.atan2(y, x)), height


def horizontal_position(east, north):
    up = -(east * east + north * north) / (2 * WGS_A)
    for _ in range(8):
        lat, lon, height = enu_to_geodetic(east, north, up)
        if abs(height) < 1e-5:
            return lat, lon
        local_up = enu_rotation(lat, lon)[2]
        up -= height / float(local_up @ RX_ROT[2])
    raise ValueError("ENU ground-footpoint inversion did not converge")


def true_bearing(lat1, lon1, lat2, lon2):
    # Identical spherical initial-bearing convention to the frozen v5 phase fit.
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    return math.atan2(math.sin(dl) * math.cos(p2),
                      math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))


def velocity_enu(lat, lon, track, speed):
    angle = math.radians(track)
    local = np.array((speed * math.sin(angle), speed * math.cos(angle), 0.0))
    return (RX_ROT @ enu_rotation(lat, lon).T @ local)[:2]


def velocity_true(lat, lon, east_speed, north_speed):
    rotation = enu_rotation(lat, lon)
    grid_velocity = np.array((east_speed, north_speed, 0.0))
    grid_velocity[2] = -float(rotation[2] @ RX_ROT.T @ grid_velocity) / float(
        rotation[2] @ RX_ROT[2])
    return (rotation @ RX_ROT.T @ grid_velocity)[:2]


def load_radars(directory, explicit=None):
    path = explicit or directory / "radar-triangulation-v5-verified.csv"
    if not path.exists() and explicit is None:
        paths = list(directory.glob("*-verified.csv"))
        if not paths:
            raise FileNotFoundError(f"No *-verified.csv in {directory}")
        path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["confidence_score"]) < 70.0:
                continue
            code = row.get("base_code") or re.split(r"[@#]", row["code"])[0]
            if not re.fullmatch(r"(II|SI)[1-9][0-9]*", code):
                raise ValueError(f"Invalid radar code: {code}")
            radar = Radar(code, *(float(row[key]) for key in (
                "lat_r", "lon_r", "period_s", "phi0_deg", "range_p05_km",
                "range_p95_km", "R_bar_max")))
            if (not all(math.isfinite(v) for v in (
                    radar.lat, radar.lon, radar.period, radar.phase,
                    radar.range_min, radar.range_max, radar.rbar))
                    or not 0 < radar.period < 60 or not 0 <= radar.range_min < radar.range_max
                    or not 0 < radar.rbar <= 1):
                raise ValueError(f"Invalid calibration for {code}")
            if code in result:
                raise ValueError(f"Multiple physical radars share {code}; ambiguous burst assignment")
            result[code] = radar
    if len(result) < 2:
        raise ValueError("At least two unambiguous calibrated radar codes are required")
    return result, path


@contextlib.contextmanager
def connect_readonly(path):
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        db.execute("PRAGMA mmap_size=268435456")
        yield db
    finally:
        db.close()


def metadata_index(path):
    chunks, ends, start, closed = [], [], None, False
    cutoff = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < cutoff:
            line = stream.readline()
            if stream.tell() > cutoff or not line.endswith(b"\n"):
                print(f"PARTIAL_METADATA ignored final unpublished line: {path}", flush=True)
                break
            if re.match(rb'\{"type"\s*:\s*"snapshot"', line):
                continue
            row = json.loads(line)
            if row["type"] == "start":
                start = row
            elif row["type"] == "chunk":
                if row["offset"] != (ends[-1] if ends else 0) or row["size"] <= 0:
                    raise ValueError(f"Noncontiguous chunks: {path}")
                chunks.append(row)
                ends.append(row["offset"] + row["size"])
            elif row["type"] == "end":
                closed = True
    if start is None or not chunks:
        raise ValueError(f"No published capture chunks: {path}")
    return start, chunks, ends, closed


def frame_time(clock, chunk, ticks):
    mono0, tick0, rate, intercept = clock
    return (mono0 + (ticks - tick0 - intercept) * (1e9 / rate)
            + chunk["realtime_ns"] - chunk["monotonic_ns"]) * 1e-9


def active_clock(raw, start, chunks, ends):
    anchors, last = [], None
    with raw.open("rb") as stream:
        for _, end, kind, body in frames(source.LimitedReader(stream, ends[-1])):
            if kind not in (0x32, 0x33):
                continue
            chunk = chunks[bisect.bisect_left(ends, end)]
            mono = chunk["monotonic_ns"]
            ticks = int.from_bytes(body[:6], "big")
            if ticks and mono - start["monotonic_ns"] >= 10_000_000_000:
                if last is None or mono - last >= 60_000_000_000:
                    anchors.append((mono, ticks))
                    last = mono
                    if len(anchors) == 3:
                        return source.fit_clock(anchors)
    raise ValueError(f"Active capture needs three clock anchors: {raw}")


def discover_segments(archive, include_active=True, limit=None):
    source.ARCHIVE = archive
    archived = source.archived_segments()
    if limit is not None:
        archived = archived[:limit]
    segments, schemas = [], {}
    for item in archived:
        db_path = item["path"]
        with connect_readonly(db_path) as db:
            schemas[db_path.name] = {
                name: sql for name, sql in db.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='table'")
            }
        raw = db_path.with_suffix(".raw")
        meta = db_path.with_suffix(".jsonl")
        if not raw.exists() or not meta.exists():
            raise FileNotFoundError(f"Original raw/jsonl pair required for {db_path.name}")
        start, chunks, ends, closed = metadata_index(meta)
        segments.append(dict(raw=raw, meta=meta, db=db_path, clock=item["clock"],
                             chunks=chunks, ends=ends, start=start, closed=closed))
    if include_active and limit is None:
        known = {row["raw"].stem for row in segments}
        for meta in sorted(archive.glob("capture-*.jsonl")):
            if meta.stem in known:
                continue
            start, chunks, ends, closed = metadata_index(meta)
            raw = meta.with_suffix(".raw")
            if closed:
                raise ValueError(f"Closed but not yet archived capture: {raw}; wait for archive timer")
            clock = active_clock(raw, start, chunks, ends)
            segments.append(dict(raw=raw, meta=meta, db=None, clock=clock,
                                 chunks=chunks, ends=ends, start=start, closed=False))
    if not segments:
        raise FileNotFoundError(f"No capture segments in {archive}")
    for segment in segments:
        if segment["raw"].stat().st_size < segment["ends"][-1]:
            raise ValueError(f"Raw file shorter than published cutoff: {segment['raw']}")
    return segments, schemas


def load_burst_fragments(args):
    segment, radars = args
    states, result = {}, []

    def add(icao, code, utc):
        key = (icao, code)
        times = states.get(key)
        if times and not 0 <= utc - times[-1] <= BURST_GAP_S:
            result.append((icao, code, tuple(times)))
            times = None
        if times is None:
            states[key] = [utc]
        else:
            times.append(utc)

    if segment["db"] is not None:
        with connect_readonly(segment["db"]) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(df11_events)")}
            if not {"icao24", "mlat_ticks", "monotonic_ns", "recv_ns"} <= columns:
                raise ValueError(f"Unsupported DF11 schema: {segment['db']}")
            for code in radars:
                code_type, code_value = re.fullmatch(r"(II|SI)(\d+)", code).groups()
                for icao, recv, mono, ticks in db.execute(
                        "SELECT icao24,recv_ns,monotonic_ns,mlat_ticks FROM df11_events "
                        "INDEXED BY df11_code_time WHERE code_type=? AND code_val=? "
                        "ORDER BY monotonic_ns,mlat_ticks", (code_type, int(code_value))):
                    add(icao, code, frame_time(segment["clock"],
                        {"realtime_ns": recv, "monotonic_ns": mono}, ticks))
    else:
        with segment["raw"].open("rb") as stream:
            for _, end, kind, body in frames(source.LimitedReader(stream, segment["ends"][-1])):
                if kind != 0x32:
                    continue
                code = interrogator(body[7:])
                label = f"{code[0]}{code[1]}" if code else ""
                if label in radars:
                    chunk = segment["chunks"][bisect.bisect_left(segment["ends"], end)]
                    add(body[8:11].hex().upper(), label,
                        frame_time(segment["clock"], chunk, int.from_bytes(body[:6], "big")))
    result.extend((icao, code, tuple(times)) for (icao, code), times in states.items())
    return result


def merge_bursts(fragments, radars):
    grouped = collections.defaultdict(list)
    for icao, code, times in fragments:
        grouped[icao, code].append(times)
    result = collections.defaultdict(list)
    for (icao, code), pieces in grouped.items():
        pending = []
        for times in sorted(pieces, key=lambda p: p[0]):
            if pending and times[0] - pending[-1] > BURST_GAP_S:
                if len(pending) >= 2:
                    utc = statistics.median(pending)
                    result[icao].append(Bearing(utc, code, radars[code].angle(utc), len(pending)))
                pending = []
            pending.extend(times)
        if len(pending) >= 2:
            utc = statistics.median(pending)
            result[icao].append(Bearing(utc, code, radars[code].angle(utc), len(pending)))
    return {
        icao: sorted(rows, key=lambda row: row.utc) for icao, rows in result.items()
        if len(rows) >= 6 and len({row.code for row in rows}) >= 2
        and not "4D03C0" <= icao <= "4D03D1"
    }


def _crc_table():
    result = []
    for byte in range(256):
        value = byte << 16
        for _ in range(8):
            value = ((value << 1) ^ (0xFFF409 if value & 0x800000 else 0)) & 0xFFFFFF
        result.append(value)
    return result


CRC_TABLE = _crc_table()


def modes_crc(payload):
    value = 0
    for byte in payload:
        value = (((value << 8) | byte) ^ CRC_TABLE[value >> 16]) & 0xFFFFFF
    return value


def decode_altitude(ac):
    if ac == 0 or ac & 0x40:
        return None
    if ac & 0x10:
        feet = ((((ac & 0x1F80) >> 2) | ((ac & 0x20) >> 1) | (ac & 15)) * 25 - 1000)
    else:
        mode_a = sum(dst for src, dst in (
            (0x1000, 0x10), (0x800, 0x1000), (0x400, 0x20), (0x200, 0x2000),
            (0x100, 0x40), (0x80, 0x4000), (0x20, 0x100), (0x10, 1),
            (8, 0x200), (4, 2), (2, 0x400), (1, 4)) if ac & src)
        if mode_a & 0x888B or not mode_a & 0x70:
            return None
        hundreds = 0
        for bit, mask in ((0x10, 7), (0x20, 3), (0x40, 1)):
            if mode_a & bit:
                hundreds ^= mask
        if hundreds & 5 == 5:
            hundreds ^= 2
        if hundreds > 5:
            return None
        five_hundreds = 0
        for bit, mask in ((2, 255), (4, 127), (0x1000, 63), (0x2000, 31),
                          (0x4000, 15), (0x100, 7), (0x200, 3), (0x400, 1)):
            if mode_a & bit:
                five_hundreds ^= mask
        if five_hundreds & 1:
            hundreds = 6 - hundreds
        feet = (five_hundreds * 5 + hundreds - 13) * 100
    return float(feet) if -1000 <= feet <= 65000 else None


def decode_bds50(payload):
    mb = int.from_bytes(payload[4:11], "big")

    def bits(first, last):
        return (mb >> (56 - last)) & ((1 << (last - first + 1)) - 1)

    # Strict shape inference, not an explicit register identifier.
    if any(not bits(i, i) for i in (1, 12, 24, 35, 46)):
        return None, "commb_unidentified"
    roll = (bits(3, 11) - 512 * bits(2, 2)) * 45.0 / 256
    track = ((bits(14, 23) - 1024 * bits(13, 13)) * 90.0 / 512) % 360
    gs, tas = bits(25, 34) * 2.0, bits(47, 56) * 2.0
    rate = (bits(37, 45) - 512 * bits(36, 36)) * 8.0 / 256
    if not (abs(roll) <= 50 and 40 <= gs <= 650 and 40 <= tas <= 650
            and abs(gs - tas) <= 150 and abs(rate) <= 8):
        return None, "commb_unidentified"
    # Reject payloads that also plausibly describe BDS 6,0.
    if (bits(1, 1) and bits(13, 13) and bits(24, 24)
            and 40 <= bits(14, 23) <= 650 and 0.1 <= bits(25, 34) * 0.004 <= 1.2):
        return None, "commb_ambiguous_50_60"
    expected_roll = math.degrees(math.atan(math.radians(rate) * gs * KT_MS / 9.80665))
    if abs(expected_roll - roll) > 15:
        return None, "commb_inconsistent_turn"
    return (track, gs * KT_MS), "bds50_velocity"


def cpr_nl(lat):
    lat = abs(lat)
    if lat >= 87:
        return 2 if lat == 87 else 1
    return int(math.floor(2 * math.pi / math.acos(
        1 - (1 - math.cos(math.pi / 30)) / math.cos(math.radians(lat)) ** 2)))


def cpr_position(even, odd):
    # Tuples: (UTC, latitude fraction, longitude fraction).
    if abs(even[0] - odd[0]) > 10:
        return None
    j = math.floor(59 * even[1] - 60 * odd[1] + 0.5)
    lat_e, lat_o = 6 * ((j % 60) + even[1]), 360 / 59 * ((j % 59) + odd[1])
    lat_e = lat_e - 360 if lat_e >= 270 else lat_e
    lat_o = lat_o - 360 if lat_o >= 270 else lat_o
    if not -90 <= lat_e <= 90 or not -90 <= lat_o <= 90 or cpr_nl(lat_e) != cpr_nl(lat_o):
        return None
    use_odd = odd[0] > even[0]
    lat = lat_o if use_odd else lat_e
    nl = cpr_nl(lat)
    m = math.floor(even[2] * (nl - 1) - odd[2] * nl + 0.5)
    ni = max(1, nl - int(use_odd))
    lon = 360 / ni * ((m % ni) + (odd[2] if use_odd else even[2]))
    return lat, (lon + 180) % 360 - 180


def decode_raw_segment(args):
    segment, addresses, cache_dir = args
    signature = hashlib.sha256(json.dumps((
        CACHE_VERSION, str(segment["raw"]), segment["ends"][-1], segment["clock"], sorted(addresses),
        RX_LAT, RX_LON
    )).encode()).hexdigest()
    target = cache_dir / f"{segment['raw'].stem}-{signature[:16]}.sqlite"
    if target.exists():
        with connect_readonly(target) as db:
            return target, json.loads(db.execute("SELECT value FROM info").fetchone()[0])
    temporary = target.with_suffix(f".partial-{os.getpid()}")
    counts = collections.Counter()
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;
            CREATE TABLE obs (icao TEXT, utc REAL, kind TEXT, a REAL, b REAL);
            CREATE TABLE reference (icao TEXT, utc REAL, lat REAL, lon REAL);
            CREATE TABLE df17_position (icao TEXT PRIMARY KEY);
            CREATE TABLE info (value TEXT);
        """)
        observations, references, seen17, cpr, last = [], [], set(), {}, {}
        previous_ticks = None
        with segment["raw"].open("rb") as stream:
            for _, end, kind, body in frames(source.LimitedReader(stream, segment["ends"][-1])):
                if kind not in (0x32, 0x33):
                    continue
                payload = body[7:]
                df = payload[0] >> 3
                counts[f"DF{df}"] += 1
                if len(payload) != (14 if df >= 16 else 7):
                    counts["bad_frame_length"] += 1
                    continue
                if df not in (4, 11, 17, 20, 21):
                    continue
                ticks = int.from_bytes(body[:6], "big")
                if ticks == 0:
                    continue
                if previous_ticks is not None and ticks + 12_000_000 < previous_ticks:
                    raise ValueError(f"Receiver clock discontinuity: {segment['raw']}")
                previous_ticks = ticks
                icao = payload[1:4].hex().upper() if df in (11, 17) else f"{modes_crc(payload):06X}"
                if icao not in addresses:
                    continue
                chunk = segment["chunks"][bisect.bisect_left(segment["ends"], end)]
                utc = frame_time(segment["clock"], chunk, ticks)
                if df == 17:
                    if len(payload) != 14 or modes_crc(payload):
                        counts["df17_bad_crc"] += 1
                        continue
                    tc = payload[4] >> 3
                    if 5 <= tc <= 18 or 20 <= tc <= 22:
                        seen17.add(icao)
                    if not (9 <= tc <= 18 or 20 <= tc <= 22):
                        continue
                    odd = (payload[6] >> 2) & 1
                    fraction_lat = (((payload[6] & 3) << 15) | (payload[7] << 7)
                                    | (payload[8] >> 1)) / 131072.0
                    fraction_lon = (((payload[8] & 1) << 16) | (payload[9] << 8)
                                    | payload[10]) / 131072.0
                    # Do not mix barometric and GNSS CPR encodings.
                    pair = cpr.setdefault((icao, tc >= 20), {})
                    pair[odd] = (utc, fraction_lat, fraction_lon)
                    if len(pair) == 2 and utc - last.get((icao, "R"), -math.inf) >= 2:
                        position = cpr_position(pair[0], pair[1])
                        if position is not None:
                            lat, lon = position
                            if source.haversine_km(RX_LAT, RX_LON, lat, lon) <= 650:
                                references.append((icao, utc, lat, lon))
                                last[icao, "R"] = utc
                                counts["reference_positions"] += 1
                elif df in (4, 20):
                    altitude = decode_altitude(int.from_bytes(payload[2:4], "big") & 0x1FFF)
                    if altitude is None:
                        counts["altitude_unavailable"] += 1
                    if altitude is not None and utc - last.get((icao, "A"), -math.inf) >= 2:
                        observations.append((icao, utc, "A", altitude * FT_M, None))
                        last[icao, "A"] = utc
                        counts["altitude_updates"] += 1
                if df in (20, 21):
                    velocity, status = decode_bds50(payload)
                    counts[status] += 1
                    if velocity is not None and utc - last.get((icao, "V"), -math.inf) >= 2:
                        observations.append((icao, utc, "V", *velocity))
                        last[icao, "V"] = utc
                if len(observations) + len(references) >= 5000:
                    db.executemany("INSERT INTO obs VALUES (?,?,?,?,?)", observations)
                    db.executemany("INSERT INTO reference VALUES (?,?,?,?)", references)
                    observations.clear()
                    references.clear()
        db.executemany("INSERT INTO obs VALUES (?,?,?,?,?)", observations)
        db.executemany("INSERT INTO reference VALUES (?,?,?,?)", references)
        db.executemany("INSERT INTO df17_position VALUES (?)", ((icao,) for icao in seen17))
        db.executescript("CREATE INDEX obs_icao ON obs(icao,utc);"
                         "CREATE INDEX reference_icao ON reference(icao,utc);")
        db.execute("INSERT INTO info VALUES (?)", (json.dumps(dict(counts)),))
        db.commit()
        db.close()
        os.replace(temporary, target)
    except BaseException:
        db.close()
        temporary.unlink(missing_ok=True)
        raise
    return target, dict(counts)


def load_aircraft(paths, icao, references=False):
    rows = []
    for path in paths:
        with connect_readonly(path) as db:
            if references:
                rows.extend(db.execute("SELECT utc,lat,lon FROM reference WHERE icao=? ORDER BY utc",
                                       (icao,)))
            else:
                rows.extend(Observation(utc, kind, (a,) if kind == "A" else (a, b))
                            for utc, kind, a, b in db.execute(
                                "SELECT utc,kind,a,b FROM obs WHERE icao=? ORDER BY utc", (icao,)))
    return sorted(rows, key=(lambda row: row[0]) if references else (lambda row: row.utc))


def reference_at(rows, utc):
    index = bisect.bisect_left(rows, (utc, -math.inf, -math.inf))
    if index < len(rows) and rows[index][0] == utc:
        return rows[index][1:]
    if index == 0 or index == len(rows):
        return None
    before, after = rows[index - 1], rows[index]
    if after[0] - before[0] > REFERENCE_MAX_GAP:
        return None
    fraction = (utc - before[0]) / (after[0] - before[0])
    if source.haversine_km(before[1], before[2], after[1], after[2]) * 1000 > (
            400 * (after[0] - before[0]) + 500):
        return None
    return (before[1] + fraction * (after[1] - before[1]),
            before[2] + fraction * (after[2] - before[2]))


def select_cohorts(bursts, with_df17, with_reference):
    known = sorted(set(bursts) & with_reference,
                   key=lambda icao: (-len(bursts[icao]), icao))
    if len(known) < 6:
        raise ValueError("At least six DF17 reference aircraft are needed for disjoint cohorts")
    calibration = set()
    codes = {row.code for rows in bursts.values() for row in rows}
    for code in sorted(codes):
        covering = [icao for icao in known if any(row.code == code for row in bursts[icao])]
        calibration.update(covering[:min(5, max(2, len(covering) // 5))])
    target = min(30, max(len(calibration), len(known) // 5))
    for icao in known:
        if len(calibration) >= target:
            break
        calibration.add(icao)
    validation = [icao for icao in known if icao not in calibration][:150]
    if not validation:
        raise ValueError("No reference aircraft left for blind validation")
    mode_s = sorted(set(bursts) - with_df17)
    return sorted(calibration), validation, mode_s


def calibrate_noise(radars, bursts, paths, aircraft):
    residuals = collections.defaultdict(list)
    contributors = collections.defaultdict(set)
    for icao in aircraft:
        reference = load_aircraft(paths, icao, references=True)
        for row in bursts[icao]:
            truth = reference_at(reference, row.utc)
            if truth is not None:
                radar = radars[row.code]
                residuals[row.code].append(wrap(true_bearing(
                    radar.lat, radar.lon, *truth) - row.angle))
                contributors[row.code].add(icao)
    calibrated, report = {}, {}
    for code, radar in radars.items():
        values = np.asarray(residuals[code])
        if len(values) < 100 or len(contributors[code]) < 2:
            raise ValueError(f"Insufficient independent noise calibration for {code}: "
                             f"{len(values)} samples / {len(contributors[code])} aircraft")
        center = math.atan2(float(np.sum(np.sin(values))), float(np.sum(np.cos(values))))
        deviations = np.array([wrap(value - center) for value in values])
        bias = wrap(center + float(np.median(deviations)))
        centered = np.array([wrap(value - bias) for value in values])
        robust_sigma = 1.4826 * float(np.median(np.abs(centered)))
        # Retain frozen phase: an uncorrected calibration bias is uncertainty too.
        sigma = max(math.radians(0.25), math.hypot(robust_sigma, bias))
        if sigma >= math.radians(45):
            raise ValueError(f"{code}: no usable mainlobe peak; calibrated sigma >= 45 degrees")
        calibrated[code] = replace(radar, sigma=sigma)
        report[code] = dict(sigma_deg=math.degrees(sigma), bias_deg=math.degrees(bias),
                           samples=len(values), aircraft=len(contributors[code]))
    return calibrated, report


def initial_position(rows, radars):
    latest = max(row.utc for row in rows)
    by_radar = {row.code: row for row in rows if latest - row.utc <= MAX_INIT_DT}
    if len(by_radar) < 2:
        return None
    matrix, target = [], []
    for code, row in by_radar.items():
        radar = radars[code]
        origin = geodetic_to_enu(radar.lat, radar.lon)[:2]
        direction = velocity_enu(radar.lat, radar.lon, math.degrees(row.angle), 1.0)
        direction /= np.linalg.norm(direction)
        normal = np.array((direction[1], -direction[0]))
        distance = 500 * (radar.range_min + radar.range_max)
        sigma = max(100.0, radar.sigma * distance)
        matrix.append(normal / sigma)
        target.append(float(normal @ origin) / sigma)
    matrix = np.asarray(matrix)
    normal_matrix = matrix.T @ matrix
    if np.linalg.cond(normal_matrix) > 1e4:
        return None
    position = np.linalg.solve(normal_matrix, matrix.T @ target)
    lat, lon = horizontal_position(*position)
    for code, row in by_radar.items():
        radar = radars[code]
        distance = source.haversine_km(radar.lat, radar.lon, lat, lon)
        if (not radar.range_min <= distance <= radar.range_max
                or abs(wrap(true_bearing(radar.lat, radar.lon, lat, lon) - row.angle)) > math.pi / 2):
            return None
    covariance = np.linalg.inv(normal_matrix) + np.eye(2) * (250 * MAX_INIT_DT) ** 2
    return position, covariance


def transition(delta):
    f = np.eye(6)
    f[:3, 3:] = np.eye(3) * delta
    q = np.zeros((6, 6))
    for index, acceleration in enumerate((2.5, 2.5, 0.8)):
        q[index, index] = acceleration ** 2 * delta ** 3 / 3
        q[index, index + 3] = q[index + 3, index] = acceleration ** 2 * delta ** 2 / 2
        q[index + 3, index + 3] = acceleration ** 2 * delta
    return f, q


def kalman_update(x, covariance, residual, h, noise, gate):
    innovation = h @ covariance @ h.T + noise
    mahalanobis = float(residual @ np.linalg.solve(innovation, residual))
    if mahalanobis > gate:
        return x, covariance, False
    gain = np.linalg.solve(innovation, h @ covariance).T
    updated = x + gain @ residual
    ikh = np.eye(6) - gain @ h
    updated_cov = ikh @ covariance @ ikh.T + gain @ noise @ gain.T
    return updated, (updated_cov + updated_cov.T) * 0.5, True


def bearing_update(x, covariance, row, radar):
    lat, lon = horizontal_position(x[0], x[1])
    expected = true_bearing(radar.lat, radar.lon, lat, lon)
    h = np.zeros((1, 6))
    for axis in (0, 1):
        shifted = x[:2].copy()
        shifted[axis] += 1.0
        lat_s, lon_s = horizontal_position(*shifted)
        h[0, axis] = wrap(true_bearing(radar.lat, radar.lon, lat_s, lon_s) - expected)
    return kalman_update(x, covariance, np.array((wrap(row.values[0] - expected),)),
                         h, np.array(((radar.sigma ** 2,),)), 10.828)


def rts_smooth(filtered, predicted, transitions):
    states = [item[0].copy() for item in filtered]
    covariances = [item[1].copy() for item in filtered]
    for index in range(len(states) - 2, -1, -1):
        f = transitions[index + 1]
        pred_x, pred_cov = predicted[index + 1]
        x, covariance = filtered[index]
        gain = np.linalg.solve(pred_cov, f @ covariance).T
        states[index] = x + gain @ (states[index + 1] - pred_x)
        covariances[index] = covariance + gain @ (covariances[index + 1] - pred_cov) @ gain.T
        covariances[index] = (covariances[index] + covariances[index].T) * 0.5
        if np.linalg.eigvalsh(covariances[index])[0] < -1e-3:
            raise ArithmeticError("RTS covariance lost positive semidefiniteness")
    return list(zip(states, covariances))


def track_flight(bearings, ancillary, radars):
    """No reference positions accepted by this interface."""
    if any(row.kind not in ("A", "V") for row in ancillary):
        raise ValueError("Only DF4/DF20 pressure altitude and BDS50 ground velocity are tracking inputs")
    events = [Observation(row.utc, "B", (row.angle,), row.code) for row in bearings] + ancillary
    events.sort(key=lambda row: (row.utc, row.kind))
    recent, initial_alt, initial_velocity = [], None, None
    x, covariance, last_time, last_bearing = None, None, None, None
    filtered, predicted, transitions, times, update_counts, radar_sets = [], [], [], [], [], []
    used_radars, n_updates, altitude_seen = set(), 0, False
    counts = collections.Counter()
    for row in events:
        if x is None:
            if row.kind == "A":
                initial_alt = row
            elif row.kind == "V":
                initial_velocity = row
            elif row.kind == "B":
                recent.append(Bearing(row.utc, row.code, row.values[0], 2))
                recent = [item for item in recent if row.utc - item.utc <= MAX_INIT_DT]
                seed = initial_position(recent, radars)
                if seed is None:
                    continue
                position, cov_xy = seed
                x = np.zeros(6)
                x[:2] = position
                covariance = np.diag((1e8, 1e8, 1e8, 200 ** 2, 200 ** 2, 30 ** 2))
                covariance[:2, :2] = cov_xy
                if initial_alt is not None and row.utc - initial_alt.utc <= 30:
                    x[2] = initial_alt.values[0]
                    covariance[2, 2] = 35 ** 2
                    altitude_seen = True
                if initial_velocity is not None and row.utc - initial_velocity.utc <= 15:
                    lat, lon = horizontal_position(*position)
                    x[3:5] = velocity_enu(lat, lon, *initial_velocity.values)
                    covariance[3:5, 3:5] = np.eye(2) * 10 ** 2
                last_time = last_bearing = row.utc
                used_radars = {item.code for item in recent}
                n_updates = len(used_radars)
                # The seed already consumed these bearings; do not assimilate one twice.
                filtered.append((x.copy(), covariance.copy()))
                predicted.append((x.copy(), covariance.copy()))
                transitions.append(np.eye(6))
                times.append(row.utc)
                update_counts.append(n_updates)
                radar_sets.append(set(used_radars))
            continue
        if row.utc - last_bearing > MAX_BEARING_GAP:
            counts["track_lost"] += 1
            counts["_resume_at"] = row.utc
            break
        delta = row.utc - last_time
        f, q = transition(delta)
        xp, pp = f @ x, f @ covariance @ f.T + q
        x, covariance = xp, pp
        accepted = False
        if row.kind == "B":
            x, covariance, accepted = bearing_update(x, covariance, row, radars[row.code])
            if accepted:
                used_radars.add(row.code)
                last_bearing = row.utc
                n_updates += 1
        elif row.kind == "A":
            h = np.zeros((1, 6))
            h[0, 2] = 1
            x, covariance, accepted = kalman_update(
                x, covariance, np.array((row.values[0] - x[2],)), h,
                np.array(((35 ** 2,),)), 10.828)
            altitude_seen |= accepted
        elif row.kind == "V":
            lat, lon = horizontal_position(x[0], x[1])
            velocity = velocity_enu(lat, lon, *row.values)
            h = np.zeros((2, 6))
            h[:, 3:5] = np.eye(2)
            x, covariance, accepted = kalman_update(
                x, covariance, velocity - x[3:5], h, np.eye(2) * 10 ** 2, 13.816)
        else:
            raise ValueError(f"Tracking cannot consume observation kind {row.kind}")
        counts[f"{row.kind}_{'accepted' if accepted else 'rejected'}"] += 1
        filtered.append((x.copy(), covariance.copy()))
        predicted.append((xp, pp))
        transitions.append(f)
        times.append(row.utc)
        update_counts.append(n_updates)
        radar_sets.append(set(used_radars))
        last_time = row.utc
    if x is None or n_updates < 6 or len(used_radars) < 2:
        return [], dict(counts)
    smoothed = rts_smooth(filtered, predicted, transitions)
    result = []
    for utc, (state, cov), updates, codes in zip(times, smoothed, update_counts, radar_sets):
        lat, lon = horizontal_position(state[0], state[1])
        local_velocity = velocity_true(lat, lon, state[3], state[4])
        speed = float(np.linalg.norm(local_velocity)) / KT_MS
        track = math.degrees(math.atan2(local_velocity[0], local_velocity[1])) % 360
        sigma = math.sqrt(max(0.0, float(np.linalg.eigvalsh(cov[:2, :2])[-1]))) / 1000
        result.append(dict(utc=utc, lat=lat, lon=lon,
                           alt_ft=state[2] / FT_M if altitude_seen else None,
                           speed_kt=speed, track_deg=track, pos_sigma_km=sigma,
                           n_updates=updates, radars=";".join(sorted(codes))))
    counts["altitude_unobserved"] += int(not altitude_seen)
    return result, dict(counts)


def flight_groups(bearings):
    group = []
    for row in bearings:
        if group and row.utc - group[-1].utc > MAX_BEARING_GAP:
            yield group
            group = []
        group.append(row)
    if group:
        yield group


def error_statistics(errors):
    values = np.asarray(errors, dtype=float)
    if not len(values):
        return dict(n_matches=0, median_km=None, mean_km=None, rmse_km=None, p90_km=None)
    return dict(n_matches=len(values), median_km=float(np.median(values)),
                mean_km=float(np.mean(values)), rmse_km=float(np.sqrt(np.mean(values ** 2))),
                p90_km=float(np.percentile(values, 90)))


def run_aircraft(args):
    icao, bearings, paths, radars, validate = args
    ancillary = load_aircraft(paths, icao)
    # Reference loading is deliberately outside track_flight and only for stage 1.
    reference = load_aircraft(paths, icao, references=True) if validate else []
    tracks, metrics, errors, counts = [], [], [], collections.Counter()
    number = 0
    for group in flight_groups(bearings):
        while len(group) >= 6 and len({row.code for row in group}) >= 2:
            number += 1
            obs = [row for row in ancillary if group[0].utc - 30 <= row.utc <= group[-1].utc]
            points, diagnostics = track_flight(group, obs, radars)
            resume_at = diagnostics.pop("_resume_at", None)
            counts.update(diagnostics)
            flight_errors = []
            for point in points:
                truth = reference_at(reference, point["utc"]) if validate else None
                if truth is not None:
                    flight_errors.append(source.haversine_km(point["lat"], point["lon"], *truth))
            if validate:
                metrics.append(dict(icao24=icao, flight_id=f"{icao}-{number}",
                                    status="ok" if flight_errors else "no_reference_overlap"
                                    if points else "not_initialized_or_insufficient_updates",
                                    **error_statistics(flight_errors)))
            errors.extend(flight_errors)
            tracks.append((f"{icao}-{number}", points))
            if resume_at is None:
                break
            group = [row for row in group if row.utc >= resume_at]
    return icao, tracks, metrics, errors, dict(counts), reference


def write_map(path, radars, overlays, summary):
    width, height = 1400, 1000
    lon0, lon1, lat0, lat1 = -5.0, 16.0, 45.0, 58.0

    def project(lon, lat):
        return 30 + (lon - lon0) / (lon1 - lon0) * 1340, 80 + (lat1 - lat) / (lat1 - lat0) * 850

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#dbeafe"/>',
             '<text x="30" y="28" font-family="sans-serif" font-size="20">V6 Mode-S tracking</text>',
             '<text x="30" y="52" font-family="sans-serif" font-size="13">'
             'DF17 truth: black dashed | blind RTS: orange | no-DF17-position RTS: blue</text>',
             '<defs><clipPath id="map"><rect x="30" y="80" width="1340" height="850"/>'
             '</clipPath></defs><g clip-path="url(#map)">']
    for polygon in source.EUROPE_POLYS:
        points = " ".join(f"{project(lon, lat)[0]:.1f},{project(lon, lat)[1]:.1f}"
                          for lon, lat in polygon)
        lines.append(f'<polygon points="{points}" fill="#f3f4f6" stroke="#64748b"/>')
    labels = []
    for icao, style, points in overlays:
        if len(points) < 2:
            continue
        color, dash = ("#111827", 'stroke-dasharray="5 4"') if style == "truth" else (
            ("#ea580c", "") if style == "blind" else ("#2563eb", ""))
        coords = " ".join(f"{project(lon, lat)[0]:.1f},{project(lon, lat)[1]:.1f}"
                          for lat, lon in points)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                     f'stroke-width="1.3" stroke-opacity="0.7" {dash}>'
                     f'<title>{escape(icao)} ({escape(style)})</title></polyline>')
        visible = [(x, y) for lat, lon in points for x, y in [project(lon, lat)]
                   if 35 <= x <= 1280 and 100 <= y <= 925]
        if visible:
            x, y = visible[-1]
            caption = icao if style == "mode_s" else f"{icao} ({style})"
            labels.append(f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" fill="{color}" '
                          f'font-family="monospace" font-size="11" stroke="white" '
                          f'stroke-width="3" paint-order="stroke">{escape(caption)}</text>')
    lines.extend(labels)
    for radar in radars.values():
        x, y = project(radar.lon, radar.lat)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#dc2626"/>')
        lines.append(f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="12">'
                     f'{escape(radar.code)}</text>')
    lines.append('</g>')
    lines.append(f'<text x="30" y="965" font-family="sans-serif" font-size="12">'
                 f'{escape(summary)}</text></svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    global RX_LAT, RX_LON, RX_ECEF, RX_ROT
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_options(parser, ARCHIVE, OUTPUT, RX_LAT, RX_LON)
    parser.add_argument("--radars", type=Path)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--no-active", action="store_true")
    args = parser.parse_args(argv)
    configure_receiver(parser, args)
    if not 1 <= args.workers <= 56 or (args.max_segments is not None and args.max_segments < 1):
        parser.error("workers must be 1..56 and max-segments must be positive")
    RX_LAT, RX_LON = args.rx_lat, args.rx_lon
    RX_ECEF, RX_ROT = ecef(RX_LAT, RX_LON), enu_rotation(RX_LAT, RX_LON)
    args.output_root = args.output
    args.output = create_run_directory("v6", __file__, args.output)
    with (args.output / ".v6.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


def execute(args):
    started = time.monotonic()
    radars, calibration_path = load_radars(args.output_root.parent, args.radars)
    print(f"V6 START workers={args.workers} loader={SOURCE_NAME} radars={calibration_path}", flush=True)
    segments, schemas = discover_segments(args.archive, not args.no_active, args.max_segments)
    fragments = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(load_burst_fragments, (segment, radars)) for segment in segments]
        for future in as_completed(futures):
            fragments.extend(future.result())
    bursts = merge_bursts(fragments, radars)
    del fragments
    if not bursts:
        raise ValueError("No aircraft with six mainlobe bursts on two calibrated radars")
    print(f"V6 LOAD segments={len(segments)} candidate_aircraft={len(bursts)}", flush=True)
    cache_dir = args.output_root.parent / "cache" / "mode-s-raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths, raw_counts = [], collections.Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode_raw_segment, (segment, set(bursts), cache_dir))
                   for segment in segments]
        for future in as_completed(futures):
            path, counts = future.result()
            paths.append(path)
            raw_counts.update(counts)
            print(f"V6 RAW {len(paths)}/{len(segments)} {path.name}", flush=True)
    with_df17, with_reference = set(), set()
    for path in paths:
        with connect_readonly(path) as db:
            with_df17.update(row[0] for row in db.execute("SELECT icao FROM df17_position"))
            with_reference.update(row[0] for row in db.execute("SELECT DISTINCT icao FROM reference"))
    calibration_ids, validation_ids, mode_s_ids = select_cohorts(bursts, with_df17, with_reference)
    radars, noise = calibrate_noise(radars, bursts, paths, calibration_ids)
    print("V6 ANGULAR_CALIBRATION", json.dumps(noise, sort_keys=True), flush=True)
    print(f"V6 COHORTS calibration={len(calibration_ids)} blind={len(validation_ids)} "
          f"mode_s_only={len(mode_s_ids)}", flush=True)
    output = args.output
    columns = ["icao24", "utc", "lat", "lon", "alt_ft", "speed_kt", "track_deg",
               "pos_sigma_km", "n_updates", "radars"]
    stats_columns = ["icao24", "flight_id", "status", "n_matches", "median_km",
                     "mean_km", "rmse_km", "p90_km"]
    overlays, metrics, all_errors, track_counts = [], [], [], collections.Counter()
    example = None
    tracked_mode_s = set()
    track_temp = output / "v6-mode-s-tracks.csv.partial"
    with track_temp.open("w", newline="") as stream, ProcessPoolExecutor(
            max_workers=args.workers) as pool:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for validate, cohort in ((True, validation_ids), (False, mode_s_ids)):
            futures = [pool.submit(run_aircraft, (icao, bursts[icao], paths, radars, validate))
                       for icao in cohort]
            for future in as_completed(futures):
                icao, tracks, flight_metrics, errors, counts, reference = future.result()
                metrics.extend(flight_metrics)
                all_errors.extend(errors)
                track_counts.update(counts)
                matched_flights = {row["flight_id"] for row in flight_metrics if row["n_matches"]}
                for flight_id, points in tracks:
                    if not points:
                        continue
                    stride = max(1, len(points) // 250)
                    if validate:
                        if flight_id in matched_flights and (
                                example is None or len(points) > len(example[1])):
                            example = flight_id, points, reference
                    else:
                        overlays.append((icao, "mode_s", [(p["lat"], p["lon"])
                                                    for p in points[::stride]]))
                    if not validate:
                        tracked_mode_s.add(icao)
                        for point in points:
                            row = {**point, "icao24": icao}
                            row["utc"] = dt.datetime.fromtimestamp(
                                row["utc"], dt.timezone.utc).isoformat(timespec="milliseconds")
                            row["alt_ft"] = "" if row["alt_ft"] is None else round(row["alt_ft"], 1)
                            for key in ("lat", "lon"):
                                row[key] = round(row[key], 7)
                            for key in ("speed_kt", "track_deg", "pos_sigma_km"):
                                row[key] = round(row[key], 3)
                            writer.writerow(row)
    if example is not None:
        flight_id, points, reference = example
        icao = flight_id.split("-", 1)[0]
        stride = max(1, len(points) // 500)
        overlays.append((icao, "blind", [(p["lat"], p["lon"]) for p in points[::stride]]))
        line, previous = [], None
        for utc, lat, lon in reference:
            if not points[0]["utc"] <= utc <= points[-1]["utc"]:
                continue
            if previous is not None and utc - previous > REFERENCE_MAX_GAP:
                overlays.append((icao, "truth", line))
                line = []
            line.append((lat, lon))
            previous = utc
        overlays.append((icao, "truth", line))
    aggregate = error_statistics(all_errors)
    metrics.append(dict(icao24="ALL", flight_id="ALL",
                        status="ok" if all_errors else "no_validated_positions", **aggregate))
    stats_temp = output / "v6-blind-validation.csv.partial"
    with stats_temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=stats_columns)
        writer.writeheader()
        writer.writerows(sorted(metrics, key=lambda row: row["flight_id"]))
    summary = dict(elapsed_s=round(time.monotonic() - started, 2),
                   calibration_aircraft=calibration_ids, blind_aircraft=len(validation_ids),
                   mode_s_candidates=len(mode_s_ids), mode_s_tracked=len(tracked_mode_s),
                   validation=aggregate, angular_calibration=noise,
                   raw_formats=dict(raw_counts), tracking=dict(track_counts),
                   blind_map_example=example[0] if example else None,
                   altitude_datum="barometric pressure altitude; not geometric ENU Up",
                   bds60="not a ground velocity; never assimilated",
                   scope="subset" if args.max_segments else "archived_only"
                   if args.no_active else "full published capture cutoff",
                   schemas=schemas)
    svg_temp = output / "v6-tracks-map.svg.partial"
    write_map(svg_temp, radars, overlays,
              f"Blind samples: {aggregate['n_matches']}; Mode-S aircraft: {len(tracked_mode_s)}; "
              f"example: {summary['blind_map_example']}; heights: pressure altitude, not geometric 3D")
    for temporary, name in ((track_temp, "v6-mode-s-tracks.csv"),
                            (stats_temp, "v6-blind-validation.csv"),
                            (svg_temp, "v6-tracks-map.svg")):
        os.replace(temporary, output / name)
    (output / "v6-run-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("V6 SUMMARY", json.dumps({key: value for key, value in summary.items()
                                   if key not in ("schemas", "calibration_aircraft")},
                                  allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
