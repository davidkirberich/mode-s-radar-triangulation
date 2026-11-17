#!/usr/bin/env python3
"""Compress and analyze closed Beast segments; never delete capture files."""

import argparse
import bisect
import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from beast_inspect import frames, interrogator


def digest_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").digest()


def compress(raw):
    output = Path(str(raw) + ".zst")
    if output.exists():
        return
    temp = Path(str(output) + f".{os.getpid()}.{time.time_ns()}.partial")
    with temp.open("xb") as f:
        subprocess.run(["zstd", "-q", "-T1", "-c", str(raw)], stdout=f, check=True)
        f.flush()
        os.fsync(f.fileno())
    with subprocess.Popen(["zstd", "-q", "-d", "-c", str(temp)],
                          stdout=subprocess.PIPE) as proc:
        digest = hashlib.file_digest(proc.stdout, "sha256").digest()
        proc.stdout.close()
        if proc.wait() or digest != digest_file(raw):
            raise ValueError(f"Compressed data failed verification: {temp}")
    os.link(temp, output)
    temp.unlink()


def read_metadata(meta):
    chunks, snapshots = [], []
    start = end = None
    with meta.open() as f:
        for line in f:
            row = json.loads(line)
            if row["type"] == "start":
                start = row
            elif row["type"] == "chunk":
                chunks.append(row)
            elif row["type"] == "snapshot":
                snapshots.append(row)
            elif row["type"] == "end":
                end = row
    if not start or not end:
        raise ValueError(f"Missing start or end metadata: {meta}")
    ends = []
    for row in chunks:
        if row["offset"] != (ends[-1] if ends else 0) or row["size"] <= 0:
            raise ValueError(f"Noncontiguous chunks: {meta}")
        ends.append(row["offset"] + row["size"])
    if not ends or end["raw_bytes"] != ends[-1]:
        raise ValueError(f"Invalid raw byte count: {meta}")
    return start, end, chunks, ends, snapshots


@contextlib.contextmanager
def decompress(path):
    proc = subprocess.Popen(["zstd", "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE)
    try:
        yield proc.stdout
        proc.stdout.close()
        if proc.wait():
            raise ValueError(f"Decompression failed: {path}")
    finally:
        proc.stdout.close()
        proc.wait()


def safe_to_delete(hour, eligible):
    return all(len(eligible.get(h, [])) == 1 and eligible[h][0][1]
               for h in (hour - 1, hour, hour + 1))


def analyze(raw, meta):
    output = raw.with_suffix(".sqlite")
    if output.exists():
        return
    start, end, chunks, ends, snapshots = read_metadata(meta)
    if raw.stat().st_size != ends[-1]:
        raise ValueError(f"Raw file differs from chunk index: {raw}")
    temp = Path(str(output) + f".{os.getpid()}.{time.time_ns()}.partial")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    db = sqlite3.connect(temp)
    try:
        db.executescript("""
            PRAGMA synchronous=FULL;
            CREATE TABLE run (metadata_json TEXT NOT NULL, status TEXT NOT NULL,
                boundary_complete INTEGER NOT NULL);
            CREATE TABLE df11_events (
                raw_offset INTEGER NOT NULL, recv_ns INTEGER NOT NULL,
                monotonic_ns INTEGER NOT NULL, mlat_ticks INTEGER NOT NULL,
                icao24 TEXT NOT NULL, code_type TEXT NOT NULL, code_val INTEGER NOT NULL,
                lat REAL, lon REAL, alt_baro, pos_age_ms INTEGER, gs REAL, track REAL,
                session_id TEXT NOT NULL
            );
            CREATE TABLE clock_anchors (monotonic_ns INTEGER NOT NULL,
                mlat_ticks INTEGER NOT NULL, raw_offset INTEGER NOT NULL);
            CREATE TABLE clock_fit (points INTEGER NOT NULL, ticks_per_second REAL,
                max_abs_residual_ticks REAL, discontinuities INTEGER NOT NULL);
        """)
        db.execute("INSERT INTO run VALUES (?, ?, 0)",
                   (json.dumps(start, separators=(",", ":")), end["status"]))
        snap_times = [s["monotonic_ns"] for s in snapshots]
        position_cache = {}
        batch, anchors = [], []
        first_frame = last_frame_end = None
        last_anchor = None
        with decompress(Path(str(raw) + ".zst")) as f:
            for offset, frame_end, kind, body in frames(f):
                if first_frame is None:
                    first_frame = offset
                last_frame_end = frame_end
                if kind not in (0x32, 0x33):
                    continue
                chunk = chunks[bisect.bisect_left(ends, frame_end)]
                mono = chunk["monotonic_ns"]
                ticks = int.from_bytes(body[:6], "big")
                if (ticks and mono - start["monotonic_ns"] >= 10_000_000_000
                        and (last_anchor is None or mono - last_anchor >= 60_000_000_000)):
                    anchors.append((mono, ticks, offset))
                    last_anchor = mono
                payload = body[7:]
                code = interrogator(payload)
                if code is None:
                    continue
                address = payload[1:4].hex().upper()
                index = bisect.bisect_right(snap_times, mono) - 1
                pos = None
                if index >= 0:
                    if index not in position_cache:
                        position_cache = {index: {str(a.get("hex", "")).lstrip("~").upper(): a
                                                  for a in snapshots[index]["aircraft"]}}
                    pos = position_cache[index].get(address)
                age = None
                if pos and isinstance(pos.get("seen_pos"), (int, float)):
                    age = round((chunk["realtime_ns"] / 1e9 - snapshots[index]["snapshot_utc"]
                                 + pos["seen_pos"]) * 1000)
                batch.append((offset, chunk["realtime_ns"], mono, ticks, address,
                              code[0], code[1], pos.get("lat") if pos else None,
                              pos.get("lon") if pos else None,
                              pos.get("alt_baro") if pos else None, age,
                              pos.get("gs") if pos else None,
                              pos.get("track") if pos else None, raw.stem))
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO df11_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        db.executemany("INSERT INTO df11_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        db.execute("UPDATE run SET boundary_complete=?",
                   (int(first_frame == 0 and last_frame_end == ends[-1]),))
        db.executemany("INSERT INTO clock_anchors VALUES (?,?,?)", anchors)
        discontinuities = sum(b[1] + 12_000_000 < a[1] for a, b in zip(anchors, anchors[1:]))
        rate = residual = None
        if len(anchors) >= 2 and not discontinuities:
            x = [(a[0] - anchors[0][0]) / 1e9 for a in anchors]
            y = [a[1] - anchors[0][1] for a in anchors]
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            variance = sum((v - xm) ** 2 for v in x)
            if variance:
                rate = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / variance
                residual = max(abs(b - (ym + rate * (a - xm))) for a, b in zip(x, y))
        db.execute("INSERT INTO clock_fit VALUES (?,?,?,?)",
                   (len(anchors), rate, residual, discontinuities))
        db.execute("CREATE INDEX df11_code_time ON df11_events(code_type,code_val,monotonic_ns)")
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite integrity check failed: {temp}")
        count = db.execute("SELECT COUNT(*) FROM df11_events").fetchone()[0]
    finally:
        db.close()
    os.link(temp, output)
    temp.unlink()
    print(f"ANALYSIS {raw.name} status={end['status']} df11={count} "
          f"anchors={len(anchors)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--end-utc", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    with (args.directory / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = False
        for meta in sorted(args.directory.glob("capture-*.jsonl")):
            if not meta.with_suffix(".closed").exists() and time.time() - meta.stat().st_mtime > 120:
                print(f"INCOMPLETE: missing close marker for {meta}; retained", flush=True)
        for marker in sorted(args.directory.glob("capture-*.closed")):
            raw, meta = marker.with_suffix(".raw"), marker.with_suffix(".jsonl")
            try:
                compress(raw)
                analyze(raw, meta)
                compress(raw.with_suffix(".sqlite"))
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError,
                    subprocess.SubprocessError) as error:
                failures = True
                print(f"ERROR: processing {raw}: {error}", file=sys.stderr, flush=True)
        eligible = {}
        uncertain = failures or any(not raw.with_suffix(".jsonl").exists()
                                    for raw in args.directory.glob("capture-*.raw"))
        for marker in sorted(args.directory.glob("capture-*.closed")):
            raw, meta = marker.with_suffix(".raw"), marker.with_suffix(".jsonl")
            try:
                start, end, _, _, _ = read_metadata(meta)
                hour = start["hour_start_ns"] // 3_600_000_000_000
                with sqlite3.connect(f"{raw.with_suffix('.sqlite').resolve().as_uri()}?mode=ro",
                                     uri=True) as db:
                    boundary_complete, = db.execute("SELECT boundary_complete FROM run").fetchone()
                eligible.setdefault(hour, []).append(
                    (raw, end["status"] == "COMPLETE" and boundary_complete))
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
                uncertain = failures = True
                print(f"ERROR: eligibility {raw}: {error}", file=sys.stderr, flush=True)
        for meta in sorted(args.directory.glob("capture-*.jsonl")):
            if meta.with_suffix(".closed").exists():
                continue
            try:
                with meta.open() as f:
                    first = json.loads(f.readline())
                hour = first["hour_start_ns"] // 3_600_000_000_000
                eligible.setdefault(hour, []).append((meta, False))
            except (OSError, ValueError, KeyError, TypeError):
                uncertain = True
        for hour, segments in sorted(eligible.items()):
            if time.time() < (hour + 1) * 3600 + 48 * 3600:
                continue
            safe = not uncertain and safe_to_delete(hour, eligible)
            for raw, _ in segments:
                print(f"DRY-RUN {'would delete' if safe else 'protected'}: {raw}", flush=True)
        if failures:
            raise RuntimeError("One or more segments could not be processed")
        if args.end_utc and time.time() >= args.end_utc:
            subprocess.run(["systemctl", "disable", "--now", "adsb-beast-archive.timer"], check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
