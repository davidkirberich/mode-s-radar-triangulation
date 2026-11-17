#!/usr/bin/env python3
"""Bounded Beast byte-stream capture and timestamped aircraft snapshots."""

import argparse
import json
import math
import os
import shutil
import signal
import socket
import sqlite3
import sys
import time
from pathlib import Path

MIN_FREE_BYTES = 5 * 1024**3
SNAPSHOT_FIELDS = (
    "hex", "flight", "lat", "lon", "seen_pos", "alt_baro", "gs", "track",
)


def create_database(path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA synchronous=FULL;
        CREATE TABLE run (
            status TEXT NOT NULL, started_ns INTEGER NOT NULL,
            ended_ns INTEGER, reason TEXT, host TEXT NOT NULL, port INTEGER NOT NULL,
            duration_s REAL NOT NULL, raw_path TEXT NOT NULL,
            snapshot_path TEXT NOT NULL
        );
        CREATE TABLE chunks (
            offset INTEGER PRIMARY KEY, size INTEGER NOT NULL,
            realtime_ns INTEGER NOT NULL, monotonic_ns INTEGER NOT NULL
        );
        CREATE TABLE snapshots (
            realtime_ns INTEGER NOT NULL, monotonic_ns INTEGER NOT NULL,
            snapshot_utc REAL NOT NULL, aircraft_json TEXT NOT NULL
        );
    """)
    return db


def read_snapshot(path):
    data = json.loads(path.read_text())
    now = data["now"]
    if not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("Invalid snapshot timestamp")
    if not -2 <= time.time() - now <= 5:
        raise ValueError("Aircraft snapshot is stale or in the future")
    aircraft = data["aircraft"]
    if not isinstance(aircraft, list):
        raise ValueError("Invalid aircraft snapshot")
    selected = [{k: a[k] for k in SNAPSHOT_FIELDS if k in a}
                for a in aircraft if isinstance(a, dict)]
    return now, json.dumps(selected, separators=(",", ":"), allow_nan=False)


def capture(host, port, raw_path, db_path, snapshot, duration, stop):
    raw_fd = os.open(raw_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    db = None
    status, reason = "incomplete", "capture did not finish"
    try:
        with os.fdopen(raw_fd, "wb", buffering=0) as raw:
            db = create_database(db_path)
            db.execute("INSERT INTO run VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?)",
                       ("recording", time.time_ns(), host, port, duration, str(raw_path),
                        str(snapshot)))
            db.commit()
            deadline = time.monotonic() + duration
            next_snapshot = next_sync = time.monotonic()
            offset = 0
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.settimeout(0.25)
                while not stop():
                    now = time.monotonic()
                    if now >= deadline:
                        status, reason = "completed", "duration elapsed"
                        break
                    if now >= next_snapshot:
                        if shutil.disk_usage(raw_path.parent).free < MIN_FREE_BYTES:
                            raise OSError("Free storage below 5 GiB")
                        snapshot_utc, aircraft = read_snapshot(snapshot)
                        db.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                                   (time.time_ns(), time.monotonic_ns(), snapshot_utc, aircraft))
                        next_snapshot = now + 1
                    try:
                        chunk = sock.recv(65536)
                        realtime_ns = time.time_ns()
                        monotonic_ns = time.monotonic_ns()
                    except socket.timeout:
                        chunk = None
                    if chunk == b"":
                        raise ConnectionError("Beast stream closed")
                    if chunk:
                        written = raw.write(chunk)
                        if written != len(chunk):
                            raise OSError(f"Short raw write: {written}/{len(chunk)}")
                        db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                   (offset, len(chunk), realtime_ns, monotonic_ns))
                        offset += len(chunk)
                    if time.monotonic() >= next_sync:
                        os.fsync(raw.fileno())
                        db.commit()
                        next_sync = time.monotonic() + 1
            if stop():
                status, reason = "stopped", "received stop signal"
            if offset == 0:
                raise ValueError("No Beast data received")
            os.fsync(raw.fileno())
            db.commit()
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
        status, reason = "incomplete", str(error)
        raise
    finally:
        if db is not None:
            try:
                db.execute("UPDATE run SET status=?, ended_ns=?, reason=?",
                           (status, time.time_ns(), reason))
                db.commit()
            finally:
                db.close()
        print(f"Capture {status}: {reason}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="New raw file")
    parser.add_argument("--db", type=Path, required=True, help="New SQLite file")
    parser.add_argument("--duration", type=float, required=True, help="Seconds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30005)
    parser.add_argument("--snapshot", type=Path, default=Path("/dev/shm/adsb-live-v2/aircraft.json"))
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0 or not 1 <= args.port <= 65535:
        parser.error("duration and port must be positive and valid")
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        capture(args.host, args.port, args.raw, args.db, args.snapshot,
                args.duration, lambda: stopped)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
