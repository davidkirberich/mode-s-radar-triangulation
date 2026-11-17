#!/usr/bin/env python3
"""Capture hourly Beast segments without decoding the live stream."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from beast_capture import MIN_FREE_BYTES, read_snapshot

COORDINATES = {"lat": 51.14597, "lon": 6.36358}


def receiver_context():
    pid = int(subprocess.check_output(
        ["systemctl", "show", "adsb-receiver-v2", "--property=MainPID", "--value"],
        text=True,
    ).strip())
    if pid <= 0:
        raise RuntimeError("Receiver service is not running")
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    argv = [os.fsdecode(value) for value in cmd]
    binary = Path(f"/proc/{pid}/exe")
    with binary.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"pid": pid, "argv": argv, "binary_sha256": digest,
            "receiver_coordinates": COORDINATES,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


class Segment:
    def __init__(self, directory, context, now_ns):
        self.started_ns = now_ns
        self.hour_ns = now_ns // 3_600_000_000_000 * 3_600_000_000_000
        stamp = datetime.fromtimestamp(now_ns / 1e9, timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = directory / f"capture-{stamp}-{os.getpid()}-{now_ns % 1_000_000_000:09d}"
        self.raw_path = base.with_suffix(".raw")
        self.meta_path = base.with_suffix(".jsonl")
        self.raw = self.raw_path.open("xb", buffering=0)
        try:
            self.meta = self.meta_path.open("xb", buffering=0)
        except OSError:
            self.raw.close()
            raise
        self.offset = 0
        self.snapshots = 0
        self.problem = None
        self.write_meta({"type": "start", "realtime_ns": now_ns,
                         "monotonic_ns": time.monotonic_ns(),
                         "hour_start_ns": self.hour_ns, "context": context})

    def write_meta(self, record):
        data = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if self.meta.write(data) != len(data):
            raise OSError("Short metadata write")

    def write_chunk(self, chunk, realtime_ns, monotonic_ns):
        if self.raw.write(chunk) != len(chunk):
            raise OSError("Short raw write")
        self.write_meta({"type": "chunk", "offset": self.offset, "size": len(chunk),
                         "realtime_ns": realtime_ns, "monotonic_ns": monotonic_ns})
        self.offset += len(chunk)

    def write_snapshot(self, snapshot_utc, aircraft):
        self.write_meta({"type": "snapshot", "realtime_ns": time.time_ns(),
                         "monotonic_ns": time.monotonic_ns(),
                         "snapshot_utc": snapshot_utc, "aircraft": json.loads(aircraft)})
        self.snapshots += 1

    def sync(self):
        os.fsync(self.raw.fileno())
        os.fsync(self.meta.fileno())

    def close(self, status, reason):
        try:
            self.sync()
            self.write_meta({"type": "end", "status": status, "reason": reason,
                             "realtime_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                             "raw_bytes": self.offset, "snapshots": self.snapshots})
            os.fsync(self.meta.fileno())
        finally:
            self.raw.close()
            self.meta.close()
        with self.meta_path.with_suffix(".closed").open("xb") as marker:
            marker.write(b"closed\n")
            marker.flush()
            os.fsync(marker.fileno())
        print(f"SEGMENT {self.raw_path} {status} {reason}", flush=True)


def record(directory, end_utc, stop, host="127.0.0.1", port=30005,
           snapshot=Path("/dev/shm/adsb-live-v2/aircraft.json")):
    context = receiver_context()
    segment = None
    status, reason = "INCOMPLETE", "interrupted"
    last_sync = last_snapshot = 0.0
    last_chunk = time.monotonic()
    previous_real = previous_mono = None

    def rotate(real):
        nonlocal segment
        hour = real // 3_600_000_000_000 * 3_600_000_000_000
        if segment is not None and hour != segment.hour_ns:
            full = (segment.started_ns - segment.hour_ns < 2_000_000_000
                    and real - (segment.hour_ns + 3_600_000_000_000) < 2_000_000_000
                    and segment.offset > 0 and segment.snapshots > 0
                    and segment.problem is None)
            segment.close("COMPLETE" if full else "INCOMPLETE",
                          "UTC hour boundary" if full else segment.problem or "partial UTC hour")
            segment = None
        if segment is None:
            segment = Segment(directory, context, real)

    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.settimeout(0.25)
            while not stop():
                real = time.time_ns()
                mono = time.monotonic_ns()
                if real >= end_utc * 1_000_000_000:
                    reason = "seven-day deadline reached"
                    break
                if previous_real is not None and abs((real - previous_real) - (mono - previous_mono)) > 2_000_000_000:
                    raise RuntimeError("Realtime clock changed by more than two seconds")
                previous_real, previous_mono = real, mono
                rotate(real)
                now = time.monotonic()
                if now - last_chunk > 10:
                    segment.problem = "stream gap over 10 seconds"
                if now - last_snapshot >= 1:
                    if last_snapshot and now - last_snapshot > 10:
                        segment.problem = "snapshot gap over 10 seconds"
                    if shutil.disk_usage(directory).free < MIN_FREE_BYTES:
                        raise OSError("Free storage below 5 GiB")
                    snapshot_utc, aircraft = read_snapshot(snapshot)
                    segment.write_snapshot(snapshot_utc, aircraft)
                    last_snapshot = now
                try:
                    chunk = sock.recv(65536)
                    real_recv, mono_recv = time.time_ns(), time.monotonic_ns()
                except socket.timeout:
                    chunk = None
                if chunk == b"":
                    raise ConnectionError("Beast stream disconnected")
                if chunk:
                    rotate(real_recv)
                    if time.monotonic() - last_chunk > 10:
                        segment.problem = "stream gap over 10 seconds"
                    segment.write_chunk(chunk, real_recv, mono_recv)
                    last_chunk = time.monotonic()
                if now - last_sync >= 1:
                    segment.sync()
                    last_sync = now
            else:
                reason = "received stop signal"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError) as error:
        reason = str(error)
        raise
    finally:
        if segment is not None:
            segment.close(status, reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--end-utc", type=int, required=True, help="Unix seconds")
    args = parser.parse_args()
    os.umask(0o077)
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        record(args.directory, args.end_utc, lambda: stopped)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
