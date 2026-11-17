#!/usr/bin/env python3
"""Inspect captured Beast frames without changing the raw capture."""

import argparse
import bisect
import json
import sqlite3
from collections import Counter
from pathlib import Path

LENGTHS = {0x31: 9, 0x32: 14, 0x33: 21}


def frames(raw):
    offset = 0
    buf = bytearray()
    while True:
        block = raw.read(65536)
        buf.extend(block)
        pos = 0
        while pos + 1 < len(buf):
            if buf[pos] != 0x1a or buf[pos + 1] not in LENGTHS:
                pos += 1
                continue
            kind = buf[pos + 1]
            i = pos + 2
            body = bytearray()
            invalid = False
            while len(body) < LENGTHS[kind] and i < len(buf):
                if buf[i] == 0x1a:
                    if i + 1 >= len(buf):
                        break
                    if buf[i + 1] != 0x1a:
                        invalid = True
                        break
                    i += 1
                body.append(buf[i])
                i += 1
            if invalid:
                pos = i
                continue
            if len(body) < LENGTHS[kind]:
                break
            yield offset + pos, offset + i, kind, bytes(body)
            pos = i
        del buf[:pos]
        offset += pos
        if not block:
            break


def interrogator(payload):
    if len(payload) != 7 or payload[0] >> 3 != 11:
        return None
    remainder = int.from_bytes(payload, "big")
    for shift in range(31, -1, -1):
        if remainder & (1 << (shift + 24)):
            remainder ^= 0xFFF409 << shift
    remainder &= 0xffffff
    if remainder >> 7:
        return None
    label, code = remainder >> 4, remainder & 15
    if label == 0:
        return "II", code
    if 1 <= label <= 4 and (label != 1 or code != 0):
        return "SI", (label - 1) * 16 + code
    return None


def inspect(raw_path, db_path):
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        status, = db.execute("SELECT status FROM run").fetchone()
        chunks = db.execute(
            "SELECT offset, size, monotonic_ns FROM chunks ORDER BY offset"
        ).fetchall()
    finally:
        db.close()
    if not chunks or chunks[0][0] != 0:
        raise ValueError("No complete chunk index")
    ends = []
    for offset, size, _ in chunks:
        if offset != (ends[-1] if ends else 0) or size <= 0:
            raise ValueError("Chunk index has gaps")
        ends.append(offset + size)
    if raw_path.stat().st_size != ends[-1]:
        raise ValueError("Raw file size differs from chunk index")
    counts = Counter()
    clock_points = {}
    first_mono = None
    resets = 0
    previous_tick = None
    with raw_path.open("rb") as raw:
        for _, end, kind, body in frames(raw):
            counts[f"beast_{kind:02x}"] += 1
            tick = int.from_bytes(body[:6], "big")
            idx = bisect.bisect_left(ends, end)
            mono = chunks[idx][2]
            if kind in (0x32, 0x33) and tick:
                if first_mono is None:
                    first_mono = mono
                if mono - first_mono >= 10_000_000_000:
                    if previous_tick is not None and tick + 12_000_000 < previous_tick:
                        resets += 1
                    previous_tick = tick
                    second = (mono - first_mono) // 1_000_000_000
                    clock_points.setdefault(second, (mono, tick))
            if kind == 0x32:
                code = interrogator(body[7:])
                if code:
                    counts[f"{code[0]}_{code[1]}"] += 1
    result = {"status": status, "raw_bytes": ends[-1], "chunks": len(chunks),
              "counts": dict(counts), "tick_backward_jumps_over_1s": resets}
    points = list(clock_points.values())
    if len(points) >= 2 and not resets:
        first_mono, first_tick = points[0]
        x = [(mono - first_mono) / 1e9 for mono, _ in points]
        y = [tick - first_tick for _, tick in points]
        x_avg, y_avg = sum(x) / len(x), sum(y) / len(y)
        variance = sum((value - x_avg) ** 2 for value in x)
        if variance:
            result["clock_fit_span_s"] = x[-1] - x[0]
            result["clock_fit_points"] = len(points)
            result["ticks_per_monotonic_second_approx"] = sum(
                (a - x_avg) * (b - y_avg) for a, b in zip(x, y)
            ) / variance
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.raw, args.db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
