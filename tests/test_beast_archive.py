import json
import sqlite3
import sys
import time

import pytest

import beast_archive


@pytest.mark.parametrize("middle_status, expected", [
    ("COMPLETE", "would delete"),
    ("INCOMPLETE", "protected"),
])
def test_archive_schema_and_eligibility(tmp_path, monkeypatch, capsys, middle_status, expected):
    monkeypatch.setattr(beast_archive, "decompress", lambda path: path.with_suffix("").open("rb"))
    monkeypatch.setattr(beast_archive, "interrogator", lambda payload: ("II", 1))
    monkeypatch.setattr(beast_archive, "compress", lambda path: None)
    hour = int(time.time() // 3600) - 100
    raw_bytes = b"\x1a\x32" + b"\x00" * 14
    for delta in (-1, 0, 1):
        raw = tmp_path / f"capture-{hour + delta}.raw"
        meta = raw.with_suffix(".jsonl")
        raw.write_bytes(raw_bytes)
        records = [
            {"type": "start", "hour_start_ns": (hour + delta) * 3_600_000_000_000,
             "monotonic_ns": 0},
            {"type": "snapshot", "monotonic_ns": 1, "snapshot_utc": 1,
             "aircraft": [{"hex": "000000", "lat": 51.0, "lon": 6.0, "seen_pos": 0}]},
            {"type": "chunk", "offset": 0, "size": len(raw_bytes),
             "monotonic_ns": 20_000_000_000, "realtime_ns": 1_000_000_000},
            {"type": "end", "raw_bytes": len(raw_bytes),
             "status": middle_status if delta == 0 else "COMPLETE"},
        ]
        meta.write_text("".join(json.dumps(record) + "\n" for record in records))
        raw.with_suffix(".closed").touch()
        beast_archive.analyze(raw, meta)
    with sqlite3.connect(tmp_path / f"capture-{hour}.sqlite") as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"run", "df11_events", "clock_anchors", "clock_fit"}
        assert db.execute("SELECT lat, lon FROM df11_events").fetchone() == (51.0, 6.0)
    monkeypatch.setattr(sys, "argv", ["beast_archive.py", "--directory", str(tmp_path)])
    beast_archive.main()
    assert f"DRY-RUN {expected}: {tmp_path / f'capture-{hour}.raw'}" in capsys.readouterr().out
    assert (tmp_path / f"capture-{hour}.raw").exists()
