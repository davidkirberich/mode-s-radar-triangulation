"""Separate experiment runs and preserve their executed source files."""
import datetime as dt
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path


def add_common_options(parser, archive, output, rx_lat, rx_lon):
    parser.add_argument("--archive", type=Path, default=archive)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--rx-lat", type=float, default=rx_lat)
    parser.add_argument("--rx-lon", type=float, default=rx_lon)


def configure_receiver(parser, args):
    if not math.isfinite(args.rx_lat) or not -90 <= args.rx_lat <= 90:
        parser.error("--rx-lat must be finite and between -90 and 90")
    if not math.isfinite(args.rx_lon) or not -180 <= args.rx_lon <= 180:
        parser.error("--rx-lon must be finite and between -180 and 180")
    os.environ["ADSB_RX_LAT"] = str(args.rx_lat)
    os.environ["ADSB_RX_LON"] = str(args.rx_lon)


def parse_common_options(archive, output, rx_lat, rx_lon):
    parser = argparse.ArgumentParser()
    add_common_options(parser, archive, output, rx_lat, rx_lon)
    args = parser.parse_args()
    configure_receiver(parser, args)
    return args


def create_run_directory(version, script, output_root):
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = Path(output_root) / "runs" / f"{timestamp}-{Path(script).stem}"
    code = directory / "code"
    code.mkdir(parents=True, exist_ok=False)
    sources = {Path(script).resolve(), Path(__file__).resolve()}
    for name in ("beast_inspect", "radar_triangulation_v4",
                 "radar_triangulation_v4_copilot", "mode_s_tracker_v6"):
        module = sys.modules.get(name)
        if module is not None:
            sources.add(Path(module.__file__).resolve())
    for source in sources:
        shutil.copy2(source, code / source.name)
    (directory / "run.json").write_text(json.dumps(
        dict(version=version, created_utc=timestamp, script=Path(script).name,
             command=[sys.executable, *sys.argv]), indent=2), encoding="utf-8")
    return directory
