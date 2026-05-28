#!/usr/bin/env python3
"""Download the Recoil engine matching BAR's current launcher-config.

Pulls the Windows 7z release from the RecoilEngine GitHub releases and
extracts it under ./engine/recoil_<version>/. Requires 7-Zip on PATH.

Usage:
    python scripts/fetch_engine.py [--package-id manual-win]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Make `bar_env` importable when this script is run directly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import (  # noqa: E402
    DEFAULT_LAUNCHER_PACKAGE_ID,
    ENGINE_DIR,
    LAUNCHER_CONFIG_URL,
)


def fetch_launcher_config(url: str = LAUNCHER_CONFIG_URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "bar-py-env/0.0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def find_engine_resource(config: dict, package_id: str) -> dict:
    for setup in config.get("setups", []):
        pkg = setup.get("package", {})
        if pkg.get("id") != package_id:
            continue
        for resource in setup.get("downloads", {}).get("resources", []):
            dest = resource.get("destination", "")
            if dest.startswith("engine/"):
                return resource
    raise RuntimeError(
        f"No engine resource found for package id '{package_id}' in launcher-config"
    )


def find_seven_zip() -> str:
    '''
    for name in ("7z", "7za", "7z.exe", "7za.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "Could not find 7z/7za on PATH. Install 7-Zip from https://www.7-zip.org/ "
        "and ensure 7z.exe is on PATH."
    )
    '''
    return "C:/Program Files/7-Zip/7z.exe"


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    print(f"  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "bar-py-env/0.0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        read = 0
        chunk = 1 << 16
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100.0 * read / total
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="")
        print()


def extract_7z(archive: Path, dest_dir: Path, sevenzip: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {dest_dir}")
    proc = subprocess.run(
        [sevenzip, "x", str(archive), f"-o{dest_dir}", "-y"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"7z extraction failed (exit {proc.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-id",
        default=DEFAULT_LAUNCHER_PACKAGE_ID,
        help="launcher-config setup package id (default: manual-win)",
    )
    args = parser.parse_args()

    print(f"Reading launcher config from {LAUNCHER_CONFIG_URL}")
    config = fetch_launcher_config()

    print(f"Looking up engine resource for package '{args.package_id}'")
    resource = find_engine_resource(config, args.package_id)
    url: str = resource["url"]
    relative_dest: str = resource["destination"]
    engine_subdir = ENGINE_DIR / Path(relative_dest).name
    print(f"  engine version : {Path(relative_dest).name}")
    print(f"  archive url    : {url}")
    print(f"  target dir     : {engine_subdir}")

    # Already fetched? Skip the network round trip.
    headless = list(engine_subdir.glob("**/spring-headless.exe")) + list(
        engine_subdir.glob("**/spring-headless")
    )
    if headless:
        print(f"\nAlready installed: {headless[0]}")
        return 0

    sevenzip = find_seven_zip()
    print(f"Using 7z: {sevenzip}")

    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "engine.7z"
        download(url, archive)
        extract_7z(archive, engine_subdir, sevenzip)

    # Verify
    headless = list(engine_subdir.glob("**/spring-headless.exe")) + list(
        engine_subdir.glob("**/spring-headless")
    )
    if not headless:
        sys.stderr.write(
            "\nERROR: extraction completed but no spring-headless binary was found.\n"
            f"Inspect {engine_subdir} manually.\n"
        )
        return 1

    print(f"\nspring-headless installed: {headless[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
