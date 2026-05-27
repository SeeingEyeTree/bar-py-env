#!/usr/bin/env python3
"""Watch a bar-py-env replay (.sdfz) in your installed BAR engine.

Spring.exe by default can only see files inside its own engine directory.
Replays reference the map / game archives that live one level up under the BAR
data dir, so we set SPRING_DATADIR before invoking the engine.

Usage:
    python scripts/play_replay.py                  # plays the most recent demo
    python scripts/play_replay.py <path-to-.sdfz>  # plays a specific demo
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Make `bar_env` importable when this script is run directly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import RUN_DIR, default_bar_data_dir  # noqa: E402


def find_most_recent_demo() -> Path | None:
    """Return the newest .sdfz under our run dirs."""
    if not RUN_DIR.exists():
        return None
    demos = list(RUN_DIR.glob("*/demos/*.sdfz"))
    if not demos:
        # Some Recoil setups write demos straight into the BAR install. Check
        # there too as a fallback.
        bar_data = default_bar_data_dir()
        if (bar_data / "demos").is_dir():
            demos = list((bar_data / "demos").glob("*.sdfz"))
    if not demos:
        return None
    return max(demos, key=lambda p: p.stat().st_mtime)


def find_spring_exe(bar_data: Path) -> Path:
    """Find the non-headless spring.exe inside the BAR install."""
    candidates = list((bar_data / "engine").glob("*/spring.exe"))
    if not candidates:
        raise FileNotFoundError(
            f"No spring.exe found under {bar_data}/engine/. "
            f"Is your BAR install at {bar_data}?"
        )
    # Prefer the newest engine if there are multiple.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    if len(sys.argv) > 1:
        demo = Path(sys.argv[1])
        if not demo.exists():
            sys.stderr.write(f"Replay not found: {demo}\n")
            return 1
    else:
        demo = find_most_recent_demo()
        if demo is None:
            sys.stderr.write(
                "No .sdfz files found under "
                f"{RUN_DIR} or in your BAR install's demos/.\n"
                "Pass the path to the .sdfz as an argument.\n"
            )
            return 1
        print(f"Most recent demo: {demo}")

    bar_data = default_bar_data_dir()
    if not bar_data.exists():
        sys.stderr.write(f"BAR data dir not found at {bar_data}\n")
        return 1

    spring = find_spring_exe(bar_data)
    print(f"Using engine    : {spring}")
    print(f"SPRING_DATADIR  : {bar_data}")
    print()

    env = os.environ.copy()
    env["SPRING_DATADIR"] = str(bar_data)
    # spring.exe blocks until the user closes the window.
    return subprocess.call([str(spring), str(demo)], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
