#!/usr/bin/env python3
"""Batch-extract BAR replay files to compact JSONL training data.

Scans --in-dir for .sdfz files and skips those that can't be played back:
  - Engine version mismatch: the demo was recorded with a different Spring
    engine binary than what's installed (always skipped — downloading game
    archives cannot fix this).
  - Game content version mismatch: the demo's BAR game version isn't
    installed locally (skipped unless --allow-download is set, which fetches
    the required archive via pr-downloader before each extraction).
  - Already extracted: an output file already exists in --out-dir.

Usage:
    python scripts/batch_extract_replays.py
    python scripts/batch_extract_replays.py --in-dir D:/BAR_Replays/Human_Pretraining --out-dir D:/BAR_Replays/training_data
    python scripts/batch_extract_replays.py --allow-download
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import default_bar_data_dir          # noqa: E402
from bar_env.headless import detect_engine_version       # noqa: E402
from bar_env.replay import (                             # noqa: E402
    ReplayExtractorConfig,
    detect_installed_game_version,
    extract,
    get_demo_engine_version,
    get_demo_game_type,
)

DEFAULT_IN_DIR  = Path("D:/BAR_Replays/Human_Pretraining")
DEFAULT_OUT_DIR = Path("D:/BAR_Replays/training_data")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--in-dir",  type=Path, default=DEFAULT_IN_DIR,
                   help=f"Folder containing .sdfz replays (default: {DEFAULT_IN_DIR})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Folder for extracted .jsonl files (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--port",    type=int, default=8765)
    p.add_argument("--speed",   type=int, default=20,
                   help="Headless gamespeed multiplier (default: 20)")
    p.add_argument("--step-interval", type=int, default=30,
                   help="Engine frames between observations (default: 30 = 1 s)")
    p.add_argument("--allow-download", action="store_true",
                   help="Use pr-downloader to fetch missing game archives for "
                        "replays whose BAR game version isn't installed. Each "
                        "archive is ~50-150 MB. Does NOT help with engine-version "
                        "mismatches (those replays are always skipped).")
    args = p.parse_args()

    if not args.in_dir.exists():
        sys.stderr.write(f"Input directory not found: {args.in_dir}\n")
        return 1

    bar_data        = default_bar_data_dir()
    installed_ver   = detect_installed_game_version(bar_data)
    engine_ver      = detect_engine_version()
    replays         = sorted(args.in_dir.glob("*.sdfz"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir        : {args.in_dir}")
    print(f"Output dir       : {args.out_dir}")
    print(f"Engine version   : {engine_ver or '(unknown)'}")
    print(f"Installed BAR    : {installed_ver or '(unknown)'}")
    print(f"Replays found    : {len(replays)}")
    print()

    n_done = n_skipped_exists = n_skipped_engine = n_skipped_ver = n_failed = 0

    for i, replay in enumerate(replays, 1):
        out_path = args.out_dir / (replay.stem + ".jsonl")
        prefix   = f"[{i:>3}/{len(replays)}] {replay.name}"

        # Already extracted?
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"{prefix}  ->  skip (already extracted)")
            n_skipped_exists += 1
            continue

        # Engine version check: Spring only plays back demos from the exact
        # same engine binary version. Downloading game archives won't fix this.
        if engine_ver:
            demo_eng = get_demo_engine_version(replay)
            if demo_eng and demo_eng != engine_ver:
                print(f"{prefix}  ->  skip (engine mismatch: demo={demo_eng}, ours={engine_ver})")
                n_skipped_engine += 1
                continue

        # Game content version check.
        if installed_ver and not args.allow_download:
            demo_ver = get_demo_game_type(replay)
            if demo_ver != installed_ver:
                print(f"{prefix}  ->  skip (game version mismatch: {demo_ver})")
                n_skipped_ver += 1
                continue

        print(f"{prefix}  ->  extracting ...")
        try:
            result = extract(ReplayExtractorConfig(
                replay_path=replay,
                output_path=out_path,
                port=args.port,
                step_interval=args.step_interval,
                speed=args.speed,
                compact=True,
                fetch_missing_archive=args.allow_download,
            ))
            size_kb = out_path.stat().st_size / 1024
            print(
                f"    done — {result['frames_written']} frames, "
                f"{result['commands_written']} cmds, "
                f"{size_kb:,.0f} KB"
            )
            n_done += 1
        except Exception:
            print(f"    ERROR:")
            traceback.print_exc()
            # Remove partial output so a re-run can retry.
            if out_path.exists():
                out_path.unlink()
            n_failed += 1

    print()
    print(f"Extracted : {n_done}")
    print(f"Skipped   : {n_skipped_exists} already done, "
          f"{n_skipped_engine} engine mismatch, "
          f"{n_skipped_ver} game version mismatch")
    if n_failed:
        print(f"Failed    : {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
