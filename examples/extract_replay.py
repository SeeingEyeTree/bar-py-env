#!/usr/bin/env python3
"""End-to-end replay extraction smoke test.

Picks the smallest .sdfz in a folder (default: ./run/ then D:/BAR_Replays/),
plays it back through spring-headless with the bridge widget in replay mode,
and writes one JSONL line per extracted observation to disk.

Usage:
    python examples/extract_replay.py
    python examples/extract_replay.py --replay D:\\BAR_Replays\\Human_Pretraining\\some.sdfz
    python examples/extract_replay.py --search-dir D:\\BAR_Replays
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import EXTRACTED_DIR  # noqa: E402
from bar_env.replay import ReplayExtractorConfig, extract, iter_jsonl  # noqa: E402

# Default output folder: replays/ inside the project root so extracted files
# are easy to find alongside the code. Falls back to EXTRACTED_DIR if the env
# var override is set (preserves existing CI / script behaviour).
_LOCAL_REPLAYS_DIR = PACKAGE_ROOT / "replays"


def find_smallest_replay(search_roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.glob("**/*.sdfz"))
    # Drop zero-byte files (broken demos from earlier debugging).
    candidates = [p for p in candidates if p.stat().st_size > 1024]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=None,
                        help="Path to a specific .sdfz to extract.")
    parser.add_argument("--search-dir", type=Path,
                        default=Path("D:/BAR_Replays/Human_Pretraining"),
                        help="Directory to search for the smallest .sdfz. "
                             "Defaults to D:/BAR_Replays/Human_Pretraining.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Explicit path for the JSONL output. By default "
                             "writes to --out-dir/<replay-stem>.extracted.jsonl.")
    parser.add_argument("--out-dir", type=Path, default=_LOCAL_REPLAYS_DIR,
                        help=f"Folder for the JSONL output (used when --output "
                             f"is not given). Defaults to {_LOCAL_REPLAYS_DIR}.")
    parser.add_argument("--step-interval", type=int, default=30,
                        help="Game frames between extracted observations.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Safety cap (0 = run until GameOver).")
    parser.add_argument("--speed", type=int, default=20,
                        help="Gamespeed multiplier. Higher = faster wall-clock. "
                             "Capped by the replay's `maxspeed` modoption "
                             "(typically 20x for BAR multiplayer).")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip the pre-flight pr-downloader call. Use this "
                             "if you know the replay's game archive is already "
                             "installed (saves a few seconds per run).")
    parser.add_argument("--summary-only", action="store_true",
                        help="After extraction, print a summary of what was captured.")
    args = parser.parse_args()

    # Resolve replay path
    if args.replay is not None:
        replay = args.replay
        if not replay.exists():
            sys.stderr.write(f"Replay not found: {replay}\n")
            return 1
    else:
        roots = [args.search_dir]
        replay = find_smallest_replay(roots)
        if replay is None:
            sys.stderr.write(
                f"No .sdfz files found under: {roots}\n"
                f"Pass --replay <path> or --search-dir <dir>.\n"
            )
            return 1
        print(f"Smallest replay found: {replay} ({replay.stat().st_size // 1024} KB)")

    if args.output:
        output = args.output
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = args.out_dir / (replay.stem + ".extracted.jsonl")
    print(f"Writing observations to: {output}")
    print()

    cfg = ReplayExtractorConfig(
        replay_path=replay,
        output_path=output,
        step_interval=args.step_interval,
        max_steps=args.max_steps,
        speed=args.speed,
        fetch_missing_archive=not args.no_fetch,
    )

    summary = extract(cfg)

    print()
    print("=== extraction summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.summary_only or summary["frames_written"] == 0:
        return 0 if summary["frames_written"] > 0 else 1

    # Cheap sanity-check on the output: counts of commands per cmd_id.
    cmd_counts: dict[int, int] = {}
    unit_set: set[int] = set()
    teams_seen: set[str] = set()
    for obs in iter_jsonl(output):
        for cmd in obs.get("commands_this_frame", []):
            cmd_counts[cmd.get("cmd_id")] = cmd_counts.get(cmd.get("cmd_id"), 0) + 1
            if cmd.get("unit") is not None:
                unit_set.add(cmd["unit"])
        for tid in (obs.get("teams") or {}).keys():
            teams_seen.add(tid)

    print()
    print("=== content summary ===")
    print(f"  teams observed   : {sorted(teams_seen)}")
    print(f"  unique commanded units : {len(unit_set)}")
    print(f"  command id histogram   : {dict(sorted(cmd_counts.items(), key=lambda kv: -kv[1])[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
