#!/usr/bin/env python3
"""Build BC training dataset from extracted replay JSONL files.

Reads all *.jsonl in REPLAY_DIR, runs feature extraction for both teams in
every game frame, and saves a compressed numpy archive ready for train_bc.py.

Usage:
    python scripts/build_bc_dataset.py
    python scripts/build_bc_dataset.py --replay-dir D:\\BAR_Replays\\training_data\\replays_jsonl
    python scripts/build_bc_dataset.py --out D:\\BAR_Replays\\training_data\\bc_dataset.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.features import (
    GRID_H, GRID_W, N_SCALARS, N_ACTIONS, ACTION_NAMES,
    load_unitdef_lookup, load_cluster_lookup,
    build_metal_spot_mask, extract_scalars, build_spatial_grid,
    classify_commands,
)
from bar_env.replay import iter_jsonl

BAR_ENV_DIR  = PACKAGE_ROOT / "bar_env"
DEFAULT_REPLAY_DIR = Path(r"D:\BAR_Replays\training_data\replays_jsonl")
DEFAULT_OUT        = Path(r"D:\BAR_Replays\training_data\bc_dataset.npz")


def build_dataset(replay_dir: Path, out_path: Path) -> None:
    print(f"Loading unit lookups from {BAR_ENV_DIR} ...")
    def_to_name    = load_unitdef_lookup(BAR_ENV_DIR)
    name_to_cluster = load_cluster_lookup(BAR_ENV_DIR)
    mobile_names   = set(name_to_cluster.keys())

    jsonl_files = sorted(replay_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {replay_dir}")
        return
    print(f"Found {len(jsonl_files)} replay files\n")

    # -----------------------------------------------------------------------
    # Pass 1: build build_def vocabulary from most-frequent def IDs
    # -----------------------------------------------------------------------
    print("Pass 1: counting build def frequencies ...")
    def_freq: Counter[int] = Counter()
    for jpath in jsonl_files:
        for obs in iter_jsonl(jpath):
            for cmd in obs.get("commands_this_frame") or []:
                cid = int(cmd.get("cmd_id", 0))
                if cid < 0:
                    def_freq[-cid] += 1

    top40 = [def_id for def_id, _ in def_freq.most_common(40)]
    def_to_vocab: dict[int, int] = {d: i for i, d in enumerate(top40)}
    VOCAB_OTHER = 40
    build_def_vocab = np.array(top40 + [-1], dtype=np.int32)  # index 40 = "other"

    print("Top-15 built unit defs:")
    for i, d in enumerate(top40[:15]):
        print(f"  [{i:2d}] def={d:4d}  name={def_to_name.get(d, '?')}")
    print()

    # -----------------------------------------------------------------------
    # Pass 2: extract features for every (frame, team) pair
    # -----------------------------------------------------------------------
    print("Pass 2: extracting features ...")

    scalars_list:    list[np.ndarray]       = []
    spatial_list:    list[np.ndarray]       = []
    action_type_list: list[int]             = []
    build_def_idx_list: list[int]           = []
    target_xz_list:  list[tuple[float, float]] = []
    game_id_list:    list[int]              = []

    for game_idx, jpath in enumerate(jsonl_files):
        frames = list(iter_jsonl(jpath))
        if len(frames) < 3:
            print(f"  [{game_idx:2d}] {jpath.name}  SKIP (too short)")
            continue

        header = frames[0]
        if header.get("type") != "header":
            print(f"  [{game_idx:2d}] {jpath.name}  SKIP (no header line)")
            continue

        map_meta = header.get("map", {})
        metal_spot_mask = build_metal_spot_mask(map_meta)

        game_frames = [f for f in frames if "frame" in f and "teams" in f]
        if not game_frames:
            print(f"  [{game_idx:2d}] {jpath.name}  SKIP (no game frames)")
            continue

        last_frame = max(f["frame"] for f in game_frames)
        n_samples_before = len(scalars_list)

        for obs in game_frames:
            teams    = obs.get("teams", {})
            commands = obs.get("commands_this_frame") or []

            for team_id in teams:
                sc  = extract_scalars(
                    obs, team_id, last_frame, map_meta,
                    name_to_cluster, def_to_name,
                )
                sp  = build_spatial_grid(
                    obs, team_id, map_meta, metal_spot_mask,
                    def_to_name, mobile_names,
                )
                atype, def_id, pos = classify_commands(
                    commands, team_id, def_to_name, mobile_names, map_meta,
                )

                scalars_list.append(sc)
                spatial_list.append(sp)
                action_type_list.append(atype)
                game_id_list.append(game_idx)

                if def_id is not None and atype in (1, 2, 3, 4, 5):
                    build_def_idx_list.append(def_to_vocab.get(def_id, VOCAB_OTHER))
                else:
                    build_def_idx_list.append(-1)

                if pos is not None and atype != 0:
                    target_xz_list.append(pos)
                else:
                    target_xz_list.append((-1.0, -1.0))

        n_added = len(scalars_list) - n_samples_before
        print(f"  [{game_idx:2d}] {jpath.name}  +{n_added} samples  last_frame={last_frame}")

    N = len(scalars_list)
    if N == 0:
        print("No samples extracted. Check replay dir.")
        return

    print(f"\nTotal samples: {N}  "
          f"(~{N // max(len(jsonl_files), 1)} per replay × {len(jsonl_files)} replays × 2 teams)\n")

    atype_arr = np.array(action_type_list, dtype=np.int32)
    print("Action type distribution:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((atype_arr == i).sum())
        print(f"  {i}  {name:<16}  {count:6d}  ({100 * count / N:.1f}%)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scalars=np.stack(scalars_list).astype(np.float32),
        spatial=np.stack(spatial_list).astype(np.float32),
        action_type=atype_arr,
        build_def_idx=np.array(build_def_idx_list, dtype=np.int32),
        target_xz=np.array(target_xz_list, dtype=np.float32),
        game_id=np.array(game_id_list, dtype=np.int32),
        build_def_vocab=build_def_vocab,
    )
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"Saved  {out_path}  ({size_mb:.1f} MB)")
    print(f"Array shapes:")
    print(f"  scalars   : ({N}, {N_SCALARS})")
    print(f"  spatial   : ({N}, 7, {GRID_H}, {GRID_W})")
    print(f"  action_type / build_def_idx / target_xz / game_id : ({N},)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
                   help=f"Directory containing *.jsonl files. Default: {DEFAULT_REPLAY_DIR}")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output .npz path. Default: {DEFAULT_OUT}")
    args = p.parse_args()

    if not args.replay_dir.exists():
        print(f"Replay dir not found: {args.replay_dir}", file=sys.stderr)
        return 1

    build_dataset(args.replay_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
