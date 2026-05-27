#!/usr/bin/env python3
"""Tiny behavior-cloning trainer for the BAR replay JSONL.

This is a deliberately stupid baseline: for each observation, predict where
the player's Commander will be K seconds in the future. The model is a
linear regression solved with normal equations -- pure numpy, no PyTorch.

Why this task? Commander movement is the simplest non-trivial output we can
get from BAR replays: it's continuous, it's always meaningful (one unit per
team), and it's directly executable as a `MOVE` order. If a 5-feature linear
model can predict it better than a constant baseline, the pipeline works
and we can swap in a real network later.

Usage:
    python examples/train_bc.py <jsonl_path> [--out model.npz] [--lookahead-frames 90]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# Spring's Armada/Cortex Commander unit-def IDs differ per BAR version and
# faction, but the early game obs we have shows def=49 as the only ~3700 HP
# unit -- that's the Commander. Rather than hardcode the def ID, we identify
# the Commander as the unit with the highest max_hp on each team at frame 0.

FEATURE_NAMES = [
    "frame_norm",      # frame / 30000  (so 1.0 is ~16 min in)
    "my_com_x_norm",   # x / map_size_x
    "my_com_z_norm",   # z / map_size_z
    "metal_norm",      # min(metal / 1000, 1.0)
    "energy_norm",     # min(energy / 1000, 1.0)
    "metal_income",    # raw / 100
    "energy_income",   # raw / 1000
    "n_units_norm",    # n / 50
    "bias",            # 1.0 -- folded into W rather than separate b
]
N_FEATURES = len(FEATURE_NAMES)


def find_commander_ids_at_start(first_obs: dict) -> dict[str, int]:
    """For each team, identify the unit-id with the largest max_hp on frame 0."""
    out: dict[str, int] = {}
    for team_id, t in (first_obs.get("teams") or {}).items():
        units = t.get("units") or []
        if not units:
            continue
        com = max(units, key=lambda u: u.get("max_hp", 0))
        out[team_id] = com["id"]
    return out


def featurize(obs: dict, team_id: str, com_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Build the feature vector for one team's POV. Returns (x, com_pos)
    where com_pos is (cx, cz) of the team's Commander, or None if missing."""
    team = (obs.get("teams") or {}).get(team_id) or {}
    units = team.get("units") or []
    com = next((u for u in units if u.get("id") == com_id), None)
    if com is None:
        return np.zeros(N_FEATURES, dtype=np.float32), None

    map_x = obs.get("map", {}).get("size_x", 8192) or 8192
    map_z = obs.get("map", {}).get("size_z", 8192) or 8192

    f = np.zeros(N_FEATURES, dtype=np.float32)
    f[0] = (obs.get("frame", 0)) / 30000.0
    f[1] = com.get("x", 0) / max(map_x, 1)
    f[2] = com.get("z", 0) / max(map_z, 1)
    f[3] = min((team.get("metal") or 0) / 1000.0, 5.0)
    f[4] = min((team.get("energy") or 0) / 1000.0, 5.0)
    f[5] = (team.get("metal_income") or 0) / 100.0
    f[6] = (team.get("energy_income") or 0) / 1000.0
    f[7] = len(units) / 50.0
    f[8] = 1.0
    return f, np.array([com["x"], com["z"]], dtype=np.float32)


def build_dataset(
    jsonl_path: Path,
    lookahead_frames: int = 90,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Walk the JSONL once. For every (frame_now, frame_now+K) where both
    have a Commander present, emit one training example per team."""
    # Pass 1: gather all observations into a list keyed by sim frame.
    obs_by_frame: dict[int, dict] = {}
    first_obs = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ob = json.loads(line)
            if first_obs is None:
                first_obs = ob
            obs_by_frame[ob.get("frame", 0)] = ob
    if first_obs is None:
        raise RuntimeError(f"No observations in {jsonl_path}")

    com_ids = find_commander_ids_at_start(first_obs)
    print(f"  identified commanders: {com_ids}")

    sorted_frames = sorted(obs_by_frame.keys())
    if len(sorted_frames) < 2:
        raise RuntimeError("Need at least 2 frames in JSONL")
    step = sorted_frames[1] - sorted_frames[0]  # usually 30
    lookahead_steps = max(1, lookahead_frames // step)
    print(f"  step interval     : {step} frames")
    print(f"  lookahead         : {lookahead_steps} step(s) = {lookahead_steps*step} frames")

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for i, frame in enumerate(sorted_frames):
        target_idx = i + lookahead_steps
        if target_idx >= len(sorted_frames):
            break
        future_frame = sorted_frames[target_idx]
        obs_now = obs_by_frame[frame]
        obs_future = obs_by_frame[future_frame]
        for team_id, com_id in com_ids.items():
            fx, pos_now = featurize(obs_now, team_id, com_id)
            _, pos_future = featurize(obs_future, team_id, com_id)
            if pos_now is None or pos_future is None:
                continue  # commander died / not present
            X.append(fx)
            Y.append(pos_future)

    X_arr = np.stack(X) if X else np.zeros((0, N_FEATURES), np.float32)
    Y_arr = np.stack(Y) if Y else np.zeros((0, 2), np.float32)
    meta = {
        "step_interval": int(step),
        "lookahead_steps": int(lookahead_steps),
        "n_examples": int(len(X)),
        "feature_names": FEATURE_NAMES,
    }
    return X_arr, Y_arr, meta


def fit_linear(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form least squares: W = (X^T X + lambda I)^-1 X^T Y.

    Returns (W, train_mse). Adds tiny ridge regularization for stability.
    """
    n, d = X.shape
    XtX = X.T @ X
    XtX += 1e-3 * np.eye(d, dtype=X.dtype)  # ridge
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    Y_hat = X @ W
    mse = float(np.mean((Y_hat - Y) ** 2))
    return W, mse


def baseline_mse(X: np.ndarray, Y: np.ndarray) -> float:
    """Constant-predict-mean baseline for comparison."""
    if len(Y) == 0:
        return float("nan")
    mean = Y.mean(axis=0, keepdims=True)
    return float(np.mean((Y - mean) ** 2))


def identity_mse(X: np.ndarray, Y: np.ndarray) -> float:
    """'Predict commander stays where it is' baseline (uses input features 1,2).

    Since x[1] and x[2] are normalized commander x/z (over map size), we can't
    invert without knowing map size at this layer, but we approximate by
    computing the MSE assuming the model would output target = input position.
    Less useful but informative.
    """
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path,
                        help="Path to an extracted replay JSONL.")
    parser.add_argument("--out", type=Path, default=Path("bc_model.npz"),
                        help="Where to write the model weights.")
    parser.add_argument("--lookahead-frames", type=int, default=90,
                        help="Predict the commander's position this many frames "
                             "ahead. 90 frames = ~3 seconds.")
    args = parser.parse_args()

    if not args.jsonl.exists():
        sys.stderr.write(f"JSONL not found: {args.jsonl}\n")
        return 1

    print(f"reading {args.jsonl}")
    X, Y, meta = build_dataset(args.jsonl, lookahead_frames=args.lookahead_frames)
    print(f"  examples          : {len(X)}")
    print(f"  feature dim       : {X.shape[1] if len(X) else 0}")
    print(f"  target dim        : 2 (x, z) in map units")
    if len(X) < 20:
        sys.stderr.write(
            "Too few examples for a meaningful fit. Extract a longer replay "
            "or a higher-resolution JSONL (smaller step-interval).\n"
        )
        return 1

    W, mse_lin = fit_linear(X, Y)
    mse_const = baseline_mse(X, Y)
    rmse_lin = math.sqrt(mse_lin)
    rmse_const = math.sqrt(mse_const)
    print()
    print(f"  train RMSE        : {rmse_lin:8.1f} map units  (linear regression)")
    print(f"  baseline RMSE     : {rmse_const:8.1f} map units  (predict mean of Y)")
    print(f"  improvement       : {(1 - rmse_lin / max(rmse_const, 1e-6)) * 100:.1f}%")
    print(f"  weights shape     : {W.shape}")
    print()

    np.savez(
        args.out,
        W=W,
        feature_names=np.array(FEATURE_NAMES),
        step_interval=meta["step_interval"],
        lookahead_steps=meta["lookahead_steps"],
        n_examples=meta["n_examples"],
        train_rmse=float(rmse_lin),
        baseline_rmse=float(rmse_const),
    )
    print(f"saved model       : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
