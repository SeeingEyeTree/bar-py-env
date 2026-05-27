#!/usr/bin/env python3
"""Play a trained BC model in a live BAR game against SimpleAI.

The model is the one produced by examples/train_bc.py: a linear regression
that maps a small feature vector to a (target_x, target_z) the Commander
should move toward. Every step, we featurize the current observation the
same way the trainer did, project to a target, and issue a MOVE order to
the Commander.

It will not win. Its purpose is to demonstrate that the pipeline -- replay
extraction -> training -> live inference -- works end-to-end with a real
trained model. A v0.0.3 with a proper neural network and a real action
vocabulary is the next step.

Usage:
    python examples/bc_agent.py [--model bc_model.npz] [--max-steps 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env import BarEnv  # noqa: E402
from bar_env.env import EnvConfig  # noqa: E402
from bar_env.start_script import StartScriptConfig  # noqa: E402


# Must match the trainer's FEATURE_NAMES ordering exactly.
FEATURE_NAMES = [
    "frame_norm", "my_com_x_norm", "my_com_z_norm",
    "metal_norm", "energy_norm", "metal_income", "energy_income",
    "n_units_norm", "bias",
]


def featurize_live(obs: dict, com_unit: dict) -> np.ndarray:
    """Same feature vector the trainer uses, computed from a live obs.

    The live obs (mode='live') has a different shape than the replay obs
    (mode='replay'): live is from one team's POV with `my_units`/`resources`
    top-level; replay is god's-eye with `teams[<id>].units`/`teams[<id>].metal`.
    """
    map_x = obs.get("map", {}).get("size_x", 8192) or 8192
    map_z = obs.get("map", {}).get("size_z", 8192) or 8192
    res = obs.get("resources") or {}
    units = obs.get("my_units") or []

    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    f[0] = obs.get("frame", 0) / 30000.0
    f[1] = com_unit.get("x", 0) / max(map_x, 1)
    f[2] = com_unit.get("z", 0) / max(map_z, 1)
    f[3] = min((res.get("metal") or 0) / 1000.0, 5.0)
    f[4] = min((res.get("energy") or 0) / 1000.0, 5.0)
    f[5] = (res.get("metal_income") or 0) / 100.0
    f[6] = (res.get("energy_income") or 0) / 1000.0
    f[7] = len(units) / 50.0
    f[8] = 1.0
    return f


def find_commander(obs: dict) -> dict | None:
    """Pick the unit with the largest max_hp in the player's `my_units` list."""
    units = obs.get("my_units") or []
    if not units:
        return None
    return max(units, key=lambda u: u.get("max_hp", 0))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("bc_model.npz"),
                        help="Path to a model file produced by train_bc.py.")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Game steps before quitting (~minutes at default "
                             "step_interval=30).")
    parser.add_argument("--enemy-ai", default="SimpleAI",
                        help="Skirmish AI shortname to play against. "
                             "Default: SimpleAI. (Also valid: NullAI, BARb.)")
    parser.add_argument("--map", dest="map_name",
                        default="Comet Catcher Remake 1.8",
                        help="Map to play on.")
    args = parser.parse_args()

    if not args.model.exists():
        sys.stderr.write(
            f"Model not found: {args.model}\n"
            f"Run: python examples/train_bc.py <jsonl>\n"
        )
        return 1

    print(f"loading model: {args.model}")
    data = np.load(args.model, allow_pickle=True)
    W = data["W"]                                # shape (n_features, 2)
    print(f"  feature names : {list(data['feature_names'])}")
    print(f"  weights shape : {W.shape}")
    print(f"  train RMSE    : {float(data['train_rmse']):.1f}")
    print(f"  baseline RMSE : {float(data['baseline_rmse']):.1f}")
    print()

    # Wire SimpleAI as the opponent. We override the default NullAI here so
    # the agent has an actual opponent reacting to it.
    from bar_env import start_script  # noqa: E402
    original_render = start_script.render_start_script

    def render_with_enemy(cfg: StartScriptConfig) -> str:
        cfg.enemy_ai_short_name = args.enemy_ai
        cfg.enemy_ai_name = args.enemy_ai
        return original_render(cfg)

    start_script.render_start_script = render_with_enemy  # type: ignore[assignment]

    env_cfg = EnvConfig(
        map_name=args.map_name,
        step_interval=30,
        max_steps=args.max_steps,
        connect_timeout_s=180.0,
        step_timeout_s=60.0,
    )

    print(f"starting BarEnv: map={env_cfg.map_name}  vs {args.enemy_ai}")
    print()

    with BarEnv(env_cfg) as env:
        obs = env.reset()
        print(f"  reset OK. frame={obs.get('frame')} my_units={len(obs.get('my_units', []))}")

        map_x = obs.get("map", {}).get("size_x", 8192) or 8192
        map_z = obs.get("map", {}).get("size_z", 8192) or 8192

        for step_idx in range(env_cfg.max_steps):
            com = find_commander(obs)
            actions: list[dict] = []
            if com is not None:
                feat = featurize_live(obs, com)
                pred = feat @ W           # shape (2,) -- (target_x, target_z)
                target_x = clamp(float(pred[0]), 0.0, float(map_x))
                target_z = clamp(float(pred[1]), 0.0, float(map_z))
                actions.append({
                    "unit": com["id"],
                    "cmd": "MOVE",
                    "params": [target_x, 0.0, target_z],
                })
            obs, done = env.step(actions)
            if (step_idx + 1) % 10 == 0 or done:
                print(f"  step {step_idx+1:3d}: frame={obs.get('frame')} "
                      f"my_units={len(obs.get('my_units', []))} "
                      f"enemies_visible={len(obs.get('enemies', []))} "
                      f"sent={len(actions)} done={done}")
            if done:
                print(f"  -> terminal, result={obs.get('result')}")
                break

        print("\nclosing env (engine will quitforce after GameOver flush)...")

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
