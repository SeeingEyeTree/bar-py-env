#!/usr/bin/env python3
"""Two-phase behavioral cloning bot for BAR.

Phase 1  (first BUILD_ORDER_SECONDS seconds of game time):
  Replays the commander build order extracted from a reference JSONL.
  Uses examples/build_order_from_jsonl.extract_build_order() — no re-implementation.

Phase 2  (after build order completes):
  Model inference via SimpleBCModel from scripts/train_bc.py.
  Outputs one of: BUILD action on commander, FIGHT on army, or nothing (NO_OP).

Usage:
    python scripts/bc_bot.py
    python scripts/bc_bot.py --model bc_model.pt
    python scripts/bc_bot.py --reference-jsonl "D:\\BAR_Replays\\...\\replay.jsonl"
    python scripts/bc_bot.py --side Armada --enemy-ai BARb --max-steps 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PACKAGE_ROOT / "scripts"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))   # ensure scripts/train_bc.py takes priority

from bar_env import BarEnv, actions as act
from bar_env.env import EnvConfig
from bar_env.features import (
    GRID_H, GRID_W, N_SCALARS, ACTION_NAMES,
    load_unitdef_lookup, load_cluster_lookup,
    build_metal_spot_mask, extract_scalars, build_spatial_grid,
)
from bar_env.start_script import StartScriptConfig

# Import build order extractor from examples (reuse, don't reimplement)
sys.path.append(str(PACKAGE_ROOT / "examples"))
from build_order_from_jsonl import extract_build_order, _to_action  # type: ignore[import]
from train_bc import SimpleBCModel  # reuse model class from sibling script

BAR_ENV_DIR = PACKAGE_ROOT / "bar_env"
DEFAULT_MODEL = PACKAGE_ROOT / "bc_model.pt"
DEFAULT_REPLAY_DIR = Path(r"D:\BAR_Replays\training_data\replays_jsonl")
BUILD_ORDER_SECONDS = 55.0
FPS = 30

# ---------------------------------------------------------------------------
# Observation format conversion  (live → replay-style god's-eye frame)
# ---------------------------------------------------------------------------

def live_obs_to_frame(obs: dict, my_team_id: str = "0") -> dict[str, Any]:
    """Convert live BarEnv observation to the replay dict format features.py expects.

    Live format:   obs["my_units"], obs["resources"], obs["enemies"]
    Replay format: obs["teams"]["0"]["units"], obs["teams"]["0"]["metal"], ...
    """
    res = obs.get("resources") or {}
    my_units   = obs.get("my_units") or []
    vis_enemies = obs.get("enemies") or []

    my_team: dict[str, Any] = {
        "metal":          res.get("metal", 0),
        "energy":         res.get("energy", 0),
        "metal_income":   res.get("metal_income", 0),
        "energy_income":  res.get("energy_income", 0),
        "metal_pull":     res.get("metal_pull", 0),
        "energy_pull":    res.get("energy_pull", 0),
        "metal_storage":  res.get("metal_storage", 1000),
        "energy_storage": res.get("energy_storage", 1000),
        "units":          my_units,
    }
    # Enemy team: only visible units; eco is unknown (zeroed)
    enemy_id = "1" if my_team_id == "0" else "0"
    en_team: dict[str, Any] = {
        "metal": 0, "energy": 0, "metal_income": 0, "energy_income": 0,
        "metal_pull": 0, "energy_pull": 0,
        "metal_storage": 1000, "energy_storage": 1000,
        "units": vis_enemies,
    }
    return {
        "frame":  obs.get("frame", 0),
        "teams":  {my_team_id: my_team, enemy_id: en_team},
        "commands_this_frame": [],
    }


# ---------------------------------------------------------------------------
# Build order coordinate mirroring
# ---------------------------------------------------------------------------

def _mirror_cmd(cmd: dict, map_w: float, map_h: float) -> dict:
    """Point-reflect a replay command 180° about the map centre.

    Spawn positions in BAR 1v1 are at diagonally opposite corners, so when the
    live spawn is on the opposite side from the reference replay we apply a full
    point reflection: x -> map_w-x, z -> map_h-z.  Facing (4th BUILD param, 0-3)
    rotates by 180° as well.
    """
    params = [float(p) for p in cmd.get("params") or []]
    if len(params) >= 3:
        params[0] = map_w - params[0]
        params[2] = map_h - params[2]
    if len(params) == 4:
        params[3] = (int(params[3]) + 2) % 4
    return {**cmd, "params": params}


# ---------------------------------------------------------------------------
# Action decoding (model output -> Spring action dicts)
# ---------------------------------------------------------------------------

def decode_action(
    action_type: int,
    build_def_logits: np.ndarray,
    target_xz: np.ndarray,          # shape (2,), normalized [0,1]
    obs: dict,
    my_team_id: str,
    def_to_name: dict[int, str],
    build_def_vocab: list[int],
    map_meta: dict,
) -> list[dict]:
    """Convert model predictions to a list of Spring action dicts."""
    map_w = float(map_meta.get("size_x", 8192))
    map_h = float(map_meta.get("size_z", 6144))
    x_world = float(target_xz[0]) * map_w
    z_world = float(target_xz[1]) * map_h

    units = obs.get("my_units") or []
    if not units:
        return []

    # Find commander (highest max_hp)
    commander = max(units, key=lambda u: u.get("max_hp", 0))
    com_id    = commander["id"]

    # Identify army units (mobile, not commander)
    army_ids = [
        u["id"] for u in units
        if u["id"] != com_id and u.get("max_hp", 0) < 3000
    ]

    # Find a factory (very high max_hp non-commander)
    factory = next(
        (u for u in units if u["id"] != com_id and u.get("max_hp", 0) >= 3000),
        None,
    )

    if action_type == 0:   # NO_OP
        return []

    if action_type in (1, 2, 4, 5):   # BUILD_ECO / BUILD_LAB / BUILD_DEFENSE / BUILD_OTHER
        def_idx = int(np.argmax(build_def_logits))
        if def_idx < len(build_def_vocab) and build_def_vocab[def_idx] > 0:
            def_id   = build_def_vocab[def_idx]
            def_name = def_to_name.get(def_id, str(def_id))
            return [act.build(com_id, def_name, x_world, z_world)]
        return []

    if action_type == 3:   # BUILD_UNIT (factory queue)
        def_idx = int(np.argmax(build_def_logits))
        if factory and def_idx < len(build_def_vocab) and build_def_vocab[def_idx] > 0:
            def_id   = build_def_vocab[def_idx]
            def_name = def_to_name.get(def_id, str(def_id))
            return act.factory_queue(factory["id"], def_name, count=1)
        return []

    if action_type == 7:   # ATTACK
        targets = army_ids if army_ids else [com_id]
        return [act.fight(units=targets, x=x_world, z=z_world)]

    if action_type == 6:   # MOVE
        targets = army_ids if army_ids else [com_id]
        return [act.move(units=targets, x=x_world, z=z_world)]

    if action_type == 8:   # SUPPORT — guard commander with army
        if army_ids:
            return [act.guard(units=army_ids, target_unit=com_id)]

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_reference_jsonl(replay_dir: Path) -> Path | None:
    files = sorted(replay_dir.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--reference-jsonl", type=Path, default=None,
                    help="JSONL to extract build order from. Defaults to largest file in "
                         f"{DEFAULT_REPLAY_DIR}.")
    ap.add_argument("--build-order-seconds", type=float, default=BUILD_ORDER_SECONDS)
    ap.add_argument("--build-order-team", type=int, default=0,
                    help="Which team's build order to replay (default 0).")
    ap.add_argument("--side",      default="Cortex",
                    choices=["Armada", "Cortex", "Legion"])
    ap.add_argument("--enemy-ai",  default="BARb",
                    help="AI shortname. Default: BARb")
    ap.add_argument("--map",       default="Comet Catcher Remake 1.8")
    ap.add_argument("--port",      type=int, default=8765)
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="Max env steps (~33 min at 1s/step).")
    ap.add_argument("--speed",     type=int, default=10)
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    if not args.model.exists():
        print(f"Model not found: {args.model}\nRun scripts/train_bc.py first.",
              file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SimpleBCModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sc_mean = np.array(ckpt["sc_mean"], dtype=np.float32)
    sc_std  = np.array(ckpt["sc_std"],  dtype=np.float32)
    build_def_vocab = list(ckpt["build_def_vocab"])

    def_to_name    = load_unitdef_lookup(BAR_ENV_DIR)
    name_to_cluster = load_cluster_lookup(BAR_ENV_DIR)

    # -----------------------------------------------------------------------
    # Load build order
    # -----------------------------------------------------------------------
    ref_jsonl = args.reference_jsonl
    if ref_jsonl is None:
        ref_jsonl = _find_reference_jsonl(DEFAULT_REPLAY_DIR)
    if ref_jsonl is None or not ref_jsonl.exists():
        print(f"No reference JSONL found. Pass --reference-jsonl <path>.", file=sys.stderr)
        return 1

    print(f"Extracting {args.build_order_seconds:.0f}s build order from: {ref_jsonl}")
    bo = extract_build_order(ref_jsonl, team=args.build_order_team,
                             seconds=args.build_order_seconds)
    print(f"  {len(bo['commands'])} commands captured")
    print(f"  commander spawn: ({bo['spawn_x']:.0f}, {bo['spawn_z']:.0f})\n")

    build_order_cmds = list(bo["commands"])   # sorted by frame
    build_order_done = False
    bo_idx = 0
    bo_frame_limit = int(args.build_order_seconds * FPS)

    # -----------------------------------------------------------------------
    # Launch env
    # -----------------------------------------------------------------------
    opposite = "Cortex" if args.side == "Armada" else "Armada"
    script_cfg = StartScriptConfig(
        map_name=args.map,
        player_name="BCBot",
        player_side=args.side,
        enemy_side=opposite,
        enemy_ai_short_name=args.enemy_ai,
        enemy_ai_name=args.enemy_ai,
        death_mode="com",
        start_pos_type=0,
        record_demo=1,
    )
    env_cfg = EnvConfig(
        port=args.port,
        step_interval=30,
        max_steps=args.max_steps,
        connect_timeout_s=180.0,
        step_timeout_s=60.0,
        script_config=script_cfg,
        speed=args.speed,
    )

    print(f"== BCBot  {args.side} vs {args.enemy_ai}  on {args.map} ==")
    print(f"Speed: {args.speed}x   Build order: first {args.build_order_seconds:.0f}s   "
          f"Max steps: {args.max_steps}\n")

    map_meta: dict = {}
    metal_spot_mask: np.ndarray | None = None
    last_frame: int = args.max_steps * 30
    MY_TEAM = "0"

    with BarEnv(env_cfg) as env:
        obs = env.reset()
        map_meta = obs.get("map") or {}
        metal_spot_mask = build_metal_spot_mask(map_meta)
        map_w = float(map_meta.get("size_x", 8192))
        map_h = float(map_meta.get("size_z", 6144))
        print(f"Connected at frame {obs.get('frame')}. "
              f"Metal spots: {len(map_meta.get('metal_spots', []))}\n")

        # Find live commander id and spawn position
        my_units    = obs.get("my_units") or []
        live_com_id = min((u["id"] for u in my_units), default=None) if my_units else None
        live_com    = next((u for u in my_units if u["id"] == live_com_id), None) if live_com_id else None
        live_com_x  = float(live_com.get("x", map_w / 2)) if live_com else map_w / 2

        # Mirror build order coords if live spawn is on the opposite side from reference
        ref_left    = bo["spawn_x"] < map_w / 2
        live_left   = live_com_x   < map_w / 2
        mirror_bo   = ref_left != live_left
        if mirror_bo:
            print(f"  Spawn mismatch: ref_x={bo['spawn_x']:.0f}  live_x={live_com_x:.0f} "
                  f"-> mirroring build order coordinates")
        else:
            print(f"  Spawn match: ref_x={bo['spawn_x']:.0f}  live_x={live_com_x:.0f}")

        for step in range(args.max_steps):
            cur_frame = int(obs.get("frame", 0))
            out_actions: list[dict] = []

            # Phase 1: build order
            if not build_order_done:
                while bo_idx < len(build_order_cmds) and \
                      build_order_cmds[bo_idx]["frame"] <= cur_frame:
                    if live_com_id is not None:
                        cmd = build_order_cmds[bo_idx]
                        if mirror_bo:
                            cmd = _mirror_cmd(cmd, map_w, map_h)
                        cmd_act = _to_action(cmd, live_com_id)
                        out_actions.append(cmd_act)
                    bo_idx += 1

                if cur_frame >= bo_frame_limit or bo_idx >= len(build_order_cmds):
                    build_order_done = True
                    print(f"  step {step:>4d}  frame={cur_frame:>6d}  "
                          f"Build order complete -> switching to model inference")

            # Phase 2: model inference
            else:
                frame_dict = live_obs_to_frame(obs, MY_TEAM)
                sc = extract_scalars(
                    frame_dict, MY_TEAM, last_frame, map_meta,
                    name_to_cluster, def_to_name,
                )
                sc_norm = (sc - sc_mean) / sc_std
                sp = build_spatial_grid(
                    frame_dict, MY_TEAM, map_meta,
                    metal_spot_mask, def_to_name,
                    set(name_to_cluster.keys()),
                )

                sc_t = torch.from_numpy(sc_norm).unsqueeze(0).float()
                sp_t = torch.from_numpy(sp).unsqueeze(0).float()

                with torch.no_grad():
                    logits_at, logits_bd, pred_pos = model(sc_t, sp_t)

                atype   = int(logits_at.argmax(dim=1).item())
                bd_arr  = logits_bd.squeeze(0).numpy()
                pos_arr = pred_pos.squeeze(0).numpy()

                out_actions = decode_action(
                    atype, bd_arr, pos_arr,
                    obs, MY_TEAM,
                    def_to_name, build_def_vocab, map_meta,
                )

            obs, done = env.step(out_actions)

            if step % 60 == 0 or done:
                res    = obs.get("resources") or {}
                phase  = "BUILD_ORDER" if not build_order_done else "MODEL"
                nunits = len(obs.get("my_units") or [])
                print(
                    f"  step {step:>4d}  frame={cur_frame:>6d}  [{phase:<12}]"
                    f"  units={nunits:3d}"
                    f"  M={res.get('metal',0):6.0f}/{res.get('metal_storage',0):.0f}"
                    f"  E={res.get('energy',0):6.0f}/{res.get('energy_storage',0):.0f}"
                    + (f"  [{obs.get('result')}]" if done else "")
                )
            if done:
                break

    demo = env.demo_path()
    if demo and demo.exists():
        print(f"\nReplay saved: {demo}  ({demo.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
