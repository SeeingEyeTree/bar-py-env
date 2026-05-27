#!/usr/bin/env python3
"""Play back the first 60 seconds of a build order from an extracted replay JSONL.

Reads an already-extracted .jsonl (produced by examples/extract_replay.py),
pulls out every command the commander issued in the first N seconds, then
launches a fresh headless game and replays those commands at the right frame.

No .sdfz file needed -- the JSONL contains everything required.

Usage:
    python examples/build_order_from_jsonl.py
    python examples/build_order_from_jsonl.py --jsonl path/to/replay.jsonl
    python examples/build_order_from_jsonl.py --seconds 30 --parse-only
    python examples/build_order_from_jsonl.py --side Cortex --map "Supreme Isthmus v2.1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env import BarEnv                          # noqa: E402
from bar_env.env import EnvConfig                   # noqa: E402
from bar_env.replay import iter_jsonl               # noqa: E402
from bar_env.start_script import StartScriptConfig  # noqa: E402

# Default: the JSONL the user already extracted.
DEFAULT_JSONL = Path(
    r"C:\Users\malco\AppData\Local\Programs\Beyond-All-Reason\data\demos\build_order.jsonl"
)

FPS = 30  # BAR runs at 30 simulation frames per second


# ---------------------------------------------------------------------------
# Build order extraction
# ---------------------------------------------------------------------------

def _cmd_label(cid: int) -> str:
    if cid < 0:
        return f"BUILD(def={-cid})"
    return {
        0: "STOP", 5: "WAIT", 10: "MOVE", 14: "FIGHT", 15: "PATROL",
        16: "AREA_ATTACK", 20: "ATTACK", 25: "GUARD",
        40: "REPAIR", 45: "FIRE_STATE", 50: "MOVE_STATE",
        65: "SELFD", 75: "LOAD_UNITS", 80: "UNLOAD_UNITS", 85: "ONOFF",
        90: "RECLAIM", 95: "CLOAK", 100: "STOCKPILE", 105: "MANUALFIRE",
        110: "RESTORE", 115: "REPEAT", 120: "TRAJECTORY",
        125: "RESURRECT", 130: "CAPTURE",
    }.get(cid, f"CMD_{cid}")


def extract_build_order(
    jsonl_path: Path,
    *,
    team: int = 0,
    seconds: float = 60.0,
) -> dict[str, Any]:
    """Return commander spawn info and its commands from the first `seconds`."""
    frame_limit = int(seconds * FPS)
    team_str = str(team)

    # Step 1: find commander on earliest obs that has units for this team.
    commander_uid: int | None = None
    commander_def = 0
    spawn_x = spawn_y = spawn_z = 0.0

    for obs in iter_jsonl(jsonl_path):
        t = (obs.get("teams") or {}).get(team_str)
        if not (t and t.get("units")):
            continue
        first = min(t["units"], key=lambda u: u.get("id", 1 << 30))
        commander_uid = int(first["id"])
        commander_def = int(first.get("def") or 0)
        spawn_x = float(first.get("x") or 0.0)
        spawn_y = float(first.get("y") or 0.0)
        spawn_z = float(first.get("z") or 0.0)
        break

    if commander_uid is None:
        raise RuntimeError(
            f"No units found for team {team} in {jsonl_path}.\n"
            f"Check that --team matches the player's team in the replay."
        )

    # Step 2: collect commander commands within the time window, deduped.
    commands: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for obs in iter_jsonl(jsonl_path):
        obs_frame = int(obs.get("frame") or 0)
        if obs_frame > frame_limit + 60:  # a little slack past the window
            break
        for c in obs.get("commands_this_frame") or []:
            cmd_frame = int(c.get("frame") or obs_frame)
            if cmd_frame >= frame_limit:
                continue
            if c.get("team") != team:
                continue
            if c.get("unit") != commander_uid:
                continue
            # REMOVE (2) and INSERT (1) reference internal engine command tags
            # that differ between games — skip them; the shift-queue ordering
            # handles sequencing correctly without them.
            if c.get("cmd_id") in (1, 2):
                continue
            cid = c.get("cmd_id")
            params = list(c.get("params") or [])
            opts = dict(c.get("options") or {})
            # Spring re-broadcasts commands multiple times; deduplicate.
            key = (
                cmd_frame,
                int(cid) if cid is not None else 0,
                tuple(round(float(p), 1) for p in params),
            )
            if key in seen:
                continue
            seen.add(key)
            commands.append({
                "frame": cmd_frame,
                "cmd_id": int(cid) if cid is not None else 0,
                "params": params,
                "options": opts,
            })

    commands.sort(key=lambda c: c["frame"])
    return {
        "commander_uid": commander_uid,
        "commander_def": commander_def,
        "spawn_x": spawn_x,
        "spawn_y": spawn_y,
        "spawn_z": spawn_z,
        "commands": commands,
    }


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def _to_action(cmd: dict[str, Any], live_commander_uid: int) -> dict[str, Any]:
    """Convert a captured replay command to the dict the widget accepts."""
    cid = int(cmd["cmd_id"])
    params = [float(p) for p in cmd.get("params") or []]
    # Keep only the modifier flags the widget uses; drop internal bookkeeping.
    options = {
        k: v for k, v in (cmd.get("options") or {}).items()
        if k in ("shift", "ctrl", "alt", "right", "meta") and v
    }
    if cid < 0:
        return {
            "unit": live_commander_uid,
            "cmd": "BUILD",
            "def": -cid,
            "params": params,
            "options": options,
        }
    return {
        "unit": live_commander_uid,
        "cmd": cid,
        "params": params,
        "options": options,
    }


def _action_label(act: dict[str, Any]) -> str:
    if act.get("cmd") == "BUILD":
        return f"BUILD(def={act.get('def')})"
    return _cmd_label(int(act.get("cmd", 0)))


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_build_order(bo: dict[str, Any], args: argparse.Namespace) -> int:
    opposite = "Cortex" if args.side == "Armada" else "Armada"
    script_cfg = StartScriptConfig(
        map_name=args.map,
        player_name="PyEnv-BuildOrder",
        player_side=args.side,
        enemy_side=opposite,
        enemy_ai_short_name="NullAI",
        enemy_ai_name="NullAI",
        start_pos_type=3,                       # use explicit StartPosX/Z
        player_start_pos_x=bo["spawn_x"],
        player_start_pos_z=bo["spawn_z"],
        speed=0,                                # uncapped headless speed
    )
    cfg = EnvConfig(
        map_name=script_cfg.map_name,
        port=args.port,
        step_interval=30,
        max_steps=args.max_steps,
        connect_timeout_s=180.0,
        step_timeout_s=60.0,
        script_config=script_cfg,
        speed=20,
    )

    print("== launching bot match ==")
    print(f"  map    : {script_cfg.map_name}")
    print(f"  side   : {script_cfg.player_side} vs {script_cfg.enemy_side} (NullAI)")
    print(f"  spawn  : ({bo['spawn_x']:.0f}, {bo['spawn_z']:.0f})")
    print()

    queue = list(bo["commands"])  # sorted by frame
    queue_idx = 0
    sent_total = 0

    with BarEnv(cfg) as env:
        obs = env.reset()
        my_units = obs.get("my_units") or []
        if not my_units:
            print("ERROR: first observation has no units. Cannot find commander.")
            return 1

        commander = min(my_units, key=lambda u: u.get("id", 1 << 30))
        live_uid = int(commander["id"])
        print(
            f"  commander  id={live_uid}  def={commander.get('def')}  "
            f"@ ({commander.get('x', 0):.0f}, {commander.get('z', 0):.0f})"
        )
        if (
            abs(commander.get("x", 0) - bo["spawn_x"]) > 64
            or abs(commander.get("z", 0) - bo["spawn_z"]) > 64
        ):
            print(
                "  [warn] commander is >64 units from the captured spawn position.\n"
                "         Build positions may not match the original replay.\n"
                "         Try start_pos_type=2 or check the map's start boxes."
            )
        print()

        build_order_done = False
        for step_idx in range(args.max_steps):
            current_frame = int(obs.get("frame") or 0)

            # Pop all commands whose replay-frame has been reached.
            to_send: list[dict[str, Any]] = []
            while queue_idx < len(queue) and queue[queue_idx]["frame"] <= current_frame:
                to_send.append(_to_action(queue[queue_idx], live_uid))
                queue_idx += 1

            if to_send:
                sent_total += len(to_send)
                labels = ", ".join(_action_label(a) for a in to_send)
                print(
                    f"  step {step_idx:>3d}  frame={current_frame:>5d}  "
                    f"sent {len(to_send):>2d}: {labels}"
                )

            obs, done = env.step(to_send)
            if done:
                print(f"  -> game ended (result={obs.get('result')})")
                break

            if not build_order_done and queue_idx >= len(queue):
                build_order_done = True
                remaining = args.max_steps - step_idx - 1
                print(
                    f"  step {step_idx+1:>3d}  build order complete "
                    f"({remaining} idle steps remaining)"
                )

    print(f"\n  sent {sent_total} / {len(queue)} commands total.")

    demo = env.demo_path()
    if demo and demo.exists():
        print(
            f"\nBot replay saved: {demo}  ({demo.stat().st_size / 1024:,.0f} KB)\n"
            f"Open Chobby -> Watch -> Replays to review the bot's build."
        )
    else:
        print("\n(no replay file found; check the engine log)")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--jsonl", type=Path, default=DEFAULT_JSONL,
        help=f"Extracted replay JSONL to learn from. Default: {DEFAULT_JSONL}",
    )
    p.add_argument(
        "--team", type=int, default=0,
        help="Team whose commander to replay (default: 0).",
    )
    p.add_argument(
        "--seconds", type=float, default=60.0,
        help="How many seconds of the replay to capture (default: 60).",
    )
    p.add_argument(
        "--side", default="Cortex", choices=["Armada", "Cortex", "Legion"],
        help="Faction for the bot (default: Cortex).",
    )
    p.add_argument(
        "--map", default="Comet Catcher Remake 1.8",
        help="Map to play on (default: Comet Catcher Remake 1.8).",
    )
    p.add_argument(
        "--port", type=int, default=8765,
    )
    p.add_argument(
        "--max-steps", type=int, default=150,
        help="Env steps to run (default: 150 = 2.5 min at 1s/step).",
    )
    p.add_argument(
        "--parse-only", action="store_true",
        help="Print the extracted build order without launching a game.",
    )
    args = p.parse_args()

    if not args.jsonl.exists():
        sys.stderr.write(f"JSONL file not found: {args.jsonl}\n")
        return 1

    print(f"Loading build order from: {args.jsonl}")
    bo = extract_build_order(args.jsonl, team=args.team, seconds=args.seconds)

    print()
    print("=== build order ===")
    print(f"  team             : {args.team}")
    print(f"  commander uid    : {bo['commander_uid']}  (def {bo['commander_def']})")
    print(f"  spawn (x, z)     : ({bo['spawn_x']:.0f}, {bo['spawn_z']:.0f})")
    print(f"  commands captured: {len(bo['commands'])} in first {args.seconds:.0f}s "
          f"(frames 0–{int(args.seconds * FPS)})")
    print()
    print(f"  {'frame':>5}  {'command':<22}  params")
    print(f"  {'-----':>5}  {'-------':<22}  ------")
    for c in bo["commands"]:
        print(
            f"  {c['frame']:>5d}  {_cmd_label(c['cmd_id']):<22}  {c['params']}"
        )

    if args.parse_only:
        return 0

    print()
    return run_build_order(bo, args)


if __name__ == "__main__":
    raise SystemExit(main())
