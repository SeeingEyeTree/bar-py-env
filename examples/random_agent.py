#!/usr/bin/env python3
"""End-to-end smoke test: a random agent against a NullAI on Comet Catcher.

The agent picks a random map position each step and orders every owned unit
to MOVE there. After a handful of steps it quits the engine.

Usage:
    python examples/random_agent.py

A successful run prints "OK" at the end. If it doesn't, check the engine log
under ./run/<timestamp>/headless.log.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make the package importable when this script is run directly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env import BarEnv, actions  # noqa: E402
from bar_env.env import EnvConfig  # noqa: E402


def random_actions(obs: dict) -> list[dict]:
    """For every owned unit, issue a random action from the new vocabulary.

    Mix is deliberately move-heavy because that's the only action that always
    succeeds for any unit; the rest are interesting test cases for the widget's
    action plumbing.

    NOTE: this still won't play well -- a real agent needs to know which
    unit-def IDs are buildable by which builders. We don't have that mapping
    plumbed through yet, so we don't issue BUILD orders here. The next round
    will add a UnitDefs dump on first observation so an agent has access.
    """
    map_x = obs.get("map", {}).get("size_x", 0) or 0
    map_z = obs.get("map", {}).get("size_z", 0) or 0
    out: list[dict] = []
    for unit in obs.get("my_units", []):
        uid = unit["id"]
        roll = random.random()
        target_x = random.uniform(0, max(map_x, 1))
        target_z = random.uniform(0, max(map_z, 1))
        if roll < 0.7:
            out.append(actions.move(uid, target_x, target_z))
        elif roll < 0.85:
            # Patrol along a random axis -- often more visible than move
            out.append(actions.patrol(uid, target_x, target_z))
        elif roll < 0.92:
            # Toggle fire state at random
            state = random.choice([actions.FIRE_HOLD, actions.FIRE_RETURN, actions.FIRE_AT_WILL])
            out.append(actions.fire_state(uid, state))
        elif roll < 0.97:
            # Toggle move state at random
            state = random.choice([actions.MOVE_HOLD_POS, actions.MOVE_MANEUVER, actions.MOVE_ROAM])
            out.append(actions.move_state(uid, state))
        else:
            out.append(actions.stop(uid))
    return out


def main() -> int:
    cfg = EnvConfig(
        map_name="Comet Catcher Remake 1.8",
        port=8765,
        step_interval=30,   # 1 second of game time per step at 30 fps
        max_steps=30,       # 30s of game time -- fast iteration while debugging
        connect_timeout_s=120.0,  # generous: first-time map/game load is slow
        step_timeout_s=60.0,
    )

    print("== bar-py-env: random_agent smoke test ==")
    print(f"map           : {cfg.map_name}")
    print(f"port          : {cfg.port}")
    print(f"step_interval : {cfg.step_interval} frames")
    print(f"max_steps     : {cfg.max_steps}")
    print()

    with BarEnv(cfg) as env:
        print("Resetting env (this launches spring-headless)...")
        obs = env.reset()
        print(
            f"  connected. first obs: frame={obs.get('frame')} "
            f"my_units={len(obs.get('my_units', []))} "
            f"visible_enemies={len(obs.get('enemies', []))}"
        )

        for step_idx in range(cfg.max_steps):
            acts = random_actions(obs)
            obs, done = env.step(acts)
            # Avoid spamming the terminal on long runs: log every 30 steps + last
            if (step_idx + 1) % 30 == 0 or done or step_idx == cfg.max_steps - 1:
                print(
                    f"  step {step_idx + 1}: frame={obs.get('frame')} "
                    f"my_units={len(obs.get('my_units', []))} "
                    f"actions_sent={len(acts)} done={done}"
                )
            if done:
                print(f"  -> terminal observation, result={obs.get('result')}")
                break

        print("Closing env (quitting engine)...")
        # env.close() runs automatically via __exit__ here.

    demo = env.demo_path()
    print()
    if demo is not None:
        size_kb = demo.stat().st_size / 1024
        print(f"replay saved : {demo}  ({size_kb:,.0f} KB)")
        if size_kb < 50:
            print("WARNING: demo file is unusually small; it may not have finalized.")
        else:
            print("To watch it: open BAR's Chobby launcher -> Watch -> Replays.")
            print("Recoil writes demos straight into your BAR install's demos/ folder,")
            print("so Chobby finds them automatically.")
    else:
        print("(no replay file was produced -- check StartScriptConfig.record_demo)")
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
