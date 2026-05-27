#!/usr/bin/env python3
"""Simple rules-based bot: economy → production → attack.

Plays as Cortex against BARb AI on Comet Catcher Remake 1.8.

Build order:
  1. Commander builds a mex at each of the 2 nearest metal spots.
  2. Commander builds 3 solar collectors (shift-queued).
  3. Commander builds a vehicle plant (shift-queued).
  4. Vehicle plant queues raiders on repeat.
  5. Once 3+ raiders are idle, send a fight-move wave to the enemy base.

Usage:
    python examples/rules_bot.py

A replay is written to your BAR install's demos/ folder. Open Chobby →
Watch → Replays to view it after the game ends.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env import BarEnv, actions  # noqa: E402
from bar_env.env import EnvConfig  # noqa: E402
from bar_env.start_script import StartScriptConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Faction defs (string names resolved by the widget via UnitDefNames).
# ---------------------------------------------------------------------------

FACTIONS: dict[str, dict[str, str]] = {
    "Cortex": {
        "mex":    "cormex",
        "solar":  "corsolar",
        "vp":     "corvp",
        "raider": "corraid",
    },
    "Armada": {
        "mex":    "armmex",
        "solar":  "armsolar",
        "vp":     "armvp",
        "raider": "armflash",
    },
}

# max_hp thresholds for classifying units without knowing def names.
FACTORY_HP_MIN = 3000   # vehicle plants: ~5000 HP
RAIDER_HP_MAX  = 600    # raiders: ~440 HP

WAVE_SIZE = 3           # idle raiders needed before we launch an attack


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class RulesBot:
    def __init__(self, faction: str = "Cortex") -> None:
        self.defs = FACTIONS[faction]

        self.com_id:     int | None = None
        self.factory_id: int | None = None
        self.map_x = 0.0
        self.map_z = 0.0
        self.com_x = 0.0
        self.com_z = 0.0

        # Build queue: list of (def_name, x, z, shift) populated on first obs
        # once we have metal spot data. None = not yet initialised.
        self._build_queue: list[tuple[str, float, float, bool]] | None = None
        self._build_index = 0       # next entry in _build_queue to issue
        self._factory_queued = False

        # Raiders that have already been ordered to attack.
        self._ordered_raiders: set[int] = set()

    # --- helpers ------------------------------------------------------------

    def _classify(self, unit: dict) -> str:
        hp = unit.get("max_hp", 0)
        if hp >= FACTORY_HP_MIN:
            return "factory"
        if 0 < hp <= RAIDER_HP_MAX:
            return "raider"
        return "building"

    def _nearest_spots(
        self, spots: list[dict], n: int
    ) -> list[dict]:
        """Return up to n spots nearest to the commander, closest first."""
        return sorted(
            spots,
            key=lambda s: (s["x"] - self.com_x) ** 2 + (s["z"] - self.com_z) ** 2,
        )[:n]

    def _make_build_queue(
        self, spots: list[dict]
    ) -> list[tuple[str, float, float, bool]]:
        """Return the ordered list of (def, x, z, shift) build orders."""
        queue: list[tuple[str, float, float, bool]] = []

        # 2 mexes at the 2 nearest metal spots.
        for i, spot in enumerate(self._nearest_spots(spots, 2)):
            queue.append((self.defs["mex"], float(spot["x"]), float(spot["z"]), i > 0))

        # 3 solar collectors beside the commander.
        for i in range(3):
            queue.append((
                self.defs["solar"],
                self.com_x + 200.0 + i * 180.0,
                self.com_z,
                True,
            ))

        # 1 vehicle plant behind the commander.
        queue.append((self.defs["vp"], self.com_x - 500.0, self.com_z + 300.0, True))
        return queue

    # --- main loop ----------------------------------------------------------

    def act(self, obs: dict) -> list[dict]:
        my_units: list[dict] = obs.get("my_units", [])
        if not my_units:
            return []

        map_info = obs.get("map", {})
        self.map_x = float(map_info.get("size_x", 8192))
        self.map_z = float(map_info.get("size_z", 8192))

        # --- identify key units ------------------------------------------
        if self.com_id is None:
            com = min(my_units, key=lambda u: u["id"])
            self.com_id = com["id"]
            self.com_x  = float(com["x"])
            self.com_z  = float(com["z"])

        if self.factory_id is None:
            for u in my_units:
                if u["id"] != self.com_id and self._classify(u) == "factory":
                    self.factory_id = u["id"]
                    break

        # --- initialise build queue once metal spots are available --------
        if self._build_queue is None and self.com_id is not None:
            spots = map_info.get("metal_spots") or []
            # Initialise even if spots is empty (fallback to commander pos).
            if spots or obs.get("frame", 0) > 30:
                if not spots:
                    spots = [{"x": self.com_x, "z": self.com_z, "m": 1}]
                self._build_queue = self._make_build_queue(spots)

        out: list[dict] = []

        # --- Phase 1: Economy (issue one build order per step) -----------
        if (
            self.com_id is not None
            and self._build_queue is not None
            and self._build_index < len(self._build_queue)
        ):
            def_name, x, z, shift = self._build_queue[self._build_index]
            out.append(actions.build(
                self.com_id, def_name, x, z,
                options={"shift": True} if shift else None,
            ))
            self._build_index += 1

        # --- Phase 2: Production (factory queues raiders once ready) -----
        if self.factory_id is not None and not self._factory_queued:
            out += actions.factory_queue(self.factory_id, self.defs["raider"], count=5)
            out.append(actions.repeat(self.factory_id, on=True))
            self._factory_queued = True

        # --- Phase 3: Combat (send raider waves) -------------------------
        idle_raiders = [
            u for u in my_units
            if u["id"] != self.com_id
            and u["id"] != self.factory_id
            and self._classify(u) == "raider"
            and u["id"] not in self._ordered_raiders
        ]
        if len(idle_raiders) >= WAVE_SIZE:
            target_x = self.map_x - self.com_x
            target_z = self.map_z - self.com_z
            uids = [u["id"] for u in idle_raiders]
            out.append(actions.fight(units=uids, x=target_x, z=target_z))
            self._ordered_raiders.update(uids)

        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    script_cfg = StartScriptConfig(
        map_name="Comet Catcher Remake 1.8",
        player_side="Cortex",
        enemy_side="Armada",
        enemy_ai_short_name="BARb",
        enemy_ai_version="stable",
        death_mode="com",    # game ends when either commander dies
        start_pos_type=0,
        record_demo=1,
    )
    cfg = EnvConfig(
        port=8765,
        step_interval=30,    # 1 game-second per step
        max_steps=2000,      # ~33 game-minutes
        connect_timeout_s=180.0,
        step_timeout_s=60.0,
        script_config=script_cfg,
        speed=10,            # 10× faster than realtime
    )

    print("== bar-py-env: rules_bot vs BARb (deathmode=com) ==")
    print(f"map   : {script_cfg.map_name}")
    print(f"sides : {script_cfg.player_side} (bot) vs {script_cfg.enemy_side} (BARb)")
    print(f"speed : {cfg.speed}×")
    print()

    bot = RulesBot(faction="Cortex")

    with BarEnv(cfg) as env:
        print("Launching engine (this may take 30-60 s on first run)...")
        obs = env.reset()
        spots = obs.get("map", {}).get("metal_spots") or []
        print(f"Connected at frame {obs.get('frame')}. Metal spots on map: {len(spots)}\n")

        done = False
        step = 0
        while not done:
            acts  = bot.act(obs)
            obs, done = env.step(acts)
            step += 1
            if step % 60 == 0 or done:
                res = obs.get("resources", {})
                print(
                    f"  step {step:5d}  frame {obs.get('frame', 0):7d}"
                    f"  units={len(obs.get('my_units', [])):3d}"
                    f"  M={res.get('metal', 0):6.0f}/{res.get('metal_storage', 0):.0f}"
                    f"  E={res.get('energy', 0):6.0f}/{res.get('energy_storage', 0):.0f}"
                    + (f"  [{obs.get('result')}]" if done else "")
                )

    demo = env.demo_path()
    print()
    if demo is not None:
        size_kb = demo.stat().st_size / 1024
        print(f"Replay saved: {demo}  ({size_kb:,.0f} KB)")
        print("Open BAR → Chobby → Watch → Replays to watch it.")
    else:
        print("(no replay file produced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
