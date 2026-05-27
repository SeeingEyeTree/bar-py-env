#!/usr/bin/env python3
"""Replay a user's first-minute build order with a scripted bot.

Pipeline:
  1. Extract the replay (.sdfz) to a JSONL of observations + per-frame
     commands. Cached: if <replay>.extracted.jsonl already exists it's
     reused (pass --reparse to force re-extract).
  2. Parse the JSONL + demo header to identify
       (a) the human player's team and faction,
       (b) the player's commander spawn position, and
       (c) every command issued to that commander in the first N minutes
           (default 1).
  3. Launch a fresh game via BarEnv vs NullAI, with the bot spawned at
     the captured position and faction.
  4. Issue each captured command at its original frame (strict timing).

The fresh match produces its own .sdfz in BAR's demos folder; open it in
Chobby's replay viewer to eyeball whether the bot's build matches yours.

Usage:
    python examples/replay_user_build.py \\
        --replay "C:/Users/malco/AppData/Local/Programs/Beyond-All-Reason/data/demos/<file>.sdfz"
    python examples/replay_user_build.py --replay ... --minutes 1
    python examples/replay_user_build.py --replay ... --team 0     # override
    python examples/replay_user_build.py --replay ... --reparse    # ignore caches
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env import BarEnv  # noqa: E402
from bar_env.config import EXTRACTED_DIR  # noqa: E402
from bar_env.env import EnvConfig  # noqa: E402
from bar_env.replay import (  # noqa: E402
    ReplayExtractorConfig,
    extract,
    iter_jsonl,
)
from bar_env.start_script import StartScriptConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Demo header parsing
# ---------------------------------------------------------------------------
#
# We need three things out of the demo's startscript header:
#   - which team the human player is on  (PLAYERN.team)
#   - what faction each team is using    (TEAMN.side)
#   - which map (sanity check)
#
# The startscript is plain TDF text in the first ~64KB of the gzipped demo.
# Recoil's parser is loose; we just regex for the fields we need.


_SECTION_RE = re.compile(
    rb"\[\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\]\s*\{(?P<body>[^}]*)\}",
    re.DOTALL,
)


def _kv(body: bytes, key: str) -> str | None:
    """Pull a single `key=value;` out of a TDF section body."""
    m = re.search(
        rb"\b" + re.escape(key.encode()) + rb"\s*=\s*([^;\r\n]+);",
        body,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).decode("utf-8", errors="ignore").strip()


@dataclass
class DemoHeader:
    map_name: str | None = None
    game_type: str | None = None
    teams: dict[int, dict[str, Any]] = field(default_factory=dict)   # team_id -> {side, allyteam, start_pos_x/z}
    players: list[dict[str, Any]] = field(default_factory=list)      # [{name, team}, ...]
    ais: list[dict[str, Any]] = field(default_factory=list)          # [{shortname, team}, ...]


def parse_demo_header(sdfz_path: Path) -> DemoHeader:
    with gzip.open(sdfz_path, "rb") as f:
        head = f.read(65536)
    hdr = DemoHeader()
    # Top-level GAME fields
    hdr.map_name = _kv(head, "MapName")
    hdr.game_type = _kv(head, "GameType")
    for m in _SECTION_RE.finditer(head):
        name = m.group("name").decode("ascii").upper()
        body = m.group("body")
        if name.startswith("TEAM"):
            try:
                tid = int(name[4:])
            except ValueError:
                continue
            side = _kv(body, "Side") or _kv(body, "side")
            allyteam = _kv(body, "AllyTeam") or _kv(body, "allyteam")
            sx = _kv(body, "StartPosX")
            sz = _kv(body, "StartPosZ")
            hdr.teams[tid] = {
                "side": side,
                "allyteam": int(allyteam) if allyteam and allyteam.isdigit() else None,
                "start_pos_x": float(sx) if sx else None,
                "start_pos_z": float(sz) if sz else None,
            }
        elif name.startswith("PLAYER"):
            n = _kv(body, "Name")
            t = _kv(body, "Team")
            hdr.players.append({
                "name": n,
                "team": int(t) if t and t.isdigit() else None,
                "section": name,
            })
        elif name.startswith("AI"):
            t = _kv(body, "Team")
            sn = _kv(body, "ShortName")
            hdr.ais.append({
                "shortname": sn,
                "team": int(t) if t and t.isdigit() else None,
                "section": name,
            })
    return hdr


def pick_human_team(hdr: DemoHeader) -> int | None:
    """The human player's team -- prefer the player who isn't an AI section."""
    ai_teams = {a["team"] for a in hdr.ais if a["team"] is not None}
    for p in hdr.players:
        if p["team"] is None:
            continue
        if p["team"] in ai_teams:
            continue
        return p["team"]
    # Fallback: first player
    for p in hdr.players:
        if p["team"] is not None:
            return p["team"]
    return None


def side_to_player_side(side_str: str | None) -> str:
    """Normalize a TDF side string to StartScriptConfig.player_side."""
    if not side_str:
        return "Armada"
    s = side_str.strip().lower()
    if s in ("armada", "arm"):
        return "Armada"
    if s in ("cortex", "cor"):
        return "Cortex"
    if s in ("legion", "leg"):
        return "Legion"
    # Unknown -- pass through as-is; the engine will tell us if it's wrong.
    return side_str.strip()


def opposite_side(side: str) -> str:
    if side == "Armada":
        return "Cortex"
    if side == "Cortex":
        return "Armada"
    return "Cortex"


# ---------------------------------------------------------------------------
# Build-order extraction from JSONL
# ---------------------------------------------------------------------------


@dataclass
class BuildOrder:
    replay_path: str
    map_name: str | None
    player_team: int
    player_side: str
    spawn_x: float
    spawn_y: float
    spawn_z: float
    commander_def_id: int
    commander_replay_unit_id: int
    minutes_captured: float
    # Each command is the raw widget-observed event, normalized to our action
    # vocabulary: {frame, cmd_id, params, options}. We replay them in-order.
    commands: list[dict[str, Any]] = field(default_factory=list)
    # Bookkeeping / sanity stats
    n_observations_scanned: int = 0
    n_commands_other_units: int = 0
    n_commands_other_teams: int = 0


def extract_build_order(
    jsonl_path: Path,
    sdfz_path: Path,
    *,
    team_override: int | None,
    minutes: float,
    fps: int = 30,
) -> BuildOrder:
    hdr = parse_demo_header(sdfz_path)
    team_id = team_override if team_override is not None else pick_human_team(hdr)
    if team_id is None:
        raise RuntimeError(
            "Could not identify the human player's team from the demo header. "
            "Pass --team N explicitly."
        )

    side = side_to_player_side(hdr.teams.get(team_id, {}).get("side"))
    frame_limit = int(round(minutes * 60 * fps))

    # 1) Find the commander on the first observation that has units for this team.
    commander_uid: int | None = None
    commander_def: int = 0
    obs_spawn_x = obs_spawn_y = obs_spawn_z = 0.0
    n_obs = 0
    for obs in iter_jsonl(jsonl_path):
        n_obs += 1
        teams = obs.get("teams") or {}
        t = teams.get(str(team_id))
        if t is None:
            continue
        units = t.get("units") or []
        if not units:
            continue
        # Lowest-id unit at the earliest observation is the commander.
        first = min(units, key=lambda u: u.get("id", 1 << 30))
        commander_uid = int(first["id"])
        commander_def = int(first.get("def") or 0)
        obs_spawn_x = float(first.get("x") or 0.0)
        obs_spawn_y = float(first.get("y") or 0.0)
        obs_spawn_z = float(first.get("z") or 0.0)
        break

    if commander_uid is None:
        raise RuntimeError(
            f"No observations found containing team {team_id} units. "
            f"Available teams in the JSONL may differ -- run "
            f"examples/analyze_replay.py to see what's there."
        )

    # Prefer the demo header's StartPosX/StartPosZ when present -- that's the
    # exact lobby-resolved spawn, while the first observation captures the
    # commander up to one step_interval (~1s of game time) after spawn,
    # by which point it may already have moved.
    hdr_team = hdr.teams.get(team_id, {})
    hdr_x = hdr_team.get("start_pos_x")
    hdr_z = hdr_team.get("start_pos_z")
    if hdr_x is not None and hdr_z is not None:
        spawn_x, spawn_z = float(hdr_x), float(hdr_z)
        # Keep y from the obs -- the header doesn't include map height.
        spawn_y = obs_spawn_y
    else:
        spawn_x, spawn_y, spawn_z = obs_spawn_x, obs_spawn_y, obs_spawn_z

    # 2) Collect commands issued to that commander in the first N minutes.
    bo = BuildOrder(
        replay_path=str(sdfz_path),
        map_name=hdr.map_name,
        player_team=team_id,
        player_side=side,
        spawn_x=spawn_x, spawn_y=spawn_y, spawn_z=spawn_z,
        commander_def_id=commander_def,
        commander_replay_unit_id=commander_uid,
        minutes_captured=minutes,
        n_observations_scanned=n_obs,
    )
    seen_keys: set[tuple[int, int, tuple[float, ...]]] = set()
    for obs in iter_jsonl(jsonl_path):
        obs_frame = int(obs.get("frame") or 0)
        # Stop scanning observations once we're well past the time window;
        # there's nothing left to find.
        if obs_frame > frame_limit + 60:
            break
        for c in obs.get("commands_this_frame") or []:
            # Each captured command has its own `frame` field (the engine
            # frame when widget:UnitCommand fired). That's authoritative;
            # obs_frame just tells us when the widget flushed the buffer.
            cmd_frame = int(c.get("frame") or obs_frame)
            if cmd_frame >= frame_limit:
                continue
            unit = c.get("unit")
            team = c.get("team")
            if team is not None and team != team_id:
                bo.n_commands_other_teams += 1
                continue
            if unit != commander_uid:
                bo.n_commands_other_units += 1
                continue
            cmd_id = c.get("cmd_id")
            params = list(c.get("params") or [])
            options = c.get("options") or {}
            # Spring re-broadcasts the same player command multiple times in
            # the demo stream (selection refreshes, etc.). Dedup on the tuple
            # (frame, cmd_id, params).
            key = (cmd_frame, int(cmd_id) if cmd_id is not None else 0,
                   tuple(float(p) for p in params))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bo.commands.append({
                "frame": cmd_frame,
                "cmd_id": int(cmd_id) if cmd_id is not None else 0,
                "params": params,
                "options": options,
            })
    bo.commands.sort(key=lambda c: c["frame"])
    return bo


# ---------------------------------------------------------------------------
# Replay the build order in a fresh game
# ---------------------------------------------------------------------------


def cmd_name(cid: int) -> str:
    """Human-readable label for a Spring command id."""
    if cid < 0:
        return f"BUILD(def={-cid})"
    return {
        0: "STOP", 5: "WAIT", 10: "MOVE", 14: "FIGHT", 15: "PATROL",
        16: "AREA_ATTACK", 20: "ATTACK", 25: "GUARD",
        40: "REPAIR", 45: "FIRE_STATE", 50: "MOVE_STATE",
        65: "SELFD", 75: "LOAD_UNITS", 80: "UNLOAD_UNITS", 85: "ONOFF",
        90: "RECLAIM", 95: "CLOAK", 100: "STOCKPILE", 105: "MANUALFIRE",
        115: "REPEAT", 120: "TRAJECTORY", 125: "RESURRECT", 130: "CAPTURE",
    }.get(cid, f"CMD_{cid}")


def command_to_action(c: dict[str, Any], commander_uid: int) -> dict[str, Any] | None:
    """Convert a captured replay command into the dict the widget accepts.

    Returns None if the command can't be expressed (rare for the commands a
    human commander actually issues in the first minute -- mostly BUILD/MOVE).
    """
    cid = int(c["cmd_id"])
    params = list(c.get("params") or [])
    options = dict(c.get("options") or {})
    # Strip our private bookkeeping flags ('bits' from int-bitmask fallbacks).
    options.pop("bits", None)
    if cid < 0:
        # BUILD -- cmd_id is -def_id; params are [x, y, z, facing].
        return {
            "unit": int(commander_uid),
            "cmd": "BUILD",
            "def": int(-cid),
            "params": [float(p) for p in params],
            "options": options,
        }
    # Other named commands: just pass the numeric cmd_id through; the widget
    # accepts ints directly via resolve_command's number branch.
    return {
        "unit": int(commander_uid),
        "cmd": int(cid),
        "params": [float(p) for p in params],
        "options": options,
    }


def play_build_order(bo: BuildOrder, *, port: int, max_steps: int) -> int:
    # Place the bot on the same StartPos as the human player. Mirror the
    # enemy across the map midpoint as a default (Comet Catcher Remake is
    # symmetric); user can tweak by editing StartScriptConfig later.
    script_cfg = StartScriptConfig(
        map_name=bo.map_name or "Comet Catcher Remake 1.8",
        player_name="PyEnv-BuildOrder",
        player_side=bo.player_side,
        enemy_side=opposite_side(bo.player_side),
        enemy_ai_short_name="NullAI",
        enemy_ai_name="NullAI",
        start_pos_type=3,  # chooseBeforeGame -- use TEAMN.StartPosX/Z
        player_start_pos_x=bo.spawn_x,
        player_start_pos_z=bo.spawn_z,
    )
    # We don't actually know the map dimensions until reset() returns the
    # first obs. Defer setting the enemy mirror until then.

    cfg = EnvConfig(
        map_name=script_cfg.map_name,
        port=port,
        step_interval=30,         # 1s of sim time per env step
        max_steps=max_steps,
        connect_timeout_s=180.0,
        step_timeout_s=60.0,
        script_config=script_cfg,
        speed=20,
    )

    print("== replay_user_build: launching bot match ==")
    print(f"  map              : {script_cfg.map_name}")
    print(f"  player side      : {script_cfg.player_side} vs {script_cfg.enemy_side}")
    print(f"  spawn (x, z)     : ({bo.spawn_x:.0f}, {bo.spawn_z:.0f})")
    print(f"  commands queued  : {len(bo.commands)} (first {bo.minutes_captured:.1f} min)")
    print(f"  max env steps    : {max_steps}")
    print()

    queue = list(bo.commands)  # already sorted by frame
    queue_idx = 0
    sent_total = 0
    failed_total = 0

    with BarEnv(cfg) as env:
        obs = env.reset()
        my_units = obs.get("my_units") or []
        if not my_units:
            print("WARNING: first observation has no units for the bot's team. "
                  "Cannot identify commander; aborting.")
            return 1
        commander = min(my_units, key=lambda u: u.get("id", 1 << 30))
        commander_uid = int(commander["id"])
        commander_def = int(commander.get("def") or 0)
        print(f"  bot commander    : id={commander_uid} def={commander_def} "
              f"@ ({commander.get('x'):.0f}, {commander.get('z'):.0f})")
        if abs(commander.get("x", 0) - bo.spawn_x) > 50 or \
           abs(commander.get("z", 0) - bo.spawn_z) > 50:
            print("  [warn] bot commander spawned > 50 units away from the "
                  "captured position. The engine may have ignored our "
                  "StartPosX/Z (try start_pos_type=2 in start_script.py).")

        for step_idx in range(max_steps):
            current_frame = int(obs.get("frame") or 0)
            # Pop everything whose original-replay frame is <= current_frame.
            to_send: list[dict[str, Any]] = []
            while queue_idx < len(queue) and queue[queue_idx]["frame"] <= current_frame:
                act = command_to_action(queue[queue_idx], commander_uid)
                if act is None:
                    failed_total += 1
                else:
                    to_send.append(act)
                queue_idx += 1
            if to_send:
                sent_total += len(to_send)
                names = [cmd_name(int(c.get("cmd_id", 0))) if "cmd_id" in c
                         else cmd_name(-int(c.get("def", 0)) if c.get("cmd") == "BUILD"
                                       else int(c.get("cmd", 0)))
                         for c in to_send]
                print(f"  step {step_idx:>3d} frame={current_frame:>4d} "
                      f"sent {len(to_send):>2d}: {', '.join(names)}")
            obs, done = env.step(to_send)
            if done:
                print(f"  -> terminal observation, result={obs.get('result')}")
                break
            if queue_idx >= len(queue) and not to_send:
                # Build order exhausted; let the game continue for a few more
                # steps then quit so the demo gets written and you can review.
                pass

        print(f"\n  sent {sent_total} commands "
              f"({failed_total} unrepresentable, "
              f"{len(queue) - queue_idx} not yet dispatched).")

    demo = env.demo_path()
    if demo is not None:
        size_kb = demo.stat().st_size / 1024
        print(f"\nbot replay saved : {demo}  ({size_kb:,.0f} KB)")
        print("Open Chobby -> Watch -> Replays to compare with the original.")
    else:
        print("\n(no replay file was produced; check the engine log)")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def jsonl_path_for(replay: Path, out_dir: Path) -> Path:
    """Where the extracted JSONL for `replay` lives inside `out_dir`."""
    return out_dir / (replay.stem + ".extracted.jsonl")


def build_order_path_for(replay: Path, out_dir: Path) -> Path:
    """Where the parsed build-order JSON for `replay` lives inside `out_dir`."""
    return out_dir / (replay.stem + ".build_order.json")


def maybe_extract(replay: Path, *, reparse: bool, out_dir: Path) -> Path:
    """Return the path to the .extracted.jsonl, running extraction if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = jsonl_path_for(replay, out_dir)
    if jsonl.exists() and jsonl.stat().st_size > 0 and not reparse:
        print(f"Reusing cached extraction: {jsonl} "
              f"({jsonl.stat().st_size // 1024} KB)")
        return jsonl
    print(f"Extracting replay -> {jsonl}")
    cfg = ReplayExtractorConfig(
        replay_path=replay,
        output_path=jsonl,
        step_interval=30,           # 1Hz obs is plenty; widget:UnitCommand
                                    # records each command's exact frame in
                                    # its own `frame` field independent of
                                    # how often we flush obs.
        max_steps=0,
        speed=20,
    )
    summary = extract(cfg)
    print(f"  extraction: {summary}")
    if summary.get("frames_written", 0) == 0:
        raise RuntimeError("Extraction produced 0 frames; replay may be broken.")
    return jsonl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True,
                   help="Path to the .sdfz to learn the build order from.")
    p.add_argument("--team", type=int, default=None,
                   help="Team id of the human player (default: auto-detect "
                        "from the demo header).")
    p.add_argument("--minutes", type=float, default=1.0,
                   help="How many minutes of the replay to capture (default 1).")
    p.add_argument("--reparse", action="store_true",
                   help="Re-run extraction even if a cached JSONL exists.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bot-max-steps", type=int, default=120,
                   help="How many env steps (= seconds) to run the bot match "
                        "for. Default 120 = 2 min (1 min build + 1 min to "
                        "see the result).")
    p.add_argument("--out-dir", type=Path, default=EXTRACTED_DIR,
                   help=f"Folder for extracted JSONL and parsed build-order "
                        f"JSON. Defaults to {EXTRACTED_DIR} (override globally "
                        f"with $BAR_PY_ENV_EXTRACTED_DIR).")
    p.add_argument("--save-build-order", type=Path, default=None,
                   help="Optional: explicit path for the parsed build order "
                        "JSON. Defaults to --out-dir/<replay>.build_order.json.")
    p.add_argument("--no-save-build-order", action="store_true",
                   help="Skip writing the parsed build-order JSON.")
    p.add_argument("--parse-only", action="store_true",
                   help="Stop after parsing -- don't launch the bot match. "
                        "Useful for debugging the parse.")
    args = p.parse_args()

    if not args.replay.exists():
        sys.stderr.write(f"Replay not found: {args.replay}\n")
        return 1

    jsonl = maybe_extract(args.replay, reparse=args.reparse, out_dir=args.out_dir)

    # Show demo-header summary so the user can confirm team selection.
    hdr = parse_demo_header(args.replay)
    print()
    print("=== demo header ===")
    print(f"  map      : {hdr.map_name}")
    print(f"  gametype : {hdr.game_type}")
    print(f"  teams    :")
    for tid in sorted(hdr.teams):
        t = hdr.teams[tid]
        print(f"    team {tid}: side={t['side']:<8} "
              f"ally={t['allyteam']} "
              f"start=({t['start_pos_x']}, {t['start_pos_z']})")
    print(f"  players  :")
    for pl in hdr.players:
        print(f"    {pl['section']}: name={pl['name']:<20} team={pl['team']}")
    if hdr.ais:
        print(f"  AIs      :")
        for ai in hdr.ais:
            print(f"    {ai['section']}: short={ai['shortname']:<12} team={ai['team']}")

    print()
    bo = extract_build_order(
        jsonl, args.replay,
        team_override=args.team,
        minutes=args.minutes,
    )
    print("=== parsed build order ===")
    print(f"  player team       : {bo.player_team} (side: {bo.player_side})")
    print(f"  commander         : replay-uid={bo.commander_replay_unit_id} "
          f"def={bo.commander_def_id}")
    print(f"  spawn (x, y, z)   : ({bo.spawn_x:.0f}, {bo.spawn_y:.0f}, {bo.spawn_z:.0f})")
    print(f"  commands captured : {len(bo.commands)} in {bo.minutes_captured:.1f} min")
    print(f"  obs scanned       : {bo.n_observations_scanned}")
    print(f"  ignored (other teams) : {bo.n_commands_other_teams}")
    print(f"  ignored (other units) : {bo.n_commands_other_units}")
    print()
    print("  first 10 commands (frame -> cmd):")
    for c in bo.commands[:10]:
        print(f"    f={c['frame']:>4d}  {cmd_name(c['cmd_id']):<14}  params={c['params']}")
    if len(bo.commands) > 10:
        print(f"    ... ({len(bo.commands) - 10} more)")

    # Default: always write the parsed build order JSON next to the JSONL,
    # so the user has a human-readable record. --no-save-build-order opts out;
    # --save-build-order PATH overrides the location.
    if not args.no_save_build_order:
        bo_path = args.save_build_order or build_order_path_for(args.replay, args.out_dir)
        bo_path.parent.mkdir(parents=True, exist_ok=True)
        bo_path.write_text(json.dumps(asdict(bo), indent=2), encoding="utf-8")
        print(f"\nWrote build order: {bo_path}")

    if args.parse_only:
        return 0

    print()
    return play_build_order(bo, port=args.port, max_steps=args.bot_max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
