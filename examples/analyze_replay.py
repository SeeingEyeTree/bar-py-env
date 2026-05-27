#!/usr/bin/env python3
"""Quick stats on an extracted replay JSONL file.

Tells you: how many frames, how many commands, command type histogram, how
many distinct units, how many of those frames have a Commander present.

Usage:
    python examples/analyze_replay.py <path-to-jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


# Spring command IDs we care about (subset; full list lives in
# https://github.com/beyond-all-reason/RecoilEngine/blob/master/rts/Game/UI/Command.h)
CMD_NAMES = {
    0: "STOP", 5: "WAIT", 7: "TIMEWAIT", 10: "MOVE", 14: "FIGHT", 15: "PATROL",
    16: "AREA_ATTACK", 20: "ATTACK", 21: "ATTACK_GROUND", 25: "GUARD",
    34: "AISELECT", 35: "STOCKPILE", 40: "REPAIR", 45: "FIRE_STATE",
    50: "MOVE_STATE", 55: "SETBASE", 60: "INTERNAL", 65: "SELFD", 70: "SET_WANTED_MAX_SPEED",
    75: "LOAD_UNITS", 80: "LOAD_UNITS_2", 90: "UNLOAD_UNIT", 95: "UNLOAD_UNITS",
    105: "ONOFF", 110: "RECLAIM", 120: "CLOAK", 125: "STOCKPILE", 130: "MANUALFIRE",
    135: "RESTORE", 140: "REPEAT", 145: "TRAJECTORY", 150: "RESURRECT",
    155: "CAPTURE", 160: "AUTOREPAIRLEVEL", 165: "LOOPBACKATTACK",
    170: "IDLEMODE", 175: "FAILED",
}

# Unit-def IDs we tend to see in early BAR games. Replays use the def IDs from
# the BAR mod's UnitDefs.lua; the int below is just for sanity-printing.
COMMON_UNIT_DEFS = {
    49: "Commander",     # ARMCOM / CORCOM (varies per faction; ID may differ)
    149: "Mex (T1)",
    209: "Solar (T1)",
}


def cmd_name(cid: int) -> str:
    if cid < 0:
        return f"BUILD(-{cid})"
    return CMD_NAMES.get(cid, f"CMD_{cid}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: analyze_replay.py <path-to-jsonl>\n")
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        sys.stderr.write(f"File not found: {path}\n")
        return 1

    n_frames = 0
    n_commands_total = 0
    frames_with_command = 0
    cmd_hist: Counter = Counter()
    unit_def_hist: Counter = Counter()
    teams_seen: set[str] = set()
    last_frame = 0
    first_frame = None
    max_unit_count_per_team: dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obs = json.loads(line)
            n_frames += 1
            frame = obs.get("frame", 0)
            if first_frame is None:
                first_frame = frame
            last_frame = frame
            cmds = obs.get("commands_this_frame") or []
            n_commands_total += len(cmds)
            if cmds:
                frames_with_command += 1
            for c in cmds:
                cmd_hist[c.get("cmd_id")] += 1
                unit_def_hist[c.get("def")] += 1
            for tid, t in (obs.get("teams") or {}).items():
                teams_seen.add(tid)
                n = len(t.get("units") or [])
                max_unit_count_per_team[tid] = max(max_unit_count_per_team.get(tid, 0), n)

    duration_frames = (last_frame - (first_frame or 0))
    duration_s = duration_frames / 30.0
    print(f"file               : {path.name}")
    print(f"size               : {path.stat().st_size / 1024:.1f} KB")
    print(f"observations       : {n_frames}")
    print(f"first / last frame : {first_frame} / {last_frame}")
    print(f"game duration      : {duration_s:.1f} s sim time "
          f"(~{duration_s / 60:.1f} min)")
    print(f"teams              : {sorted(teams_seen)}")
    print(f"max units per team : {max_unit_count_per_team}")
    print(f"commands total     : {n_commands_total} "
          f"(across {frames_with_command} frames with at least one command)")
    print()
    print("top command IDs:")
    for cid, n in cmd_hist.most_common(15):
        print(f"  {n:6d}  cmd_id={cid:<5d}  {cmd_name(cid)}")
    print()
    print("top issuing unit defs (the unit that received the command):")
    for did, n in unit_def_hist.most_common(10):
        label = COMMON_UNIT_DEFS.get(did, "")
        print(f"  {n:6d}  def={did:<5d}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
