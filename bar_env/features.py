"""Feature extraction for behavioral cloning training and inference.

Shared by scripts/build_bc_dataset.py (offline) and scripts/bc_bot.py (live).

Observation format expected by extract_scalars / build_spatial_grid:
  The "replay" god's-eye format used by the JSONL extractor:
    obs["frame"]               int
    obs["teams"]["0"]["metal"] float   (team keyed by string id)
    obs["teams"]["0"]["units"] list[unit_dict]

  For live inference, convert via live_obs_to_frame() in bc_bot.py first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_H: int = 48   # rows  (Z axis)
GRID_W: int = 64   # cols  (X axis)
N_SCALARS: int = 41
N_CLUSTERS: int = 20

ACTION_NAMES: list[str] = [
    "NO_OP",        # 0
    "BUILD_ECO",    # 1  mex / solar / fusion / storage
    "BUILD_LAB",    # 2  factory / lab building
    "BUILD_UNIT",   # 3  mobile unit queued from factory
    "BUILD_DEFENSE",# 4  static defense
    "BUILD_OTHER",  # 5  any other construction
    "MOVE",         # 6  reposition (non-combat)
    "ATTACK",       # 7  aggressive move or direct attack
    "SUPPORT",      # 8  guard / repair / reclaim
]
N_ACTIONS: int = len(ACTION_NAMES)

# Spring cmd_ids to ignore entirely (setup / bookkeeping, not strategic)
_SKIP_CMDS: frozenset[int] = frozenset({50, 45, 85, 2, 1, -1, -2, 5})

# Unit name substrings for eco / lab / defense classification
_ECO_KW   = ("mex", "solar", "wind", "fus", "mstor", "estor", "stor",
              "geo", "aap", "tidal", "moho", "advsol", "advfus")
_LAB_KW   = ("lab", "plant", "asy", "yard", "fac", "hive", "warp")
_DEF_KW   = ("guard", "beamer", "claw", "tower", "flak", "punisher",
              "sentinel", "staticjam", "staticrad", "dgun", "nuke",
              "antinuke", "wall", "shield", "emitter", "hlt", "hllt",
              "vhlt", "llt", "mllt", "fhlt", "plasma")

# Cluster IDs in the order they map to feature indices 0-19
# T1: 7 clusters (100-106), T2: 9 (200-208), T3: 3 (300-302), Special: 1 (900+)
_CLUSTER_ORDER: list[int] = [
    100, 101, 102, 103, 104, 105, 106,          # T1  indices 0-6
    200, 201, 202, 203, 204, 205, 206, 207, 208, # T2  indices 7-15
    300, 301, 302,                               # T3  indices 16-18
    900,                                         # Special (901,902 also map here) index 19
]
assert len(_CLUSTER_ORDER) == N_CLUSTERS
_CLUSTER_IDX: dict[int, int] = {c: i for i, c in enumerate(_CLUSTER_ORDER)}
# Map 901/902 (and any other 900+) to slot 19
_CLUSTER_IDX[901] = 19
_CLUSTER_IDX[902] = 19


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_unitdef_lookup(bar_env_dir: Path) -> dict[int, str]:
    """Return {def_id: unit_name} by inverting data/unitdef_names.json."""
    p = bar_env_dir / "data" / "unitdef_names.json"
    raw: dict[str, int] = json.loads(p.read_text(encoding="utf-8"))
    return {int(v): k for k, v in raw.items()}


def load_cluster_lookup(bar_env_dir: Path) -> dict[str, int]:
    """Return {unit_name: cluster_id} from units/unit_categories_v3.json."""
    p = bar_env_dir / "units" / "unit_categories_v3.json"
    data: dict = json.loads(p.read_text(encoding="utf-8"))
    return data["assignments"]  # {name: cluster_id}


# ---------------------------------------------------------------------------
# Unit def classification
# ---------------------------------------------------------------------------

def _classify_build_by_name(name: str, is_mobile: bool) -> int:
    """Return action_type int for a BUILD command given the unit's name."""
    if is_mobile:
        return 3  # BUILD_UNIT
    n = name.lower()
    for kw in _ECO_KW:
        if kw in n:
            return 1  # BUILD_ECO
    for kw in _LAB_KW:
        if kw in n:
            return 2  # BUILD_LAB
    for kw in _DEF_KW:
        if kw in n:
            return 4  # BUILD_DEFENSE
    return 5  # BUILD_OTHER


# ---------------------------------------------------------------------------
# Static spatial features (computed once per game from the header)
# ---------------------------------------------------------------------------

def build_metal_spot_mask(map_meta: dict[str, Any]) -> np.ndarray:
    """Return (GRID_H, GRID_W) float32 mask: 1.0 at mex spots, 0 elsewhere."""
    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    map_w = float(map_meta.get("size_x", 8192))
    map_h = float(map_meta.get("size_z", 6144))
    for spot in map_meta.get("metal_spots", []):
        col = int(float(spot["x"]) / map_w * GRID_W)
        row = int(float(spot["z"]) / map_h * GRID_H)
        col = max(0, min(GRID_W - 1, col))
        row = max(0, min(GRID_H - 1, row))
        mask[row, col] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Per-frame scalar features
# ---------------------------------------------------------------------------

def _find_commander(units: list[dict]) -> dict | None:
    """Return the unit with the highest max_hp (commanders have 3700+ HP)."""
    if not units:
        return None
    best = max(units, key=lambda u: u.get("max_hp", 0))
    return best if best.get("max_hp", 0) >= 3000 else None


def _eco_slice(team: dict, f: np.ndarray, offset: int) -> None:
    """Fill 7 scalar features for one team's economy, starting at f[offset]."""
    m_storage = max(float(team.get("metal_storage", 1000)), 1.0)
    e_storage = max(float(team.get("energy_storage", 1000)), 1.0)
    m_income = float(team.get("metal_income", 0))
    e_income = float(team.get("energy_income", 0))
    m_pull   = float(team.get("metal_pull", 0))
    e_pull   = float(team.get("energy_pull", 0))
    f[offset + 0] = float(team.get("metal", 0)) / m_storage
    f[offset + 1] = float(team.get("energy", 0)) / e_storage
    f[offset + 2] = m_income / 50.0
    f[offset + 3] = e_income / 1000.0
    f[offset + 4] = max(0.0, m_pull - m_income) / 10.0
    f[offset + 5] = max(0.0, e_pull - e_income) / 100.0
    f[offset + 6] = len(team.get("units", [])) / 100.0


def extract_scalars(
    frame: dict[str, Any],
    my_team_id: str,
    last_frame: int,
    map_meta: dict[str, Any],
    name_to_cluster: dict[str, int],
    def_to_name: dict[int, str],
) -> np.ndarray:
    """Extract N_SCALARS-float feature vector for one team in one frame.

    Layout:
      [0]     frame_norm
      [1-7]   my eco (metal_ratio, energy_ratio, m_inc, e_inc, m_stall, e_stall, n_units)
      [8-27]  my cluster counts[0..19]  (count/10, capped at 3.0)
      [28-30] my commander (x/map_w, z/map_h, hp/max_hp)
      [31-37] enemy eco (same 7 fields)
      [38-40] enemy commander (x, z, hp)
    """
    teams = frame.get("teams", {})
    my  = teams.get(my_team_id, {})
    en_id = next((k for k in teams if k != my_team_id), None)
    en = teams.get(en_id, {}) if en_id else {}

    map_w = float(map_meta.get("size_x", 8192))
    map_h = float(map_meta.get("size_z", 6144))

    f = np.zeros(N_SCALARS, dtype=np.float32)

    f[0] = int(frame.get("frame", 0)) / max(last_frame, 1)

    _eco_slice(my, f, 1)
    _eco_slice(en, f, 31)

    # Cluster counts (indices 8-27)
    for u in my.get("units", []):
        name = def_to_name.get(int(u.get("def", 0)), "")
        cluster_id = name_to_cluster.get(name)
        if cluster_id is not None:
            idx = _CLUSTER_IDX.get(cluster_id)
            if idx is not None:
                f[8 + idx] = min(f[8 + idx] + 0.1, 3.0)

    # My commander (28-30)
    my_com = _find_commander(my.get("units", []))
    if my_com:
        f[28] = float(my_com.get("x", 0)) / map_w
        f[29] = float(my_com.get("z", 0)) / map_h
        f[30] = float(my_com.get("hp", 0)) / max(float(my_com.get("max_hp", 1)), 1.0)
    else:
        f[28] = f[29] = -1.0
        f[30] = 0.0

    # Enemy commander (38-40)
    en_com = _find_commander(en.get("units", []))
    if en_com:
        f[38] = float(en_com.get("x", 0)) / map_w
        f[39] = float(en_com.get("z", 0)) / map_h
        f[40] = float(en_com.get("hp", 0)) / max(float(en_com.get("max_hp", 1)), 1.0)
    else:
        f[38] = f[39] = -1.0
        f[40] = 0.0

    return f


# ---------------------------------------------------------------------------
# Per-frame spatial grid
# ---------------------------------------------------------------------------

def build_spatial_grid(
    frame: dict[str, Any],
    my_team_id: str,
    map_meta: dict[str, Any],
    metal_spot_mask: np.ndarray | None = None,
    def_to_name: dict[int, str] | None = None,
    mobile_names: set[str] | None = None,
) -> np.ndarray:
    """Build (7, GRID_H, GRID_W) float32 spatial feature tensor.

    Channels:
      0  metal spot mask (static, from metal_spot_mask arg)
      1  my unit density       (count/5 per cell)
      2  my unit HP            (sum HP / 5000 per cell)
      3  enemy unit density
      4  enemy unit HP
      5  my building density   (units not in mobile_names / 5)
      6  build-in-progress     (max bp value in cell)
    """
    teams = frame.get("teams", {})
    my   = teams.get(my_team_id, {})
    en_id = next((k for k in teams if k != my_team_id), None)
    en = teams.get(en_id, {}) if en_id else {}

    map_w = float(map_meta.get("size_x", 8192))
    map_h = float(map_meta.get("size_z", 6144))

    grid = np.zeros((7, GRID_H, GRID_W), dtype=np.float32)
    if metal_spot_mask is not None:
        grid[0] = metal_spot_mask

    def _rc(x: float, z: float) -> tuple[int, int]:
        col = max(0, min(GRID_W - 1, int(x / map_w * GRID_W)))
        row = max(0, min(GRID_H - 1, int(z / map_h * GRID_H)))
        return row, col

    for u in my.get("units", []):
        r, c = _rc(float(u.get("x", 0)), float(u.get("z", 0)))
        grid[1, r, c] = min(grid[1, r, c] + 0.2, 3.0)
        grid[2, r, c] = min(grid[2, r, c] + float(u.get("hp", 0)) / 5000.0, 5.0)
        bp = u.get("bp")
        if bp is not None:
            grid[6, r, c] = max(grid[6, r, c], float(bp))
        # Building heuristic: not in mobile_names set
        if mobile_names is not None and def_to_name is not None:
            name = def_to_name.get(int(u.get("def", 0)), "")
            if name not in mobile_names:
                grid[5, r, c] = min(grid[5, r, c] + 0.2, 3.0)

    for u in en.get("units", []):
        r, c = _rc(float(u.get("x", 0)), float(u.get("z", 0)))
        grid[3, r, c] = min(grid[3, r, c] + 0.2, 3.0)
        grid[4, r, c] = min(grid[4, r, c] + float(u.get("hp", 0)) / 5000.0, 5.0)

    return grid


# ---------------------------------------------------------------------------
# Command classification (action label extraction)
# ---------------------------------------------------------------------------

def classify_commands(
    commands: list[dict[str, Any]],
    team_id: str | int,
    def_to_name: dict[int, str],
    mobile_names: set[str],
    map_meta: dict[str, Any],
) -> tuple[int, int | None, tuple[float, float] | None]:
    """Classify a frame's commands into (action_type, def_id|None, (x_norm,z_norm)|None).

    Returns (0, None, None) = NO_OP when no relevant commands exist.

    BUILD detection:
      cmd_id < 0 and len(params) == 0  → factory queuing a unit → BUILD_UNIT (3)
      cmd_id < 0 and len(params) >= 3  → construction at position → eco/lab/def/other
    """
    map_w = float(map_meta.get("size_x", 8192))
    map_h = float(map_meta.get("size_z", 6144))
    team_int = int(team_id)

    relevant = [
        c for c in commands
        if int(c.get("team", -1)) == team_int
        and int(c.get("cmd_id", 0)) not in _SKIP_CMDS
    ]
    if not relevant:
        return 0, None, None

    type_counts: dict[int, int] = {}
    type_first_def: dict[int, int] = {}
    type_first_pos: dict[int, tuple[float, float]] = {}

    for cmd in relevant:
        cid    = int(cmd.get("cmd_id", 0))
        params = cmd.get("params") or []

        if cid < 0:
            def_id = -cid
            name   = def_to_name.get(def_id, "")
            # factory queue has no position params; construction has x,y,z,facing
            if len(params) == 0:
                atype = 3  # BUILD_UNIT (factory queue)
            else:
                is_mobile = name in mobile_names
                atype = _classify_build_by_name(name, is_mobile)

            type_counts[atype] = type_counts.get(atype, 0) + 1
            if atype not in type_first_def:
                type_first_def[atype] = def_id
            if atype not in type_first_pos and len(params) >= 3:
                type_first_pos[atype] = (
                    float(params[0]) / map_w,
                    float(params[2]) / map_h,
                )

        elif cid in (10, 15):            # MOVE, PATROL → non-aggressive move
            atype = 6
            type_counts[atype] = type_counts.get(atype, 0) + 1
            if atype not in type_first_pos and len(params) >= 3:
                type_first_pos[atype] = (float(params[0]) / map_w, float(params[2]) / map_h)

        elif cid in (14, 20, 21, 16):   # FIGHT, ATTACK, AREA_ATTACK, AREA_ATTACK_ID
            atype = 7
            type_counts[atype] = type_counts.get(atype, 0) + 1
            if atype not in type_first_pos and len(params) >= 3:
                type_first_pos[atype] = (float(params[0]) / map_w, float(params[2]) / map_h)

        elif cid in (25, 40, 90):       # GUARD, REPAIR, RECLAIM
            atype = 8
            type_counts[atype] = type_counts.get(atype, 0) + 1

    if not type_counts:
        return 0, None, None

    best = max(type_counts, key=lambda t: type_counts[t])
    return best, type_first_def.get(best), type_first_pos.get(best)
