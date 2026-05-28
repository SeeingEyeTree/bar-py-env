"""Replay visualization utilities for bar-py-env.

Renders BAR replay observations (from JSONL) as BGR images suitable for
writing to video. Units are colored by cluster (from bar_unit_clusters.json)
with team identity shown via a 2-pixel outer ring.

Coordinate system
-----------------
Spring world: X = 0..map_w (West->East), Z = 0..map_h (North->South), Y = height.
Image: pixel (0,0) = top-left = NW corner of the map.

Usage
-----
    from bar_env.vis import ReplayRenderer
    r = ReplayRenderer(8192, 6144)
    img = r.render_frame(obs)   # np.ndarray BGR (768, 1024, 3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CLUSTERS_PATH = _DATA_DIR / "bar_unit_clusters.json"
_UNITDEF_PATH  = _DATA_DIR / "unitdef_names.json"
_COSTS_PATH    = _DATA_DIR / "unitdef_costs.json"

# ---------------------------------------------------------------------------
# Load cluster + unitdef tables at module import
# ---------------------------------------------------------------------------

def _load_tables() -> tuple[dict[int, int], dict[int, dict], dict[int, float]]:
    """Return (DEF_TO_CLUSTER, CLUSTER_META, DEF_TO_EQUIV_COST).

    DEF_TO_CLUSTER:    unitDefID (int) -> cluster_id (int)
    CLUSTER_META:      cluster_id (int) -> {"name": str, "color": (B,G,R), "radius": int}
    DEF_TO_EQUIV_COST: unitDefID (int) -> metal-equivalent cost (float)
                       equiv = metalcost + energycost / 60
    """
    with open(_CLUSTERS_PATH, encoding="utf-8") as f:
        cluster_data = json.load(f)
    with open(_UNITDEF_PATH, encoding="utf-8") as f:
        name_to_id: dict[str, int] = json.load(f)
    costs_raw: dict[str, dict] = {}
    if _COSTS_PATH.exists():
        with open(_COSTS_PATH, encoding="utf-8") as f:
            # keys are string def_ids in JSON
            costs_raw = {int(k): v["equiv"] for k, v in json.load(f).items()}

    # BGR colors for each cluster (20 clusters, visually distinct)
    _CLUSTER_COLORS: dict[int, tuple[int, int, int]] = {
        100: (0,   180, 255),   # T1 light raiders      orange
        101: (0,   255, 180),   # T1 anti-air           yellow-green
        102: (80,   80, 255),   # T1 short artillery    red
        103: (0,   140, 255),   # T1 mid mobiles        amber
        104: (0,   100, 200),   # T1 rocket bots        dark orange
        106: (0,    50, 220),   # T1 heavy assault      deep red
        200: (255,   0, 200),   # T2 EMP                magenta
        201: (200,   0, 255),   # T2 mid mobiles        purple-red
        202: (255, 100, 200),   # T2 anti-air           pink
        203: (0,   255,  50),   # Crawling bombs        lime
        204: (150, 255, 150),   # Mobile anti-nuke      pale green
        205: (0,     0, 255),   # T2 heavy assault      bright red
        206: (255, 200,   0),   # T2 amphibious         bright blue
        208: (200,  50,  50),   # T2 long-range arty    dark teal
        300: (255, 255,   0),   # T3 experimental       bright cyan
        301: (180,   0, 180),   # T3 big units          magenta-purple
        302: (255, 100,   0),   # T3 long-range         bright blue
        900: (200, 200, 200),   # Behe                  bright white
        901: (150,   0, 255),   # Missile Truck         violet
        902: (0,   200, 150),   # Specific Response     teal
    }
    # Radii by cluster tier
    _CLUSTER_RADII: dict[int, int] = {
        100: 3, 101: 3, 102: 3, 103: 3, 104: 3, 106: 3,   # T1
        200: 4, 201: 4, 202: 4, 203: 3, 204: 4, 205: 4,   # T2
        206: 4, 208: 4,
        300: 5, 301: 5, 302: 5,                            # T3
        900: 6, 901: 5, 902: 4,                            # specials
    }

    cluster_meta: dict[int, dict] = {}
    for c in cluster_data["clusters"]:
        cid = c["id"]
        cluster_meta[cid] = {
            "name":   c["name"],
            "color":  _CLUSTER_COLORS.get(cid, (128, 128, 128)),
            "radius": _CLUSTER_RADII.get(cid, 3),
            "codes":  c["codes"],
        }

    def_to_cluster: dict[int, int] = {}
    for c in cluster_data["clusters"]:
        cid = c["id"]
        for code in c["codes"]:
            def_id = name_to_id.get(code)
            if def_id is not None:
                def_to_cluster[def_id] = cid

    return def_to_cluster, cluster_meta, costs_raw


DEF_TO_CLUSTER:    dict[int, int]   = {}
CLUSTER_META:      dict[int, dict]  = {}
DEF_TO_EQUIV_COST: dict[int, float] = {}

try:
    DEF_TO_CLUSTER, CLUSTER_META, DEF_TO_EQUIV_COST = _load_tables()
except FileNotFoundError as _e:
    import warnings
    warnings.warn(f"bar_env.vis: could not load unit data ({_e}). "
                  "Run scripts/gen_unitdef_map.py first.")

# ---------------------------------------------------------------------------
# Fallback colors + radii for unclustered units (buildings, commanders, etc.)
# ---------------------------------------------------------------------------

# (max_hp_threshold, BGR_color, radius)
_FALLBACK: list[tuple[float, tuple[int, int, int], int]] = [
    (3500.0, (220, 255, 255), 7),  # commander  — bright white-cyan
    (1000.0, (180,  80, 180), 5),  # factory    — purple
    ( 200.0, (150, 255, 150), 3),  # builder / constructor — pale green
    (  50.0, ( 50, 200,  50), 2),  # mex / energy — green
    (   0.0, ( 80,  80,  80), 2),  # structure  — grey
]

def _fallback_color_radius(max_hp: float) -> tuple[tuple[int, int, int], int]:
    for threshold, color, radius in _FALLBACK:
        if max_hp >= threshold:
            return color, radius
    return (80, 80, 80), 2

# Team outer ring colors (BGR)
TEAM_RING_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 80, 80),   # blue ring  = team 0
    1: (80,  80, 255),  # red ring   = team 1
}

# HUD strip height in pixels
HUD_H = 30

# Approximate LOS / radar radii in Spring world units (used by --fog mode)
_LOS_RADIUS_WORLD   = 380
_RADAR_RADIUS_WORLD = 1500


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class ReplayRenderer:
    """Renders a single replay obs dict into a BGR numpy image.

    Parameters
    ----------
    map_w, map_h : float
        Map dimensions in Spring world units (Game.mapSizeX / Z).
    img_w, img_h : int
        Output image size in pixels. Default 1024×768.
    use_fog : bool
        If True, draw approximate LOS/radar circles (black background).
    perspective_team : int
        Which team index (0 or 1) to use for fog PoV. -1 = god's-eye.
    """

    def __init__(
        self,
        map_w: float,
        map_h: float,
        img_w: int = 1024,
        img_h: int = 768,
        use_fog: bool = False,
        perspective_team: int = -1,
    ) -> None:
        self.map_w = float(map_w)
        self.map_h = float(map_h)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.use_fog = use_fog
        self.perspective_team = int(perspective_team)

        game_h = self.img_h - HUD_H
        self.scale_x = self.img_w / self.map_w
        self.scale_z = game_h / self.map_h

        # Track max feature metal seen so far for brightness normalisation
        self._max_feature_m: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(self, obs: dict[str, Any]) -> np.ndarray:
        """Return a BGR image (img_h × img_w × 3) for this observation."""
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        if self.use_fog:
            self._draw_fog(img, obs)

        self._draw_features(img, obs)
        self._draw_units(img, obs)
        self._draw_hud(img, obs)
        return img

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _w2p(self, x: float, z: float) -> tuple[int, int]:
        """World (x, z) -> image pixel (px, py). Clipped to image bounds."""
        px = int(x * self.scale_x)
        py = int(HUD_H + z * self.scale_z)
        px = max(0, min(self.img_w - 1, px))
        py = max(0, min(self.img_h - 1, py))
        return px, py

    # ------------------------------------------------------------------
    # Fog of war (approximate)
    # ------------------------------------------------------------------

    def _draw_fog(self, img: np.ndarray, obs: dict) -> None:
        """Draw approximate LOS/radar zones as additive light on black background."""
        fog = np.zeros_like(img)
        teams = obs.get("teams", {})
        own_key = str(self.perspective_team) if self.perspective_team >= 0 else None
        if own_key is None:
            # god's-eye: use all units for fog aesthetics
            all_units = [u for td in teams.values() for u in td.get("units", [])]
        else:
            td = teams.get(own_key, {})
            all_units = td.get("units", [])

        los_r   = int(_LOS_RADIUS_WORLD   * self.scale_x)
        radar_r = int(_RADAR_RADIUS_WORLD * self.scale_x)
        los_r   = max(3, los_r)
        radar_r = max(los_r + 2, radar_r)

        for u in all_units:
            px, py = self._w2p(u.get("x", 0), u.get("z", 0))
            cv2.circle(fog, (px, py), radar_r, (18, 18, 18), -1)
            cv2.circle(fog, (px, py), los_r,   (42, 42, 42), -1)

        np.maximum(img, fog, out=img)

    # ------------------------------------------------------------------
    # Features (wrecks with metal)
    # ------------------------------------------------------------------

    def _draw_features(self, img: np.ndarray, obs: dict) -> None:
        features = obs.get("features") or []
        for f in features:
            m = float(f.get("m", 0) or 0)
            if m <= 0:
                continue
            if m > self._max_feature_m:
                self._max_feature_m = m
            brightness = int(60 + 195 * min(1.0, m / self._max_feature_m))
            px, py = self._w2p(float(f.get("x", 0)), float(f.get("z", 0)))
            cv2.circle(img, (px, py), 2, (0, brightness, brightness // 2), -1)

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def _unit_appearance(
        self, unit: dict, team_idx: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        """Return (fill_BGR, ring_BGR, radius)."""
        def_id  = int(unit.get("def", 0))
        max_hp  = float(unit.get("max_hp", 0) or 0)
        hp      = float(unit.get("hp", max_hp) or max_hp)
        hp_ratio = max(0.35, hp / max_hp) if max_hp > 0 else 1.0

        cid = DEF_TO_CLUSTER.get(def_id)
        if cid is not None:
            meta   = CLUSTER_META[cid]
            fill   = meta["color"]
            radius = meta["radius"]
        else:
            fill, radius = _fallback_color_radius(max_hp)

        # Dim damaged units
        fill = (
            int(fill[0] * hp_ratio),
            int(fill[1] * hp_ratio),
            int(fill[2] * hp_ratio),
        )
        ring = TEAM_RING_COLORS.get(team_idx, (128, 128, 128))
        return fill, ring, radius

    def _should_draw(self, unit: dict, team_idx: int) -> bool:
        """Return False to hide unit (respects los/radar when data present)."""
        if self.perspective_team < 0:
            return True                          # god's-eye: show all
        if team_idx == self.perspective_team:
            return True                          # always see own units

        # Enemy unit: check los/radar fields if present
        los_list   = unit.get("los")
        radar_list = unit.get("radar")
        if los_list is None and radar_list is None:
            # No visibility data in this JSONL: fall back to god's-eye for enemies
            return True
        # Our ally ID: look it up from obs (stored as team data "ally")
        # For simplicity, show if either list is non-empty
        return bool(los_list) or bool(radar_list)

    def _draw_units(self, img: np.ndarray, obs: dict) -> None:
        teams = obs.get("teams", {})
        for team_str, team_data in teams.items():
            try:
                team_idx = int(team_str)
            except ValueError:
                team_idx = 0

            for u in team_data.get("units", []):
                if not self._should_draw(u, team_idx):
                    continue

                fill, ring, r = self._unit_appearance(u, team_idx)
                px, py = self._w2p(float(u.get("x", 0)), float(u.get("z", 0)))

                # Outer ring (team color)
                cv2.circle(img, (px, py), r + 2, ring,  -1)
                # Inner fill (cluster/type color)
                cv2.circle(img, (px, py), r,     fill,  -1)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, img: np.ndarray, obs: dict) -> None:
        # Dark grey top bar
        img[:HUD_H, :] = (30, 30, 30)

        frame = int(obs.get("frame", 0) or 0)
        total_s = frame // 30
        mins, secs = divmod(total_s, 60)
        center_text = f"F{frame:>6d}  {mins:02d}:{secs:02d}"
        (tw, _), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, center_text,
                    ((self.img_w - tw) // 2, HUD_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        teams = obs.get("teams", {})
        bar_max_w = 120
        bar_h     = 7

        for team_str, td in teams.items():
            try:
                tidx = int(team_str)
            except ValueError:
                continue
            color   = TEAM_RING_COLORS.get(tidx, (128, 128, 128))
            m       = float(td.get("metal",          0) or 0)
            ms      = float(td.get("metal_storage",  1) or 1)
            e       = float(td.get("energy",         0) or 0)
            es      = float(td.get("energy_storage", 1) or 1)
            m_pct   = min(1.0, m / ms)
            e_pct   = min(1.0, e / es)

            if tidx == 0:
                x0 = 6
            else:
                x0 = self.img_w - bar_max_w - 6

            label = f"T{tidx} M:{m:5.0f}/{ms:.0f}"
            cv2.putText(img, label, (x0, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # Metal bar (green tint within team color)
            bx = x0
            by = 15
            cv2.rectangle(img, (bx, by), (bx + bar_max_w, by + bar_h),
                          (50, 50, 50), -1)
            cv2.rectangle(img, (bx, by), (bx + int(bar_max_w * m_pct), by + bar_h),
                          (80, 200, 80), -1)

            # Energy bar
            ey = by + bar_h + 2
            cv2.rectangle(img, (bx, ey), (bx + bar_max_w, ey + bar_h),
                          (50, 50, 50), -1)
            cv2.rectangle(img, (bx, ey), (bx + int(bar_max_w * e_pct), ey + bar_h),
                          (200, 200, 80), -1)


# ---------------------------------------------------------------------------
# Grid overlay (optional --grid mode)
# ---------------------------------------------------------------------------

def draw_grid_overlay(
    img: np.ndarray,
    obs: dict[str, Any],
    map_w: float,
    map_h: float,
    cells_x: int = 64,
    cells_z: int = 48,
    alpha: float = 0.30,
) -> None:
    """Draw a semi-transparent grid cell overlay on img (in-place).

    Each cell is colored by the dominant unit cluster, brightness by total HP.
    """
    img_h, img_w = img.shape[:2]
    game_h = img_h - HUD_H
    cell_px_x = img_w  / cells_x
    cell_px_z = game_h / cells_z

    # Accumulate metal-equivalent cost per cell per (team, cluster).
    # equiv cost = metalcost + energycost/60; falls back to max_hp if cost unknown.
    from collections import defaultdict
    cells: dict[tuple[int, int], dict[tuple[int, int], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    teams = obs.get("teams", {})
    for team_str, td in teams.items():
        try:
            tidx = int(team_str)
        except ValueError:
            continue
        for u in td.get("units", []):
            x     = float(u.get("x", 0))
            z     = float(u.get("z", 0))
            def_id = int(u.get("def", 0))
            cid   = DEF_TO_CLUSTER.get(def_id)
            if cid is None:
                continue
            # Use metal-equivalent cost; fall back to max_hp if not available
            value = DEF_TO_EQUIV_COST.get(def_id) or float(u.get("max_hp", 0) or 0)
            cx = min(cells_x - 1, int(x / map_w * cells_x))
            cz = min(cells_z - 1, int(z / map_h * cells_z))
            cells[(cz, cx)][(tidx, cid)] += value

    if not cells:
        return

    max_val = max(sum(v.values()) for v in cells.values())
    if max_val <= 0:
        return

    overlay = img.copy()

    for (cz, cx), hp_map in cells.items():
        total = sum(hp_map.values())
        (tidx, dom_cid), _ = max(hp_map.items(), key=lambda kv: kv[1])

        color = CLUSTER_META.get(dom_cid, {}).get("color", (128, 128, 128))
        brightness = min(1.0, 0.2 + 0.8 * (total / max_val) ** 0.5)
        fill = tuple(int(c * brightness) for c in color)

        # Apply team tint: shift slightly toward ring color
        ring = TEAM_RING_COLORS.get(tidx, (128, 128, 128))
        fill = tuple(
            int(fill[i] * 0.7 + ring[i] * 0.3) for i in range(3)
        )

        x0 = int(cx * cell_px_x)
        y0 = int(HUD_H + cz * cell_px_z)
        x1 = int((cx + 1) * cell_px_x)
        y1 = int(HUD_H + (cz + 1) * cell_px_z)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
