"""Action helpers: typed constructors that produce the dict the widget expects.

The widget accepts action messages of this shape:

    {
      "unit":   <unitID>          # or "units": [uid1, uid2] for grouped orders
      "cmd":    <str-or-int>      # named command or raw cmd_id
      "def":    <unitDefID>       # only for BUILD
      "params": [<numbers>]
      "options": {"shift": ..., "ctrl": ..., ...}
    }

The functions in this module produce that dict for you. They map exactly to
the named commands the widget knows (see CMD_NAME_TO_ID in bridge_widget.lua).

Quick reference for Spring/Recoil command IDs:
    STOP         = 0     SELFD          = 65    ONOFF        = 85
    INSERT       = 1     LOAD_UNITS     = 75    RECLAIM      = 90
    REMOVE       = 2     UNLOAD_UNITS   = 80    CLOAK        = 95
    WAIT         = 5                            STOCKPILE    = 100
    MOVE         = 10    REPAIR         = 40    MANUALFIRE   = 105 (D-gun)
    FIGHT        = 14    FIRE_STATE     = 45    RESTORE      = 110
    PATROL       = 15    MOVE_STATE     = 50    REPEAT       = 115
    AREA_ATTACK  = 16    SELFD          = 65    TRAJECTORY   = 120
    ATTACK       = 20                           RESURRECT    = 125
    GUARD        = 25                           CAPTURE      = 130

BUILD commands are not in the table -- they use a negative cmd_id equal to
the -<unitDefID> of the thing to build. Use `build(unit, def_id, x, z)`.
"""

from __future__ import annotations

from typing import Iterable, Mapping


Options = Mapping[str, bool]


def _normalize_units(unit: int | None, units: Iterable[int] | None) -> dict:
    if units is not None:
        ulist = list(int(u) for u in units)
        if not ulist:
            raise ValueError("units cannot be empty")
        return {"units": ulist}
    if unit is None:
        raise ValueError("must pass either unit=... or units=...")
    return {"unit": int(unit)}


def _opt(options: Options | None) -> dict:
    if not options:
        return {}
    keep = {"shift", "ctrl", "alt", "right", "meta", "internal"}
    return {k: bool(v) for k, v in options.items() if k in keep and v}


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def move(unit: int | None = None, x: float = 0.0, z: float = 0.0, *,
         y: float = 0.0, units: Iterable[int] | None = None,
         options: Options | None = None) -> dict:
    return {
        **_normalize_units(unit, units),
        "cmd": "MOVE",
        "params": [float(x), float(y), float(z)],
        "options": _opt(options),
    }


def stop(unit: int | None = None, *, units: Iterable[int] | None = None,
         options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "STOP", "params": [],
            "options": _opt(options)}


def wait(unit: int | None = None, *, units: Iterable[int] | None = None,
         options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "WAIT", "params": [],
            "options": _opt(options)}


def fight(unit: int | None = None, x: float = 0.0, z: float = 0.0, *,
          y: float = 0.0, units: Iterable[int] | None = None,
          options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "FIGHT",
            "params": [float(x), float(y), float(z)], "options": _opt(options)}


def patrol(unit: int | None = None, x: float = 0.0, z: float = 0.0, *,
           y: float = 0.0, units: Iterable[int] | None = None,
           options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "PATROL",
            "params": [float(x), float(y), float(z)], "options": _opt(options)}


def guard(unit: int | None = None, target_unit: int = 0, *,
          units: Iterable[int] | None = None,
          options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "GUARD",
            "params": [int(target_unit)], "options": _opt(options)}


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

def attack_unit(unit: int | None = None, target_unit: int = 0, *,
                units: Iterable[int] | None = None,
                options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "ATTACK",
            "params": [int(target_unit)], "options": _opt(options)}


def attack_ground(unit: int | None = None, x: float = 0.0, z: float = 0.0, *,
                  y: float = 0.0, units: Iterable[int] | None = None,
                  options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "ATTACK",
            "params": [float(x), float(y), float(z)], "options": _opt(options)}


def area_attack(unit: int | None = None, x: float = 0.0, z: float = 0.0,
                radius: float = 100.0, *, y: float = 0.0,
                units: Iterable[int] | None = None,
                options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "AREA_ATTACK",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


def dgun(unit: int | None = None, target_unit: int | None = None,
         x: float | None = None, z: float | None = None, *, y: float = 0.0,
         options: Options | None = None) -> dict:
    """Commander D-gun (manual-fire). Either name a target unit or a ground point."""
    if target_unit is not None:
        params = [int(target_unit)]
    elif x is not None and z is not None:
        params = [float(x), float(y), float(z)]
    else:
        raise ValueError("dgun needs target_unit OR (x, z)")
    return {**_normalize_units(unit, None), "cmd": "MANUALFIRE",
            "params": params, "options": _opt(options)}


# ---------------------------------------------------------------------------
# Construction / economy
# ---------------------------------------------------------------------------

def build(unit: int | None = None, def_id: int | str = 0, x: float = 0.0, z: float = 0.0,
          *, facing: int = 0, y: float = 0.0, units: Iterable[int] | None = None,
          options: Options | None = None) -> dict:
    """Queue construction of unit-def `def_id` at (x, z) by builder `unit`.

    def_id may be a numeric unitDefID or a string def name (e.g. "cormex");
    string names are resolved by the widget via UnitDefNames at command time.
    facing is 0=South, 1=East, 2=North, 3=West (per Spring's convention).
    Pass options={'shift': True} to append to the build queue.
    """
    if isinstance(def_id, int) and def_id <= 0:
        raise ValueError("def_id must be a positive unitDefID or a string def name")
    return {
        **_normalize_units(unit, units),
        "cmd": "BUILD",
        "def": def_id if isinstance(def_id, str) else int(def_id),
        "params": [float(x), float(y), float(z), int(facing)],
        "options": _opt(options),
    }


def factory_queue(factory_unit: int, def_id: int | str,
                  count: int = 1, *, options: Options | None = None) -> list[dict]:
    """Queue `count` units of `def_id` to be built by a factory.

    def_id may be a numeric unitDefID or a string def name (e.g. "corraid").
    Returns a list of action dicts (one per queued unit) because Spring
    queues one at a time. Shift-queue is set automatically for repeats.
    """
    actions: list[dict] = []
    for i in range(max(1, int(count))):
        opt = dict(options or {})
        if i > 0:
            opt["shift"] = True
        actions.append({
            "unit": int(factory_unit),
            "cmd": "BUILD",
            "def": def_id if isinstance(def_id, str) else int(def_id),
            "params": [],  # factory builds don't need a position
            "options": _opt(opt),
        })
    return actions


def repair_unit(unit: int | None = None, target_unit: int = 0, *,
                units: Iterable[int] | None = None,
                options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "REPAIR",
            "params": [int(target_unit)], "options": _opt(options)}


def repair_area(unit: int | None = None, x: float = 0.0, z: float = 0.0,
                radius: float = 200.0, *, y: float = 0.0,
                units: Iterable[int] | None = None,
                options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "REPAIR",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


# Reclaim/Resurrect both use a special "feature id space" trick: feature ids
# are encoded as `Game.maxUnits + feature_id`. To keep this layer simple, we
# expose the unit form (most common for an agent), the feature form (advanced),
# and the area form. Callers issuing feature-target reclaims must compute
# the offset themselves -- see the obs payload's `Game.maxUnits` once we add it.

def reclaim_unit(unit: int | None = None, target_unit: int = 0, *,
                 units: Iterable[int] | None = None,
                 options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "RECLAIM",
            "params": [int(target_unit)], "options": _opt(options)}


def reclaim_feature(unit: int | None = None, feature_target_id: int = 0, *,
                    units: Iterable[int] | None = None,
                    options: Options | None = None) -> dict:
    """`feature_target_id` must already be `Game.maxUnits + feature_id`."""
    return {**_normalize_units(unit, units), "cmd": "RECLAIM",
            "params": [int(feature_target_id)], "options": _opt(options)}


def reclaim_area(unit: int | None = None, x: float = 0.0, z: float = 0.0,
                 radius: float = 200.0, *, y: float = 0.0,
                 units: Iterable[int] | None = None,
                 options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "RECLAIM",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


def resurrect_feature(unit: int | None = None, feature_target_id: int = 0, *,
                      units: Iterable[int] | None = None,
                      options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "RESURRECT",
            "params": [int(feature_target_id)], "options": _opt(options)}


def resurrect_area(unit: int | None = None, x: float = 0.0, z: float = 0.0,
                   radius: float = 200.0, *, y: float = 0.0,
                   units: Iterable[int] | None = None,
                   options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "RESURRECT",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


def capture_unit(unit: int | None = None, target_unit: int = 0, *,
                 units: Iterable[int] | None = None,
                 options: Options | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "CAPTURE",
            "params": [int(target_unit)], "options": _opt(options)}


# ---------------------------------------------------------------------------
# Unit states (toggle behaviors)
# ---------------------------------------------------------------------------

FIRE_HOLD       = 0
FIRE_RETURN     = 1
FIRE_AT_WILL    = 2

MOVE_HOLD_POS   = 0
MOVE_MANEUVER   = 1
MOVE_ROAM       = 2


def fire_state(unit: int | None = None, state: int = FIRE_AT_WILL, *,
               units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "FIRE_STATE",
            "params": [int(state)], "options": {}}


def move_state(unit: int | None = None, state: int = MOVE_MANEUVER, *,
               units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "MOVE_STATE",
            "params": [int(state)], "options": {}}


def on_off(unit: int | None = None, on: bool = True, *,
           units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "ONOFF",
            "params": [1 if on else 0], "options": {}}


def cloak(unit: int | None = None, on: bool = True, *,
          units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "CLOAK",
            "params": [1 if on else 0], "options": {}}


def trajectory(unit: int | None = None, high: bool = False, *,
               units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "TRAJECTORY",
            "params": [1 if high else 0], "options": {}}


def repeat(unit: int | None = None, on: bool = True, *,
           units: Iterable[int] | None = None) -> dict:
    """Toggle factory repeat-queue mode."""
    return {**_normalize_units(unit, units), "cmd": "REPEAT",
            "params": [1 if on else 0], "options": {}}


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

def load_units_target(transport: int, target_unit: int, *,
                      options: Options | None = None) -> dict:
    return {"unit": int(transport), "cmd": "LOAD_UNITS",
            "params": [int(target_unit)], "options": _opt(options)}


def load_units_area(transport: int, x: float, z: float, radius: float = 200.0, *,
                    y: float = 0.0, options: Options | None = None) -> dict:
    return {"unit": int(transport), "cmd": "LOAD_UNITS",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


def unload_units(transport: int, x: float, z: float, radius: float = 200.0, *,
                 y: float = 0.0, options: Options | None = None) -> dict:
    return {"unit": int(transport), "cmd": "UNLOAD_UNITS",
            "params": [float(x), float(y), float(z), float(radius)],
            "options": _opt(options)}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def self_destruct(unit: int | None = None, *,
                  units: Iterable[int] | None = None) -> dict:
    return {**_normalize_units(unit, units), "cmd": "SELFD", "params": [],
            "options": {}}


def stockpile(unit: int | None = None, *,
              options: Options | None = None) -> dict:
    """Queue one missile in a nuke/anti-nuke silo. Shift-click queues a batch."""
    return {**_normalize_units(unit, None), "cmd": "STOCKPILE",
            "params": [], "options": _opt(options)}


# A short list of friendly-name aliases the widget recognizes, useful for
# users building actions by string instead of by helper function.
NAMED_COMMANDS: tuple[str, ...] = (
    "STOP", "WAIT", "MOVE", "FIGHT", "PATROL", "GUARD",
    "ATTACK", "AREA_ATTACK", "MANUALFIRE", "DGUN",
    "BUILD",
    "REPAIR", "RECLAIM", "RESURRECT", "CAPTURE", "RESTORE",
    "FIRE_STATE", "MOVE_STATE", "ONOFF", "CLOAK", "TRAJECTORY", "REPEAT",
    "LOAD_UNITS", "UNLOAD_UNITS", "UNLOAD_UNIT",
    "SELFD", "STOCKPILE", "SET_WANTED_MAX_SPEED", "INSERT", "REMOVE",
)
