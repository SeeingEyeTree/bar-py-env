"""Canonical specification of the observation and action spaces for bar-py-env.

Both spaces are currently raw-dict (JSON-serializable Python dicts) mirroring
the wire protocol. A future version will lift them into numpy / Gymnasium spaces.

Coordinate system
-----------------
Spring uses a left-handed Y-up world:
  X : 0 → map_info["size_x"]   (West → East)
  Y : terrain height (metres above sea level)
  Z : 0 → map_info["size_z"]   (North → South)
A frame runs at 30 Hz (simulation frames per second at 1× speed).

Status of each field
--------------------
  [live]    : emitted by the live-play widget (build_observation)
  [replay]  : emitted by the replay widget (build_observation_replay)
  [both]    : present in both
  [planned] : not yet wired up; documented here so the API is stable

Wire note
---------
The JSON key for unit definition IDs is "def" (a Python keyword).
TypedDicts below use "def_id" as the Python name; callers reading raw obs dicts
must use obs["def"].
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAMES_PER_SECOND = 30          # BAR simulation rate at 1× game speed

# Spring heading is 0-65535 (full circle). The widget compacts it to 0-255
# (divides by 256, ≈ 1.4° resolution). 0 = North, 64 = East, 128 = South,
# 192 = West (clockwise).
HEADING_NORTH =   0
HEADING_EAST  =  64
HEADING_SOUTH = 128
HEADING_WEST  = 192

# Spring build facing: 0=South 1=East 2=North 3=West
FACING_SOUTH = 0
FACING_EAST  = 1
FACING_NORTH = 2
FACING_WEST  = 3

# Fire state
FIRE_HOLD      = 0   # never fire
FIRE_RETURN    = 1   # fire only when fired upon
FIRE_AT_WILL   = 2   # fire at any visible enemy (default)

# Move state
MOVE_HOLD_POS  = 0   # stay in place, do not chase
MOVE_MANEUVER  = 1   # default: dodge while fighting
MOVE_ROAM      = 2   # chase enemies freely


# ===========================================================================
# OBSERVATION SPACE
# ===========================================================================
#
# env.reset() and env.step() both return an obs dict of this shape.
# Use these TypedDicts for type-checking and as the authoritative reference.
# All coordinates are floating-point Spring world units unless noted.

# ---------------------------------------------------------------------------
# MetalSpot  [both]
# ---------------------------------------------------------------------------
# Static for the lifetime of the match; cached after first GameFrame.
#
#   x  : int   world X of the metal spot centre
#   z  : int   world Z of the metal spot centre
#   m  : float extraction rate in metal/second at 100% efficiency (≈ 0.5–2.5)
#
# Example:
#   {"x": 1472, "z": 3200, "m": 1.5}

# ---------------------------------------------------------------------------
# MapInfo  [both]
# ---------------------------------------------------------------------------
#
#   size_x       : float   map width  (typically 512–8192 Spring units)
#   size_z       : float   map depth  (typically 512–8192 Spring units)
#   metal_spots  : list[MetalSpot]
#                          All extractable metal spots on the map.
#                          Same list every frame once computed; may be [] on
#                          frame 0 before api_resource_spot_finder runs.

# ---------------------------------------------------------------------------
# Resources  [live]  (per-team econ snapshot)
# ---------------------------------------------------------------------------
#
#   metal          : float   current stored metal        (0 → metal_storage)
#   metal_storage  : float   metal storage cap           (starts at 1 000)
#   metal_pull     : float   metal drain rate (metal/frame)
#   metal_income   : float   metal production (metal/frame); includes mexes,
#                            reclaim income, and commander regen
#   energy         : float   current stored energy       (0 → energy_storage)
#   energy_storage : float   energy storage cap          (starts at 250)
#   energy_pull    : float   energy drain rate (energy/frame)
#   energy_income  : float   energy production (energy/frame); solar, wind, …
#
# Note: divide pull/income by FRAMES_PER_SECOND to get per-second rates.

# ---------------------------------------------------------------------------
# UnitObs  [live]
# ---------------------------------------------------------------------------
#
#   id      : int    Spring unit ID; unique within a match
#   def_id  : int    unitDefID; key into UnitDefs for type info      wire: "def"
#   x       : float  world X  (0 → size_x)
#   y       : float  world Y  (terrain elevation)
#   z       : float  world Z  (0 → size_z)
#   hp      : float  current hit points  (0 → max_hp)
#   max_hp  : float  maximum hit points  (static per unit type)
#
# [planned]
#   h       : int    heading 0–255 (0=N, 64=E, 128=S, 192=W); compact heading
#   bp      : float  build progress 0.0–1.0; only present while under construction
#   idle    : bool   True when the unit has an empty command queue
#   vx, vz  : float  velocity components (Spring units/frame)

# ---------------------------------------------------------------------------
# EnemyObs  [live]
# ---------------------------------------------------------------------------
#
#   id      : int    Spring unit ID
#   def_id  : int    unitDefID if unit is in LOS; 0 if only visible via radar
#                    wire: "def"
#   x       : float  world X
#   y       : float  world Y
#   z       : float  world Z
#
# [planned]
#   hp      : float  current HP; present only when unit is in LOS
#   max_hp  : float  max HP;    present only when unit is in LOS
#   radar   : bool   True when detected by radar only (type/hp unknown)
#   h       : int    heading; present only when unit is in LOS

# ---------------------------------------------------------------------------
# Observation  [live]  — top-level dict returned by reset() / step()
# ---------------------------------------------------------------------------
#
#   type       : "obs"            always "obs"
#   frame      : int              game frame; divide by 30 for elapsed seconds
#   done       : bool             True on the terminal step
#   result     : str | None       terminal outcome:
#                                   "win"          — we won
#                                   "loss"         — we lost
#                                   "disconnected" — engine dropped the conn
#                                   None           — game in progress
#   team       : int              our team ID (stable for the full match)
#   ally       : int              our ally-team ID
#   map        : MapInfo
#   resources  : Resources
#   my_units   : list[UnitObs]    every unit we own
#   enemies    : list[EnemyObs]   enemy units in LOS or on radar

# ---------------------------------------------------------------------------
# ReplayTeam  [replay]
# ---------------------------------------------------------------------------
# The replay obs replaces the per-team live obs with a "teams" dict keyed by
# string team ID. Each entry has:
#
#   ally           : int           ally-team ID for this team
#   metal          : float         \ same semantics as the live Resources fields
#   metal_storage  : float          |
#   metal_pull     : float          |
#   metal_income   : float          |
#   energy         : float          |
#   energy_storage : float          |
#   energy_pull    : float          |
#   energy_income  : float         /
#   units          : list[ReplayUnitObs]

# ---------------------------------------------------------------------------
# ReplayUnitObs  [replay]
# ---------------------------------------------------------------------------
# Superset of UnitObs; also includes:
#
#   h   : int             compact heading (always present in replay)
#   bp  : float           build progress (only present when 0 < bp < 1)
#   los   : list[int]     ally-team IDs for which this unit is in LOS
#   radar : list[int]     ally-team IDs for which this unit is on radar only

# ---------------------------------------------------------------------------
# ReplayObservation  [replay]  — top-level dict returned in replay mode
# ---------------------------------------------------------------------------
#
#   type                 : "obs"
#   frame                : int
#   done                 : bool
#   result               : str | None     same as live
#   map                  : MapInfo        (no metal_spots in current impl — planned)
#   teams                : dict[str, ReplayTeam]   keyed by str(team_id)
#   commands_this_frame  : list[RawCommand]
#
# RawCommand (one entry per engine command event this frame):
#   team    : int         team that issued this command
#   unit    : int         unit that received it
#   cmd_id  : int         raw Spring cmd_id (negative = BUILD, see below)
#   params  : list[float]
#   options : dict[str, bool]


# ===========================================================================
# ACTION SPACE
# ===========================================================================
#
# env.step(actions) accepts a list of action dicts.
# Multiple actions may be issued in a single step; the widget applies them all
# before advancing the simulation.
#
# Every action dict has:
#
#   "unit"  : int          target a single unit  — OR —
#   "units" : list[int]    target multiple units (grouped order)
#
#   "cmd"   : str          named command (see sections below) — OR —
#             int          raw Spring cmd_id (advanced use; prefer named form)
#
#   "params": list[float]  command arguments (see per-command docs below)
#
#   "options": dict        modifier flags (all bool, default False):
#              "shift"  — append to command queue instead of replacing
#              "ctrl"   — modifier (context-dependent per command)
#              "alt"    — modifier (context-dependent per command)
#              "right"  — right-click semantics (rare)
#              "meta"   — meta modifier (rare)
#
# The helper functions in bar_env.actions produce these dicts. Use them
# instead of building dicts by hand unless you need an unsupported command.

# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
#
#   MOVE    params: [x, y, z]
#           Move to world point. Unit stops when it arrives.
#
#   FIGHT   params: [x, y, z]
#           Move to world point, attacking any enemies encountered en route.
#           Preferred over MOVE for offensive pushes.
#
#   PATROL  params: [x, y, z]
#           Add a patrol waypoint. Unit loops through all patrol waypoints.
#           Shift-queue multiple PATROL commands to make a patrol circuit.
#
#   GUARD   params: [target_unit_id]
#           Follow and repair/protect another unit.
#
#   STOP    params: []
#           Cancel all queued orders. Unit becomes idle.
#
#   WAIT    params: []
#           Pause at current position until ordered otherwise.

# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------
#
#   ATTACK  params: [target_unit_id]       attack a specific unit
#           params: [x, y, z]              attack a ground point (area splash)
#
#   AREA_ATTACK
#           params: [x, y, z, radius]      all weapons fire into the area
#
#   MANUALFIRE  (D-gun / special weapon)
#           params: [target_unit_id]       fire at unit
#           params: [x, y, z]             fire at ground

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
#
#   BUILD   "def": str | int
#           params: [x, y, z, facing]     for mobile builders
#           params: []                    for factory queuing (position ignored)
#
#           "def" may be:
#             - a string def name, e.g. "cormex", "corvp"   (resolved by widget)
#             - a positive int unitDefID
#           facing: 0=South 1=East 2=North 3=West  (see FACING_* constants)
#           Shift-queue to append to a builder's existing queue.
#
#           BUILD sent to a factory queues one unit. Use repeat(factory_id, on=True)
#           to make the factory loop its queue.

# ---------------------------------------------------------------------------
# Repair / Reclaim / Resurrect / Capture
# ---------------------------------------------------------------------------
#
#   REPAIR  params: [target_unit_id]
#           params: [x, y, z, radius]     repair all friendlies in area
#
#   RECLAIM params: [target_unit_id]      reclaim a living unit (hostile or own)
#           params: [feature_id]          reclaim a wreckage/tree feature;
#                                         feature_id = Spring feature ID
#                                         (note: NOT Game.maxUnits offset here)
#           params: [x, y, z, radius]    reclaim all reclaimables in area
#
#   RESURRECT
#           params: [feature_id]          resurrect a wreckage into a unit
#           params: [x, y, z, radius]    resurrect all wrecks in area
#
#   CAPTURE params: [target_unit_id]     capture an enemy unit

# ---------------------------------------------------------------------------
# Unit states (toggle behaviors)
# ---------------------------------------------------------------------------
#
#   FIRE_STATE  params: [0|1|2]          0=hold 1=return 2=at-will (default 2)
#   MOVE_STATE  params: [0|1|2]          0=hold-pos 1=maneuver 2=roam (default 1)
#   ONOFF       params: [0|1]            0=disable 1=enable unit
#   CLOAK       params: [0|1]            0=decloak 1=cloak
#   TRAJECTORY  params: [0|1]            0=low arc 1=high arc (artillery)
#   REPEAT      params: [0|1]            factory repeat queue off/on
#
# State commands ignore "options" and take no "units" grouping.

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
#
#   LOAD_UNITS   params: [target_unit_id]           load one unit into transport
#                params: [x, y, z, radius]          load all units in area
#
#   UNLOAD_UNITS params: [x, y, z, radius]          unload at location

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
#
#   SELFD       params: []               self-destruct
#   STOCKPILE   params: []               queue one missile in nuke/anti-nuke silo
#                                        (shift-queue to build multiple)
#   RESTORE     params: [x, y, z, radius]  restore terrain height in area

# ---------------------------------------------------------------------------
# Raw cmd_id reference (for reading replay command streams)
# ---------------------------------------------------------------------------

CMD_STOP         =   0
CMD_INSERT       =   1   # internal; skip when replaying
CMD_REMOVE       =   2   # internal; skip when replaying
CMD_WAIT         =   5
CMD_MOVE         =  10
CMD_FIGHT        =  14
CMD_PATROL       =  15
CMD_AREA_ATTACK  =  16
CMD_ATTACK       =  20
CMD_GUARD        =  25
CMD_REPAIR       =  40
CMD_FIRE_STATE   =  45
CMD_MOVE_STATE   =  50
CMD_SELFD        =  65
CMD_LOAD_UNITS   =  75
CMD_UNLOAD_UNITS =  80
CMD_ONOFF        =  85
CMD_RECLAIM      =  90
CMD_CLOAK        =  95
CMD_STOCKPILE    = 100
CMD_MANUALFIRE   = 105
CMD_RESTORE      = 110
CMD_REPEAT       = 115
CMD_TRAJECTORY   = 120
CMD_RESURRECT    = 125
CMD_CAPTURE      = 130

# BUILD commands use a negative cmd_id: cmd_id = -(unitDefID).
# Example: cmd_id=-42 means "build the unit whose unitDefID is 42".
# When sending actions use the BUILD helper or the string def name; the widget
# converts "def" → negative cmd_id internally.

_CMD_ID_TO_NAME: dict[int, str] = {v: k for k, v in {
    "STOP":        CMD_STOP,
    "INSERT":      CMD_INSERT,
    "REMOVE":      CMD_REMOVE,
    "WAIT":        CMD_WAIT,
    "MOVE":        CMD_MOVE,
    "FIGHT":       CMD_FIGHT,
    "PATROL":      CMD_PATROL,
    "AREA_ATTACK": CMD_AREA_ATTACK,
    "ATTACK":      CMD_ATTACK,
    "GUARD":       CMD_GUARD,
    "REPAIR":      CMD_REPAIR,
    "FIRE_STATE":  CMD_FIRE_STATE,
    "MOVE_STATE":  CMD_MOVE_STATE,
    "SELFD":       CMD_SELFD,
    "LOAD_UNITS":  CMD_LOAD_UNITS,
    "UNLOAD_UNITS":CMD_UNLOAD_UNITS,
    "ONOFF":       CMD_ONOFF,
    "RECLAIM":     CMD_RECLAIM,
    "CLOAK":       CMD_CLOAK,
    "STOCKPILE":   CMD_STOCKPILE,
    "MANUALFIRE":  CMD_MANUALFIRE,
    "RESTORE":     CMD_RESTORE,
    "REPEAT":      CMD_REPEAT,
    "TRAJECTORY":  CMD_TRAJECTORY,
    "RESURRECT":   CMD_RESURRECT,
    "CAPTURE":     CMD_CAPTURE,
}.items()}


def cmd_id_to_name(cmd_id: int) -> str:
    """Return the named string for a raw Spring cmd_id, e.g. 10 → 'MOVE'.

    For BUILD commands (cmd_id < 0) returns 'BUILD(def=<unitDefID>)'.
    Unknown IDs return 'CMD_<n>'.
    """
    if cmd_id < 0:
        return f"BUILD(def={-cmd_id})"
    return _CMD_ID_TO_NAME.get(cmd_id, f"CMD_{cmd_id}")


def is_build_cmd(cmd_id: int) -> bool:
    return cmd_id < 0


def build_def_id_from_cmd(cmd_id: int) -> int:
    """Extract unitDefID from a raw BUILD cmd_id. Raises if not a BUILD command."""
    if cmd_id >= 0:
        raise ValueError(f"not a BUILD cmd_id: {cmd_id}")
    return -cmd_id
