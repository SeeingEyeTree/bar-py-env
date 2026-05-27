"""Programmatic generation of Recoil engine start scripts.

A "start script" is a TDF-like text file telling the engine what map / game /
players / AIs to load. We generate one per match.

For v1 MVP we only support a single layout: one local player (our headless
instance, which the bridge widget runs inside) vs one NullAI opponent.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path


def _detect_installed_game_type() -> str:
    """Return the exact game-version string currently installed (e.g.
    'Beyond All Reason test-30166-21baea6'), falling back to the rapid tag
    'byar:test' if the metadata file can't be read.

    Chobby matches replay GameType fields against installed archive names by
    exact string, so using the rapid tag causes 'game not found' errors.
    """
    try:
        local_appdata = __import__("os").environ.get("LOCALAPPDATA")
        if not local_appdata:
            return "byar:test"
        versions_gz = (
            Path(local_appdata)
            / "Programs/Beyond-All-Reason/data"
            / "rapid/repos-cdn.beyondallreason.dev/byar/versions.gz"
        )
        if not versions_gz.exists():
            return "byar:test"
        with gzip.open(versions_gz, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",")
                if len(parts) >= 4 and parts[0] == "byar:test":
                    return parts[3].strip()
    except Exception:
        pass
    return "byar:test"


# Resolved once at import time. On a machine with BAR installed this will be
# something like "Beyond All Reason test-30166-21baea6", which Chobby can
# match against installed archives. Falls back to the rapid tag if unknown.
DEFAULT_GAME_TYPE = _detect_installed_game_type()


@dataclass
class StartScriptConfig:
    map_name: str
    game_type: str = DEFAULT_GAME_TYPE
    player_name: str = "PyEnv"
    player_side: str = "Armada"  # Armada or Cortex
    enemy_ai_short_name: str = "NullAI"
    enemy_ai_version: str = "0.1"   # "0.1" for NullAI, "stable" for BARb/CircuitAI
    enemy_ai_name: str = "Enemy"
    enemy_side: str = "Cortex"
    # StartPosType:
    #   0 = fixed (engine picks from map defaults)
    #   1 = random
    #   2 = chooseInGame (interactive)
    #   3 = chooseBeforeGame (use [TEAM*]'s StartPosX / StartPosZ from script)
    # If you set player_start_pos_x/z or enemy_start_pos_x/z below, you almost
    # certainly want start_pos_type=3 -- otherwise the engine will ignore the
    # explicit coords and re-pick from the map's defaults.
    start_pos_type: int = 0
    # Optional explicit start coordinates (world units, same scale as Spring's
    # GetUnitPosition x/z). When set, the renderer emits StartPosX / StartPosZ
    # for that team. None = let the engine pick.
    player_start_pos_x: float | None = None
    player_start_pos_z: float | None = None
    enemy_start_pos_x: float | None = None
    enemy_start_pos_z: float | None = None
    fixed_rng_seed: int = 1
    death_mode: str = "killall"   # "killall" | "neverend" | "com" | "comcontrol"
    no_helper_ais: int = 0
    record_demo: int = 1          # 1 writes a .sdfz to the write-dir's demos/ subdir
    speed: int = 0                # 0 = uncapped (headless), N = speed multiplier


def _team_start_pos_lines(x: float | None, z: float | None) -> str:
    """Render StartPosX / StartPosZ lines, or empty string if not set."""
    if x is None or z is None:
        return ""
    # 4-space indent to match the rest of the [TEAM*] body.
    return f"        StartPosX={int(round(x))};\n        StartPosZ={int(round(z))};\n"


def render_start_script(cfg: StartScriptConfig) -> str:
    """Render a Recoil start script to a string."""
    # Note: indentation is purely cosmetic; the engine parses this loosely.
    player_pos = _team_start_pos_lines(cfg.player_start_pos_x, cfg.player_start_pos_z)
    enemy_pos = _team_start_pos_lines(cfg.enemy_start_pos_x, cfg.enemy_start_pos_z)
    return f"""[GAME]
{{
    GameType={cfg.game_type};
    MapName={cfg.map_name};
    StartPosType={cfg.start_pos_type};
    FixedRNGSeed={cfg.fixed_rng_seed};
    RecordDemo={cfg.record_demo};
    GameStartDelay=0;
    NoHelperAIs={cfg.no_helper_ais};
    MyPlayerName={cfg.player_name};
    IsHost=1;

    [MODOPTIONS]
    {{
        deathmode={cfg.death_mode};
        maxspeed={cfg.speed if cfg.speed > 0 else 100};
        minspeed=0.1;
        allowuserwidgets=1;
        allowunitcontrolwidgets=1;
    }}

    [ALLYTEAM0] {{ numallies=0; }}
    [ALLYTEAM1] {{ numallies=0; }}

    [TEAM0]
    {{
        teamleader=0;
        allyteam=0;
        side={cfg.player_side};
        rgbcolor=0.2 0.4 0.9;
{player_pos}    }}
    [TEAM1]
    {{
        teamleader=0;
        allyteam=1;
        side={cfg.enemy_side};
        rgbcolor=0.9 0.2 0.2;
{enemy_pos}    }}

    [PLAYER0]
    {{
        name={cfg.player_name};
        team=0;
    }}

    [AI0]
    {{
        host=0;
        team=1;
        shortname={cfg.enemy_ai_short_name};
        name={cfg.enemy_ai_name};
        version={cfg.enemy_ai_version};
    }}
}}
"""
