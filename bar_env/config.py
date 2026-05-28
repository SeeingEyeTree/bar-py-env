"""Paths and defaults.

Most things are derived from the package root and the user's BAR install.
"""

from __future__ import annotations

import os
from pathlib import Path

# Package root: .../bar-py-env/
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Where we download the engine into.
ENGINE_DIR = PACKAGE_ROOT / "engine"

# Where runs (write-dirs) are created.
# Order of precedence:
#   1. $BAR_PY_ENV_RUN_DIR env var
#   2. D:\BAR_Replays if the D: drive exists (Windows convenience)
#   3. <package_root>/run/ as fallback
def _default_run_dir() -> Path:
    env_override = os.environ.get("BAR_PY_ENV_RUN_DIR")
    if env_override:
        return Path(env_override)
    d_drive = Path("D:/")
    if os.name == "nt" and d_drive.exists():
        return Path("D:/BAR_Replays")
    return PACKAGE_ROOT / "run"


RUN_DIR = _default_run_dir()

# Where extracted replay artifacts (per-frame JSONL, parsed build orders, etc.)
# are written. Kept in a sibling directory of the per-run write-dirs so it's
# easy to find: e.g. D:\BAR_Replays\extracted\<replay-name>.extracted.jsonl
# Override per-script with --out-dir, or globally via $BAR_PY_ENV_EXTRACTED_DIR.
def _default_extracted_dir() -> Path:
    env_override = os.environ.get("BAR_PY_ENV_EXTRACTED_DIR")
    if env_override:
        return Path(env_override)
    return RUN_DIR / "extracted"


EXTRACTED_DIR = _default_extracted_dir()

# Where the bundled Lua bridge widget lives.
LUA_DIR = PACKAGE_ROOT / "lua"
BRIDGE_WIDGET_PATH = LUA_DIR / "bridge_widget.lua"

# BAR launcher config (source of truth for engine version + game version).
LAUNCHER_CONFIG_URL = "https://launcher-config.beyondallreason.dev/config.json"

# The launcher-config "package id" we pull the engine from. "manual-win" is the
# stable Windows alpha channel; "manual-win-test-engine" is the engine-test channel.
DEFAULT_LAUNCHER_PACKAGE_ID = "manual-win"

# Default TCP server port for the Python <-> widget bridge.
DEFAULT_PORT = 8765

# Default map for v1 MVP.
DEFAULT_MAP_NAME = "Comet Catcher Remake 1.8"


def default_bar_data_dir() -> Path:
    """Return the user's default Beyond All Reason data directory on Windows."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "Programs" / "Beyond-All-Reason" / "data"
    # Last-resort fallback used when LOCALAPPDATA isn't set (e.g. on Linux).
    return Path.home() / ".local" / "share" / "Beyond-All-Reason" / "data"


def headless_binary(engine_version: str | None = None) -> Path:
    """Find spring-headless(.exe) inside ENGINE_DIR.

    If engine_version is given (e.g. "2025.06.24"), returns the binary from
    a folder whose name contains that version. Falls back to the
    lexicographically-latest folder (= highest version) when no match exists
    or no version is specified.
    """
    if not ENGINE_DIR.exists():
        raise FileNotFoundError(
            f"Engine directory not found: {ENGINE_DIR}\n"
            f"Run: python scripts/fetch_engine.py"
        )
    candidates = list(ENGINE_DIR.glob("**/spring-headless.exe")) + list(
        ENGINE_DIR.glob("**/spring-headless")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No spring-headless binary found under {ENGINE_DIR}.\n"
            f"Run: python scripts/fetch_engine.py"
        )
    if engine_version:
        matching = [c for c in candidates if engine_version in str(c)]
        if matching:
            return matching[0]
    # Pick highest version by folder name (lexicographic order is correct for
    # "YYYY.MM.DD" version strings).
    candidates.sort(key=lambda p: str(p.parent), reverse=True)
    return candidates[0]
