#!/usr/bin/env python3
"""Download BAR 1v1 replays for pre-training, filtered by map and player list.

The original script in C:/Users/malco/AppData/Local/Programs/Python/Python311/
had three bugs that combined to download zero matching files:

  1. It hardcoded engineVersion='2025.06.12' into the filename, but every
     replay is recorded against whatever engine was current at game time.
     Fix: read `fileName` directly from the detail endpoint instead of
     reconstructing it client-side.
  2. It only fetched the list endpoint, which returns a sparse summary;
     the per-replay detail (engineVersion, fileName, skills) needs
     /replays/<id>.
  3. It applied filters client-side. The BAR API actually supports
     `preset=duel`, `hasBots=false`, `maps=<scriptName>`, and `players=<name>`.

How player filtering works: the API does AND across multiple `players=`
parameters (a game must include all of them). So for "at least one of N
players", this script issues one query per player and merges results.

Usage:
    # Default: top 8 active Comet Catcher 1v1 players, 50 replays max
    python scripts/download_replays.py

    # Specify your own player list and download count
    python scripts/download_replays.py --players Nekuodah Hermoise --max 100

    # Use a different map
    python scripts/download_replays.py --map "Supreme Isthmus v2.1"

    # Don't actually download, just print what would be fetched
    python scripts/download_replays.py --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

# Make `bar_env` importable when run directly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import default_bar_data_dir  # noqa: E402

API_BASE = "https://api.bar-rts.com/replays"
STORAGE_BASE = (
    "https://storage.uk.cloud.ovh.net/v1/"
    "AUTH_10286efc0d334efd917d476d7183232e/BAR/demos"
)
USER_AGENT = "bar-py-env/0.0.1 (replay-pretraining-downloader)"

# Default high-skill 1v1 Comet Catcher players (sampled from the last ~80
# duels on the map at the time of writing). Edit this list to whatever
# subset of players you want to clone behavior from.
DEFAULT_PLAYERS: list[str] = [
    "[APM]Nekuodah",
    "Hermoise",
    "DanceCommander",
    "mgtyalx",
    "porlex",
    "bedoab",
    "[MADO]Tripl3",
    "[Grumpi]stathis2000",
]

DEFAULT_MAP = "Comet Catcher Remake 1.8"
DEFAULT_OUTPUT_DIR = Path("D:/BAR_Replays/Human_Pretraining")
DEFAULT_MAX_DOWNLOADS = 50
DEFAULT_PAGES_PER_PLAYER = 5     # 20 results per page => up to 100 games/player
PAGE_SIZE = 20


# --------------------------------------------------------------------------
# Local BAR install introspection
# --------------------------------------------------------------------------

# BAR's launcher stores rapid tag -> archive metadata at:
#   <bar_data>/rapid/repos-cdn.beyondallreason.dev/byar/versions.gz
# Each line is "<tag>,<hash>,<deps>,<archive_name>". The `byar:test` tag points
# at whatever game version your launcher most recently synced.
_BYAR_VERSIONS_GZ = Path(
    "rapid/repos-cdn.beyondallreason.dev/byar/versions.gz"
)


def detect_installed_game_version(bar_data_dir: Path) -> str | None:
    """Return the archive_name `byar:test` currently resolves to, or None."""
    versions_gz = bar_data_dir / _BYAR_VERSIONS_GZ
    if not versions_gz.exists():
        return None
    try:
        with gzip.open(versions_gz, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",")
                if len(parts) >= 4 and parts[0] == "byar:test":
                    # parts[3] is the human archive name, e.g.
                    # "Beyond All Reason test-30166-21baea6"
                    return parts[3].strip()
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 30.0) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, timeout: float = 120.0) -> int:
    """Download a file. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        data = resp.read()
        f.write(data)
        return len(data)


# --------------------------------------------------------------------------
# API queries
# --------------------------------------------------------------------------

def list_duels_for_player(
    player: str, map_name: str, pages: int
) -> list[dict[str, Any]]:
    """Page through duel replays involving a given player on a given map."""
    out: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        q = urlencode({
            "preset": "duel",
            "hasBots": "false",
            "maps": map_name,
            "players": player,
            "limit": PAGE_SIZE,
            "page": page,
        })
        try:
            payload = _get_json(f"{API_BASE}?{q}")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  ! list query failed for {player} p{page}: {exc}", file=sys.stderr)
            break
        page_data = payload.get("data") or []
        if not page_data:
            break
        out.extend(page_data)
        if len(page_data) < PAGE_SIZE:
            break  # last page
    return out


def fetch_detail(replay_id: str) -> dict[str, Any] | None:
    try:
        return _get_json(f"{API_BASE}/{replay_id}")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  ! detail fetch failed for {replay_id}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Skill helpers
# --------------------------------------------------------------------------

def _parse_skill(raw: Any) -> float | None:
    """BAR returns skills as strings like '[15.72]'."""
    if raw is None:
        return None
    s = str(raw).strip().strip("[]")
    try:
        return float(s)
    except ValueError:
        return None


def replay_skill_summary(detail: dict[str, Any]) -> tuple[float, float, list[str]]:
    """Return (max_skill, sum_skill, [player_names])."""
    skills: list[float] = []
    names: list[str] = []
    for ally in detail.get("AllyTeams", []) or []:
        for p in ally.get("Players", []) or []:
            name = p.get("name")
            if name:
                names.append(name)
            s = _parse_skill(p.get("skill"))
            if s is not None:
                skills.append(s)
    if not skills:
        return (0.0, 0.0, names)
    return (max(skills), sum(skills), names)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--players", nargs="+", default=DEFAULT_PLAYERS,
        help="Player names to include (at least one must be in the replay). "
             "Defaults to a list of 8 high-skill 1v1 Comet Catcher regulars.",
    )
    parser.add_argument(
        "--players-file", type=Path, default=None,
        help="Optional file with one player name per line; overrides --players.",
    )
    parser.add_argument(
        "--map", dest="map_name", default=DEFAULT_MAP,
        help=f"Map scriptName to filter on. Default: '{DEFAULT_MAP}'.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save .sdfz files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max", dest="max_downloads", type=int, default=DEFAULT_MAX_DOWNLOADS,
        help=f"Max replays to download. Default: {DEFAULT_MAX_DOWNLOADS}",
    )
    parser.add_argument(
        "--pages-per-player", type=int, default=DEFAULT_PAGES_PER_PLAYER,
        help=f"How many list-pages (of {PAGE_SIZE}) to fetch per player. "
             f"Default: {DEFAULT_PAGES_PER_PLAYER}.",
    )
    parser.add_argument(
        "--min-skill", type=float, default=0.0,
        help="Drop replays where no player has at least this skill rating.",
    )
    parser.add_argument(
        "--sort-by", choices=("max", "sum"), default="max",
        help="Rank candidates by the highest player skill ('max', default) "
             "or by the total skill in the game ('sum').",
    )
    parser.add_argument(
        "--game-version", default=None,
        help="Only download replays whose `gameVersion` matches this string "
             "(e.g. 'Beyond All Reason test-30166-21baea6'). "
             "Default: auto-detect from your BAR install's rapid metadata.",
    )
    parser.add_argument(
        "--any-version", action="store_true",
        help="Skip the game-version filter and download replays from any "
             "version. You'll have to fetch missing archives at extraction "
             "time via pr-downloader.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--request-delay", type=float, default=0.1,
        help="Seconds to sleep between HTTP requests (be polite to the API).",
    )
    args = parser.parse_args()

    # Resolve player list
    if args.players_file:
        text = args.players_file.read_text(encoding="utf-8")
        players = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        print(f"Loaded {len(players)} players from {args.players_file}")
    else:
        players = list(args.players)
    if not players:
        sys.stderr.write("No players specified.\n")
        return 1

    # Resolve game-version filter.
    if args.any_version:
        target_version: str | None = None
    elif args.game_version:
        target_version = args.game_version
    else:
        target_version = detect_installed_game_version(default_bar_data_dir())
        if target_version is None:
            sys.stderr.write(
                "Could not auto-detect your installed BAR game version.\n"
                "Pass --game-version 'Beyond All Reason test-XXXXX-YYYYYYY' "
                "or --any-version to disable the filter.\n"
            )
            return 1

    print(f"Map           : {args.map_name}")
    print(f"Players       : {players}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Max downloads : {args.max_downloads}")
    print(f"Sort by       : {args.sort_by} skill")
    if args.min_skill > 0:
        print(f"Min skill     : {args.min_skill}")
    if target_version:
        print(f"Game version  : {target_version}")
    else:
        print(f"Game version  : <any> (no filter)")
    print()

    # Step 1: list candidate replay IDs per player, dedup
    print("== Step 1: collecting candidate replay IDs ==")
    candidate_ids: set[str] = set()
    summaries: dict[str, dict[str, Any]] = {}
    for player in players:
        time.sleep(args.request_delay)
        replays = list_duels_for_player(player, args.map_name, args.pages_per_player)
        new = 0
        for r in replays:
            rid = r.get("id")
            if rid and rid not in candidate_ids:
                candidate_ids.add(rid)
                summaries[rid] = r
                new += 1
        print(f"  {player:25s}  {len(replays):3d} listed, {new:3d} new (total {len(candidate_ids)})")

    if not candidate_ids:
        print("\nNo replays found matching the filters.")
        return 1
    print(f"\n{len(candidate_ids)} unique candidate replays.\n")

    # Step 2: fetch detail for each (gives us fileName + skill info)
    print("== Step 2: fetching replay details ==")
    details: list[dict[str, Any]] = []
    rejected_version = 0
    rejected_aborted = 0
    rejected_skill = 0
    for i, rid in enumerate(sorted(candidate_ids), 1):
        time.sleep(args.request_delay)
        d = fetch_detail(rid)
        if d is None:
            continue
        if not d.get("gameEndedNormally", True):
            rejected_aborted += 1
            continue
        if d.get("hasBots"):
            continue  # safety
        if target_version and d.get("gameVersion") != target_version:
            rejected_version += 1
            continue
        max_skill, sum_skill, names = replay_skill_summary(d)
        if max_skill < args.min_skill:
            rejected_skill += 1
            continue
        details.append({
            "id": rid,
            "fileName": d.get("fileName"),
            "gameVersion": d.get("gameVersion"),
            "max_skill": max_skill,
            "sum_skill": sum_skill,
            "names": names,
            "durationMs": d.get("durationMs", 0),
        })
        if i % 25 == 0:
            print(f"  fetched {i}/{len(candidate_ids)} details, kept {len(details)}")

    print(f"\n{len(details)} replays passed filters.")
    if target_version:
        print(f"  rejected (wrong game version) : {rejected_version}")
    print(f"  rejected (aborted game)       : {rejected_aborted}")
    if args.min_skill > 0:
        print(f"  rejected (low skill)          : {rejected_skill}")
    print()
    if not details:
        return 1

    # Step 3: sort by skill, descending
    key = "max_skill" if args.sort_by == "max" else "sum_skill"
    details.sort(key=lambda r: r[key], reverse=True)

    print("== Step 3: downloading top {} ==".format(min(args.max_downloads, len(details))))
    output_dir: Path = args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for rec in details:
        if downloaded >= args.max_downloads:
            break
        fname = rec["fileName"]
        if not fname:
            continue
        dest = output_dir / fname
        encoded = quote(fname, safe="-_.()")
        url = f"{STORAGE_BASE}/{encoded}"

        size_kb = ""
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            print(f"  [skip] {fname}  (already exists, {dest.stat().st_size // 1024} KB)")
            continue

        if args.dry_run:
            print(f"  [DRY] {rec['max_skill']:5.2f}  {rec['names']}  {fname}")
            downloaded += 1
            continue

        try:
            time.sleep(args.request_delay)
            n = _download(url, dest)
            size_kb = f"{n // 1024} KB"
            downloaded += 1
            print(f"  [{downloaded:3d}/{args.max_downloads}] skill={rec['max_skill']:5.2f}  "
                  f"{size_kb:>8}  {fname}")
        except HTTPError as e:
            print(f"  ! HTTP {e.code} downloading {fname}", file=sys.stderr)
        except URLError as e:
            print(f"  ! URL error downloading {fname}: {e}", file=sys.stderr)

    print()
    print(f"Done. Downloaded {downloaded} new replay(s), skipped {skipped} already-present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
