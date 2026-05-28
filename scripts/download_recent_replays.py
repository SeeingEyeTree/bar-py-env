#!/usr/bin/env python3
"""Download all recent BAR 1v1 replays matching a map + faction criteria.

Unlike download_replays.py (which queries by specific player names), this
script fetches ALL 1v1 replays on a given map from the last N hours and
keeps any where at least one player chose the specified faction.

Useful for building broad training sets without curating a player list.

Usage:
    python scripts/download_recent_replays.py
    python scripts/download_recent_replays.py --hours 48 --faction Cortex
    python scripts/download_recent_replays.py --dry-run
    python scripts/download_recent_replays.py --game-version "Beyond All Reason test-30220-b67c43a"

Defaults:
    --map      "Comet Catcher Remake 1.8"
    --faction  Cortex
    --hours    24
    --output   D:/BAR_Replays/training_data
    (no version filter — all versions downloaded; use batch_extract_replays.py to extract)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bar_env.config import default_bar_data_dir  # noqa: E402

API_BASE = "https://api.bar-rts.com/replays"
STORAGE_BASE = (
    "https://storage.uk.cloud.ovh.net/v1/"
    "AUTH_10286efc0d334efd917d476d7183232e/BAR/demos"
)
USER_AGENT = "bar-py-env/0.0.1 (recent-replay-downloader)"
PAGE_SIZE = 50
REQUEST_DELAY = 0.15  # seconds between API calls

DEFAULT_MAP    = "Comet Catcher Remake 1.8"
DEFAULT_FACTION = "Cortex"
DEFAULT_HOURS  = 24
DEFAULT_OUTPUT = Path("D:/BAR_Replays/training_data")
MIN_DURATION_S = 120  # skip games shorter than 2 minutes (likely crashes/surrenders)


# ---------------------------------------------------------------------------
# Version detection (shared with download_replays.py)
# ---------------------------------------------------------------------------

import gzip as _gzip

_BYAR_VERSIONS_GZ = Path(
    "rapid/repos-cdn.beyondallreason.dev/byar/versions.gz"
)


def detect_installed_game_version(bar_data_dir: Path) -> str | None:
    versions_gz = bar_data_dir / _BYAR_VERSIONS_GZ
    if not versions_gz.exists():
        return None
    try:
        with _gzip.open(versions_gz, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",")
                if len(parts) >= 4 and parts[0] == "byar:test":
                    return parts[3].strip()
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 30.0) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest: Path, timeout: float = 180.0) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        data = resp.read()
        f.write(data)
        return len(data)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _parse_time(ts: str) -> datetime:
    """Parse ISO 8601 UTC timestamp from API."""
    ts = ts.rstrip("Z")
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=timezone.utc)


def fetch_recent_ids(map_name: str, cutoff: datetime) -> list[str]:
    """Page through the replay list newest-first, stop when we pass the cutoff.

    Returns list of replay IDs whose startTime >= cutoff.
    """
    ids: list[str] = []
    page = 1
    while True:
        q = urlencode({
            "preset": "duel",
            "hasBots": "false",
            "maps": map_name,
            "limit": PAGE_SIZE,
            "page": page,
        })
        try:
            payload = _get_json(f"{API_BASE}?{q}")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  ! list page {page} failed: {exc}", file=sys.stderr)
            break

        rows = payload.get("data") or []
        if not rows:
            break

        reached_cutoff = False
        for r in rows:
            ts = r.get("startTime") or ""
            if not ts:
                continue
            try:
                dt = _parse_time(ts)
            except ValueError:
                continue
            if dt < cutoff:
                reached_cutoff = True
                break
            ids.append(r["id"])

        print(f"  page {page}: {len(rows)} results, {len(ids)} in window so far")

        if reached_cutoff or len(rows) < PAGE_SIZE:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return ids


def fetch_detail(replay_id: str) -> dict[str, Any] | None:
    try:
        time.sleep(REQUEST_DELAY)
        return _get_json(f"{API_BASE}/{replay_id}")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  ! detail failed for {replay_id[:8]}: {exc}", file=sys.stderr)
        return None


def has_faction(detail: dict, faction: str) -> bool:
    """Return True if at least one human player chose this faction."""
    for at in detail.get("AllyTeams", []) or []:
        for p in at.get("Players", []) or []:
            if (p.get("faction") or "").lower() == faction.lower():
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--map", dest="map_name", default=DEFAULT_MAP,
                   help=f"Map to filter on. Default: '{DEFAULT_MAP}'")
    p.add_argument("--faction", default=DEFAULT_FACTION,
                   help=f"Required faction (Cortex or Armada). Default: {DEFAULT_FACTION}")
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                   help=f"How many hours back to look. Default: {DEFAULT_HOURS}")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Directory to save .sdfz files. Default: {DEFAULT_OUTPUT}")
    p.add_argument("--game-version", default=None,
                   help="Only download replays matching this exact game version string. "
                        "Default: no version filter (download any version).")
    p.add_argument("--min-duration", type=int, default=MIN_DURATION_S,
                   help=f"Skip replays shorter than this many seconds. "
                        f"Default: {MIN_DURATION_S}")
    p.add_argument("--dry-run", action="store_true",
                   help="Print matches without downloading.")
    args = p.parse_args()

    # Resolve version filter — default is no filter (download any version).
    # Pass --game-version explicitly to restrict to one specific version.
    target_version: str | None = args.game_version or None

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=args.hours)

    print(f"Map           : {args.map_name}")
    print(f"Faction       : {args.faction}")
    print(f"Window        : last {args.hours:.0f} hours (since {cutoff.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"Game version  : {target_version or '<any>'}")
    print(f"Min duration  : {args.min_duration}s")
    print(f"Output        : {args.output}")
    if args.dry_run:
        print("  [DRY RUN - no files will be downloaded]")
    print()

    # Step 1: collect IDs in time window
    print("== Step 1: listing recent replays ==")
    candidate_ids = fetch_recent_ids(args.map_name, cutoff)
    print(f"\n{len(candidate_ids)} replays in the window.\n")
    if not candidate_ids:
        print("Nothing to download.")
        return 0

    # Step 2: filter by faction, gameEndedNormally, version, duration
    print("== Step 2: fetching details and applying filters ==")
    to_download: list[dict] = []
    rej_ended   = 0
    rej_faction = 0
    rej_version = 0
    rej_short   = 0

    for i, rid in enumerate(candidate_ids, 1):
        d = fetch_detail(rid)
        if d is None:
            continue

        if not d.get("gameEndedNormally", True):
            rej_ended += 1
            continue

        dur_s = (d.get("durationMs") or 0) / 1000
        if dur_s < args.min_duration:
            rej_short += 1
            continue

        if target_version and d.get("gameVersion") != target_version:
            rej_version += 1
            continue

        if not has_faction(d, args.faction):
            rej_faction += 1
            continue

        fname = d.get("fileName")
        if not fname:
            continue

        players = [p["name"] for at in d.get("AllyTeams", [])
                   for p in at.get("Players", [])]
        to_download.append({
            "id":       rid,
            "fileName": fname,
            "players":  players,
            "dur_s":    int(dur_s),
            "version":  d.get("gameVersion", ""),
        })

        if i % 20 == 0:
            print(f"  checked {i}/{len(candidate_ids)}, queued {len(to_download)}")

    print(f"\n{len(to_download)} replays passed all filters.")
    print(f"  rejected (abnormal end)   : {rej_ended}")
    print(f"  rejected (no {args.faction} player) : {rej_faction}")
    if target_version:
        print(f"  rejected (wrong version)  : {rej_version}")
    print(f"  rejected (too short <{args.min_duration}s) : {rej_short}")
    print()

    if not to_download:
        print("Nothing to download.")
        return 0

    # Step 3: download
    print(f"== Step 3: downloading {len(to_download)} replay(s) ==")
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0
    for rec in to_download:
        fname = rec["fileName"]
        dest  = args.output / fname
        url   = f"{STORAGE_BASE}/{quote(fname, safe='-_.()')}"

        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            print(f"  [skip] {fname[:60]}  ({dest.stat().st_size // 1024} KB already)")
            continue

        players_str = " vs ".join(rec["players"])
        if args.dry_run:
            print(f"  [DRY]  {rec['dur_s']//60}:{rec['dur_s']%60:02d}  {players_str}  {fname[:50]}")
            downloaded += 1
            continue

        try:
            time.sleep(REQUEST_DELAY)
            n = _download_file(url, dest)
            downloaded += 1
            print(f"  [{downloaded:3d}] {n // 1024:>6} KB  {rec['dur_s']//60}:{rec['dur_s']%60:02d}  "
                  f"{players_str}  {fname[:45]}")
        except HTTPError as e:
            failed += 1
            print(f"  ! HTTP {e.code}: {fname}", file=sys.stderr)
        except URLError as e:
            failed += 1
            print(f"  ! URL error: {e}: {fname[:40]}", file=sys.stderr)

    print()
    if args.dry_run:
        print(f"Dry run: {downloaded} would be downloaded, {skipped} already present.")
    else:
        print(f"Done. {downloaded} downloaded, {skipped} skipped (already present), {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
