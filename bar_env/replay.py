"""Path B replay extractor.

Launches spring-headless on a .sdfz with the bridge widget in replay mode,
receives the (observation, commands-this-frame) stream from the widget over
TCP, and dumps it to a JSONL file suitable for behavior-cloning preprocessing.

No actions are sent back -- the widget is a sink during replay extraction.
"""

from __future__ import annotations

import gzip
import json
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bar_env import protocol
from bar_env.config import DEFAULT_PORT, default_bar_data_dir
from bar_env.headless import HeadlessRun, launch_replay


# --------------------------------------------------------------------------
# Replay metadata helpers
# --------------------------------------------------------------------------

_GAMETYPE_RE = re.compile(rb"[Gg]ame[Tt]ype\s*=\s*([^;\r\n]+);")
_DEMO_ENGINE_VER_RE = re.compile(r"_(\d{4}\.\d{2}\.\d{2})\.sdfz$", re.IGNORECASE)


def get_demo_engine_version(sdfz_path: Path) -> str | None:
    """Return the Spring engine version embedded in the demo filename.

    BAR filenames follow the convention '<date>_<map>_<engine_ver>.sdfz',
    e.g. '2026-02-19_..._2025.06.19.sdfz' → '2025.06.19'.
    """
    m = _DEMO_ENGINE_VER_RE.search(sdfz_path.name)
    return m.group(1) if m else None


def detect_installed_game_version(bar_data_dir: Path) -> str | None:
    """Return the 'Beyond All Reason test-XXXXX-YYYYYYY' currently installed.

    Reads <bar_data>/rapid/repos-cdn.beyondallreason.dev/byar/versions.gz and
    finds the archive name the `byar:test` tag points to. Returns None if the
    file is missing or unreadable.
    """
    versions_gz = (
        bar_data_dir
        / "rapid"
        / "repos-cdn.beyondallreason.dev"
        / "byar"
        / "versions.gz"
    )
    if not versions_gz.exists():
        return None
    try:
        with gzip.open(versions_gz, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",")
                if len(parts) >= 4 and parts[0] == "byar:test":
                    return parts[3].strip()
    except OSError:
        return None
    return None


def get_demo_game_type(sdfz_path: Path) -> str | None:
    """Return the demo's `gametype` string ("Beyond All Reason test-XXXX-YYYY").

    The startscript is stored in the demo's header as plain text immediately
    after a 352-byte struct. Decompressing the first ~64 KB is more than
    enough to reach it without parsing the binary header layout.
    """
    try:
        with gzip.open(sdfz_path, "rb") as f:
            head = f.read(65536)
    except OSError:
        return None
    m = _GAMETYPE_RE.search(head)
    if not m:
        return None
    return m.group(1).decode("utf-8", errors="ignore").strip()


def gametype_to_rapid_tag(gametype: str) -> str | None:
    """Map a `gametype` string back to a rapid-resolvable tag.

    "Beyond All Reason test-30154-d7430fc" -> "byar:test-30154-d7430fc"
    """
    if not gametype:
        return None
    g = gametype.strip()
    prefix = "Beyond All Reason "
    if g.lower().startswith(prefix.lower()):
        return "byar:" + g[len(prefix):]
    return None


def find_pr_downloader(bar_data_dir: Path) -> Path | None:
    """Locate pr-downloader.exe in the user's BAR install."""
    candidates = list(bar_data_dir.glob("engine/*/pr-downloader.exe"))
    candidates += list(bar_data_dir.glob("engine/*/pr-downloader"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_game_archive(
    rapid_tag: str, bar_data_dir: Path, timeout_s: float = 300.0
) -> bool:
    """Try to fetch a rapid tag into the user's BAR pool. Returns True on success.

    pr-downloader is a no-op when the archive is already installed, so it's
    safe to call before every replay launch.
    """
    pr = find_pr_downloader(bar_data_dir)
    if pr is None:
        print(f"  [warn] pr-downloader not found in {bar_data_dir}/engine/*/")
        return False
    print(f"  fetching '{rapid_tag}' via pr-downloader ...")
    try:
        result = subprocess.run(
            [
                str(pr),
                "--filesystem-writepath", str(bar_data_dir),
                "--download-game", f"rapid://{rapid_tag}",
            ],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  [warn] pr-downloader failed: {exc}")
        return False
    if result.returncode != 0:
        # Newer versions of the rapid tag may not exist; not fatal but useful to surface.
        print(f"  [warn] pr-downloader exited {result.returncode}")
        if result.stderr:
            print(f"         stderr: {result.stderr.strip()[:400]}")
        return False
    return True


@dataclass
class ReplayExtractorConfig:
    replay_path: Path
    output_path: Path                 # .jsonl to write to
    port: int = DEFAULT_PORT
    step_interval: int = 30           # game frames between extracted observations
    max_steps: int = 0                # safety cap; 0 = no cap, end on GameOver
    speed: int = 20                   # gamespeed multiplier; capped by maxspeed
    connect_timeout_s: float = 240.0  # demo loads can be slow on first run
    step_timeout_s: float = 60.0
    bar_data_dir: Path | None = None
    progress_every: int = 30          # how often to print a progress line
    # If True, try to fetch the demo's game-archive via pr-downloader before
    # launch. Required for replays older than your currently-installed BAR
    # version. Safe no-op when the archive is already on disk.
    fetch_missing_archive: bool = True
    # Write a compact JSONL: map info moved to a one-time header line,
    # redundant fields dropped, numeric precision reduced.
    compact: bool = True


# ---------------------------------------------------------------------------
# Compact output helpers
# ---------------------------------------------------------------------------

def _rnd(v: float, decimals: int) -> float | int:
    """Round v; return int when decimals==0 for smaller JSON output."""
    if decimals == 0:
        return int(round(v))
    return round(float(v), decimals)


def _compact_obs(msg: dict[str, Any]) -> dict[str, Any]:
    """Strip redundant fields and lower numeric precision for training data."""
    out: dict[str, Any] = {"frame": msg["frame"]}
    if msg.get("done"):
        out["done"] = True
    if msg.get("result") is not None:
        out["result"] = msg["result"]

    teams: dict[str, Any] = {}
    for tid, t in (msg.get("teams") or {}).items():
        units = []
        for u in (t.get("units") or []):
            entry: dict[str, Any] = {
                "id":     u["id"],
                "def":    u["def"],
                "x":      _rnd(u.get("x", 0), 0),
                "y":      _rnd(u.get("y", 0), 1),
                "z":      _rnd(u.get("z", 0), 0),
                "hp":     _rnd(u.get("hp", 0), 1),
                "max_hp": int(u.get("max_hp", 0)),
                "h":      int(u.get("h", 0)),        # compact heading 0-255
            }
            if u.get("bp") is not None:              # build progress 0-1
                entry["bp"] = round(float(u["bp"]), 2)
            # Visibility: ally team IDs that can see this unit.
            if u.get("los"):
                entry["los"] = u["los"]
            if u.get("radar"):
                entry["radar"] = u["radar"]
            units.append(entry)
        teams[tid] = {
            "metal":          _rnd(t.get("metal", 0), 1),
            "metal_income":   _rnd(t.get("metal_income", 0), 2),
            "metal_pull":     _rnd(t.get("metal_pull", 0), 2),
            "metal_storage":  int(t.get("metal_storage", 1000)),
            "energy":         _rnd(t.get("energy", 0), 1),
            "energy_income":  _rnd(t.get("energy_income", 0), 2),
            "energy_pull":    _rnd(t.get("energy_pull", 0), 2),
            "energy_storage": int(t.get("energy_storage", 1000)),
            "ally":           t.get("ally", 0),
            "units":          units,
        }
    out["teams"] = teams

    # Features: wrecks + reclaimable terrain (trees, rocks, mexes).
    # Only features with remaining metal or energy are recorded.
    # m/e are the actual remaining amounts (total * reclaimLeft from Lua).
    feats = []
    for f in (msg.get("features") or []):
        entry = {
            "id":  f["id"],
            "def": f["def"],
            "x":   int(f.get("x", 0)),
            "z":   int(f.get("z", 0)),
            "m":   _rnd(f.get("m", 0), 1),
        }
        if f.get("e", 0) > 0:
            entry["e"] = _rnd(f["e"], 1)
        if f.get("los"):
            entry["los"] = f["los"]
        feats.append(entry)
    if feats:
        out["features"] = feats

    cmds = []
    for c in (msg.get("commands_this_frame") or []):
        # Only keep modifier flags that are actually set.
        opts = {k: True for k in ("shift", "ctrl", "alt", "right")
                if c.get("options", {}).get(k)}
        entry: dict[str, Any] = {
            "frame":  c["frame"],
            "team":   c.get("team"),
            "unit":   c.get("unit"),
            "cmd_id": c.get("cmd_id"),
        }
        params = c.get("params")
        if params:
            entry["params"] = [_rnd(p, 1) for p in params]
        if opts:
            entry["options"] = opts
        cmds.append(entry)
    if cmds:
        out["commands_this_frame"] = cmds

    return out


def _make_header(msg: dict[str, Any], game_version: str | None) -> dict[str, Any]:
    """First line of a compact JSONL: map metadata + game version."""
    hdr: dict[str, Any] = {"type": "header"}
    if msg.get("map"):
        hdr["map"] = msg["map"]
    if game_version:
        hdr["game_version"] = game_version
    return hdr


def _tail(path: Path | None, n: int = 12) -> str:
    """Return the last `n` lines of a text/utf-8 file (best-effort)."""
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def extract(cfg: ReplayExtractorConfig) -> dict[str, Any]:
    """Run the engine and write JSONL. Returns a small summary dict."""
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    server: socket.socket | None = None
    conn: socket.socket | None = None
    run: HeadlessRun | None = None
    frames_written = 0
    commands_written = 0
    last_frame = -1

    try:
        # 0. Pre-flight: make sure the demo's game-archive is installed.
        # Replays embed an exact "Beyond All Reason test-XXXXX-YYYYYYY" version
        # in their header; if the user's BAR install doesn't have that version
        # the engine fails fast with a content_error before our widget can ever
        # connect. pr-downloader will fetch missing archives into the rapid pool.
        # We skip the fetch if the demo's version is already what `byar:test`
        # currently resolves to in the user's local rapid metadata -- calling
        # pr-downloader unnecessarily wastes time and can disturb a working
        # install (the launcher sometimes leaves the pool in a state where
        # pr-downloader rewrites files in a way the engine doesn't like).
        bar_data = cfg.bar_data_dir or default_bar_data_dir()
        if cfg.fetch_missing_archive:
            gtype = get_demo_game_type(cfg.replay_path)
            rapid_tag = gametype_to_rapid_tag(gtype) if gtype else None
            installed = detect_installed_game_version(bar_data)
            if rapid_tag:
                print(f"  demo gametype : {gtype}")
                print(f"  rapid tag     : {rapid_tag}")
                if installed:
                    print(f"  installed     : {installed}")
                if installed and gtype == installed:
                    print("  pre-flight    : skipped (demo matches installed version)")
                else:
                    ensure_game_archive(rapid_tag, bar_data)
            else:
                print(f"  [warn] could not extract gametype from {cfg.replay_path}; "
                      f"skipping pre-flight archive fetch.")

        # 1. bind TCP server first so the widget can connect on game start
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", cfg.port))
        srv.listen(1)
        # We'll use a short timeout and poll the subprocess in a loop so that
        # an engine exit (e.g. content_error) surfaces immediately instead of
        # waiting the full connect_timeout_s.
        srv.settimeout(1.0)
        server = srv

        # 2. launch the engine on the replay
        run = launch_replay(
            replay_path=cfg.replay_path,
            bar_data_dir=cfg.bar_data_dir,
            port=cfg.port,
            step_interval=cfg.step_interval,
            max_steps=cfg.max_steps,
            speed=cfg.speed,
        )

        # 3. accept the widget's connection -- but bail out fast if the engine
        # has already exited (the common failure mode here).
        deadline = time.monotonic() + cfg.connect_timeout_s
        conn = None
        while time.monotonic() < deadline:
            try:
                conn, _addr = srv.accept()
                break
            except socket.timeout:
                if run.process.poll() is not None:
                    raise RuntimeError(
                        "spring-headless exited before the widget could "
                        f"connect (exit code {run.process.returncode}).\n"
                        f"  headless log tail:\n"
                        f"    " + _tail(run.log_file, 8).replace("\n", "\n    ")
                    )
        if conn is None:
            raise TimeoutError(
                f"Widget didn't connect within {cfg.connect_timeout_s}s.\n"
                f"  headless log : {run.log_file}\n"
                f"  write dir    : {run.write_dir}"
            )
        conn.settimeout(cfg.step_timeout_s)

        # 4. handshake
        hello = protocol.recv_msg(conn)
        if hello.get("type") != "hello":
            raise protocol.ProtocolError(f"unexpected handshake: {hello}")
        protocol.send_msg(conn, {"type": "hello", "ok": True})

        # 5. receive obs stream until done
        game_version = get_demo_game_type(cfg.replay_path) if cfg.compact else None
        header_written = False
        with open(cfg.output_path, "w", encoding="utf-8") as out:
            while True:
                try:
                    msg = protocol.recv_msg(conn)
                except (protocol.ProtocolError, socket.timeout, ConnectionError):
                    # Widget disconnected (likely from quitforce). That's the
                    # expected end of stream after a terminal observation.
                    break
                if msg.get("type") != "obs":
                    continue
                if cfg.compact:
                    if not header_written:
                        hdr = _make_header(msg, game_version)
                        out.write(json.dumps(hdr, separators=(",", ":")))
                        out.write("\n")
                        header_written = True
                    row = _compact_obs(msg)
                else:
                    row = msg
                out.write(json.dumps(row, separators=(",", ":")))
                out.write("\n")
                frames_written += 1
                commands_written += len(msg.get("commands_this_frame") or [])
                last_frame = msg.get("frame", last_frame)
                if frames_written % cfg.progress_every == 0:
                    print(
                        f"  frame={last_frame:<6d}  "
                        f"obs={frames_written}  cmds={commands_written}"
                    )
                if msg.get("done"):
                    break
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass
        if server is not None:
            try: server.close()
            except Exception: pass
        if run is not None:
            # Wait for engine to exit cleanly (it should from the GameOver hook).
            if run.wait(timeout=60) is None:
                run.terminate(timeout=10)

    return {
        "replay": str(cfg.replay_path),
        "output": str(cfg.output_path),
        "frames_written": frames_written,
        "commands_written": commands_written,
        "last_frame": last_frame,
        "headless_log": str(run.log_file) if run else None,
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Convenience reader for the JSONL the extractor writes."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
