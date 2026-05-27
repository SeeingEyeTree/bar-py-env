"""Launching and managing the spring-headless subprocess."""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from bar_env.config import (
    BRIDGE_WIDGET_PATH,
    DEFAULT_PORT,
    RUN_DIR,
    default_bar_data_dir,
    headless_binary,
)


@dataclasses.dataclass
class HeadlessRun:
    """Handle to a running spring-headless subprocess + its write-dir."""

    process: subprocess.Popen
    write_dir: Path
    log_file: Path

    def is_running(self) -> bool:
        return self.process.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self, timeout: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def find_demo(self) -> Path | None:
        """Return the path to the recorded demo (.sdfz), if one exists.

        Recoil's demo-writing target depends on isolation setup. Under our
        --isolation + SpringData=<BAR install> setup, demos actually land in
        the BAR install's demos/ folder (which is convenient -- Chobby's
        replay viewer picks them up automatically). We also check the run's
        write-dir as a fallback for engine versions / setups that route them
        there.

        Returns the most recently modified .sdfz, scoped to demos newer than
        when this run started, so we don't accidentally pick up a stale demo
        from a previous BAR session.
        """
        start_mtime = self.write_dir.stat().st_mtime
        candidate_dirs = [self.write_dir / "demos"]
        bar_data = default_bar_data_dir()
        if bar_data and (bar_data / "demos").is_dir():
            candidate_dirs.append(bar_data / "demos")

        newest: Path | None = None
        newest_mtime = 0.0
        for d in candidate_dirs:
            if not d.is_dir():
                continue
            for f in d.glob("*.sdfz"):
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if mt >= start_mtime and mt > newest_mtime:
                    newest = f
                    newest_mtime = mt
        return newest


def _new_write_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_DIR / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_springsettings(write_dir: Path, bar_data_dir: Path, port: int) -> None:
    # SpringData tells the engine where to find game/map data dirs.
    # On Windows we use backslashes; springsettings.cfg tolerates them.
    spring_data = str(bar_data_dir)
    (write_dir / "springsettings.cfg").write_text(
        "\n".join([
            f"SpringData = {spring_data}",
            "LuaSocketEnabled = 1",
            # Allow connections to localhost on our port. Modern engines allow
            # all TCP by default but we make it explicit anyway.
            f"TCPAllowConnect = 127.0.0.1:{port}",
            "TCPAllowListen = ",
            "LogFlushLevel = 0",
            "",
        ]),
        encoding="utf-8",
    )


def _install_bridge_widget(
    write_dir: Path, port: int, step_interval: int, max_steps: int,
    mode: str = "live", speed: int = 20,
) -> None:
    widgets_dir = write_dir / "LuaUI" / "Widgets"
    widgets_dir.mkdir(parents=True, exist_ok=True)
    if not BRIDGE_WIDGET_PATH.exists():
        raise FileNotFoundError(f"Bridge widget not found at {BRIDGE_WIDGET_PATH}")
    shutil.copyfile(BRIDGE_WIDGET_PATH, widgets_dir / "bridge_widget.lua")

    # Start from a clean LuaUI/Config so we control widget ordering.
    config_dir = write_dir / "LuaUI" / "Config"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Pre-enable our widget in BAR's widget-order config. Without this entry,
    # barwidgets.lua line 585 disables user widgets by default even when the
    # `allowuserwidgets` modoption is set. The engine merges this with the
    # mod's default config on first run, so mod widgets still auto-enable.
    # BAR's Game.gameShortName is "BYAR", which determines the config path:
    # `LuaUI/Config/BYAR.lua`.
    (config_dir / "BYAR.lua").write_text(
        "\n".join([
            "-- Pre-seeded by bar-py-env to force the bridge widget on.",
            "return {",
            "    allowUserWidgets = true,",
            "    data = {},",
            "    order = {",
            '        ["BAR Python Env Bridge"] = 1,',
            "    },",
            "}",
            "",
        ]),
        encoding="utf-8",
    )

    # Drop a small key=value config file the widget reads on Initialize.
    (config_dir / "bar_py_env.txt").write_text(
        "\n".join([
            "host=127.0.0.1",
            f"port={port}",
            f"step_interval={step_interval}",
            f"max_steps={max_steps}",
            f"mode={mode}",
            f"speed={speed}",
            "",
        ]),
        encoding="utf-8",
    )


def launch(
    *,
    start_script_text: str,
    bar_data_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    step_interval: int = 30,
    max_steps: int = 10,
    speed: int = 20,
    extra_args: list[str] | None = None,
    capture_log: bool = True,
) -> HeadlessRun:
    """Launch spring-headless with the given start script.

    Returns a HeadlessRun handle. Caller is responsible for waiting on or
    terminating the subprocess (or, normally, just letting the bridge widget
    quit it via SendCommands("quit")).
    """
    binary = headless_binary()
    bar_data_dir = Path(bar_data_dir) if bar_data_dir else default_bar_data_dir()
    if not bar_data_dir.exists():
        raise FileNotFoundError(
            f"BAR data dir not found at {bar_data_dir}.\n"
            f"Pass bar_data_dir=... or install BAR to the default location."
        )

    write_dir = _new_write_dir()
    _write_springsettings(write_dir, bar_data_dir, port)
    _install_bridge_widget(
        write_dir, port=port, step_interval=step_interval,
        max_steps=max_steps, speed=speed,
    )

    script_path = write_dir / "startscript.txt"
    script_path.write_text(start_script_text, encoding="utf-8")

    log_path = write_dir / "headless.log"
    log_handle = open(log_path, "wb") if capture_log else None

    args = [
        str(binary),
        "--isolation",
        "--write-dir", str(write_dir),
        str(script_path),
    ]
    if extra_args:
        args.extend(extra_args)

    env = os.environ.copy()
    # Tell the engine to use our SpringData (also picked up from springsettings.cfg).
    env.setdefault("SPRING_DATADIR", str(bar_data_dir))

    proc = subprocess.Popen(
        args,
        cwd=str(write_dir),
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
        env=env,
    )
    return HeadlessRun(process=proc, write_dir=write_dir, log_file=log_path)


def detect_engine_version() -> str | None:
    """Return the Spring version string of the installed engine binary.

    Runs 'spring-headless --version' and parses the output line:
    'spring-headless.exe version 2025.06.19 (Headless)' → '2025.06.19'
    Returns None if the binary can't be found or the output is unexpected.
    """
    try:
        binary = headless_binary()
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in (result.stdout + result.stderr).splitlines():
            parts = line.split()
            if "version" in parts:
                idx = parts.index("version")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    except Exception:
        pass
    return None


def wait_for_process_or_die(run: HeadlessRun, deadline_s: float) -> None:
    """Block until either the headless process exits or the deadline passes."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if run.process.poll() is not None:
            return
        time.sleep(0.5)


def launch_replay(
    *,
    replay_path: Path,
    bar_data_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    step_interval: int = 30,
    max_steps: int = 0,  # 0 = no cap; let the replay end naturally on GameOver
    speed: int = 20,
    extra_args: list[str] | None = None,
    capture_log: bool = True,
) -> HeadlessRun:
    """Launch spring-headless to play back a .sdfz with the widget in replay mode.

    The engine recognizes a .sdfz launch target and enters replay mode
    automatically; BAR's barwidgets.lua force-enables user widgets while
    replaying or spectating, so we don't need any modoptions for that.
    """
    replay_path = Path(replay_path)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay not found: {replay_path}")

    binary = headless_binary()
    bar_data_dir = Path(bar_data_dir) if bar_data_dir else default_bar_data_dir()
    if not bar_data_dir.exists():
        raise FileNotFoundError(
            f"BAR data dir not found at {bar_data_dir}.\n"
            f"Pass bar_data_dir=... or install BAR to the default location."
        )

    write_dir = _new_write_dir()
    _write_springsettings(write_dir, bar_data_dir, port)
    _install_bridge_widget(
        write_dir, port=port, step_interval=step_interval,
        max_steps=max_steps, mode="replay", speed=speed,
    )

    log_path = write_dir / "headless.log"
    log_handle = open(log_path, "wb") if capture_log else None

    args = [
        str(binary),
        "--isolation",
        "--write-dir", str(write_dir),
        str(replay_path),
    ]
    if extra_args:
        args.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("SPRING_DATADIR", str(bar_data_dir))

    proc = subprocess.Popen(
        args,
        cwd=str(write_dir),
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
        env=env,
    )
    return HeadlessRun(process=proc, write_dir=write_dir, log_file=log_path)
