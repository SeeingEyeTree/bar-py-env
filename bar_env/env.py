"""BarEnv: minimal pysc2-style environment wrapper.

v0.0.1 scope:
  - reset() launches a fresh headless game and blocks until the widget connects
  - step(actions) sends actions, returns next observation
  - close() requests the engine to quit and cleans up the socket / subprocess

Observation and action payloads are still raw dicts (mirroring the wire
protocol); the next version will lift them into numpy / Gymnasium spaces.
"""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from typing import Any

from bar_env import protocol
from bar_env.config import DEFAULT_MAP_NAME, DEFAULT_PORT
from bar_env.headless import HeadlessRun, launch
from bar_env.start_script import StartScriptConfig, render_start_script


@dataclasses.dataclass
class EnvConfig:
    map_name: str = DEFAULT_MAP_NAME
    port: int = DEFAULT_PORT
    step_interval: int = 30          # game frames between obs/action exchanges
    max_steps: int = 10              # widget-side safety cap
    connect_timeout_s: float = 60.0  # how long to wait for widget to connect
    step_timeout_s: float = 30.0     # per-step socket timeout
    bar_data_dir: Path | None = None
    # Optional: a fully-built StartScriptConfig. If set, it overrides
    # `map_name` (and everything else default about the start script) so
    # callers can specify start positions, factions, opponent AI, etc.
    script_config: StartScriptConfig | None = None
    # Multiplier for game simulation speed (1 = realtime, 20 = max safe).
    # Default matches headless.launch()'s prior behavior so this change is
    # backwards-compatible.
    speed: int = 20


class BarEnv:
    """Single-game BAR environment."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._run: HeadlessRun | None = None
        self._last_obs: dict[str, Any] | None = None
        # Set after close() so callers can still retrieve the demo path.
        self._last_run_demo: Path | None = None
        self._last_write_dir: Path | None = None
        self._closed = False

    # ----- lifecycle -----

    def reset(self) -> dict[str, Any]:
        """Start a fresh game and return the first observation."""
        self._cleanup_match(quit_engine=True)

        # 1. bind TCP server first so the widget can connect on game start
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.config.port))
        srv.listen(1)
        srv.settimeout(self.config.connect_timeout_s)
        self._server = srv

        # 2. spawn headless
        script_cfg = self.config.script_config or StartScriptConfig(
            map_name=self.config.map_name
        )
        script = render_start_script(script_cfg)
        self._run = launch(
            start_script_text=script,
            bar_data_dir=self.config.bar_data_dir,
            port=self.config.port,
            step_interval=self.config.step_interval,
            max_steps=self.config.max_steps,
            speed=self.config.speed,
        )
        self._last_write_dir = self._run.write_dir
        self._last_run_demo = None  # populated on close()

        # 3. wait for widget to dial in
        try:
            conn, _addr = srv.accept()
        except socket.timeout as e:
            # Capture the log path BEFORE cleanup wipes self._run.
            log_path = self._run.log_file if self._run else None
            write_dir = self._run.write_dir if self._run else None
            self._cleanup_match(quit_engine=True)
            raise TimeoutError(
                f"Widget didn't connect within {self.config.connect_timeout_s}s.\n"
                f"  headless log : {log_path}\n"
                f"  write dir    : {write_dir}"
            ) from e
        conn.settimeout(self.config.step_timeout_s)
        self._conn = conn

        # 4. handshake
        hello = protocol.recv_msg(conn)
        if hello.get("type") != "hello":
            raise protocol.ProtocolError(f"unexpected handshake from widget: {hello}")
        protocol.send_msg(conn, {"type": "hello", "ok": True})

        # 5. first real observation
        first = protocol.recv_msg(conn)
        if first.get("type") != "obs":
            raise protocol.ProtocolError(f"expected 'obs', got {first}")
        self._last_obs = first
        return first

    def step(self, actions: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        """Send an action list, return (next_obs, done)."""
        if self._conn is None:
            raise RuntimeError("step() called before reset()")
        protocol.send_msg(self._conn, {"type": "actions", "actions": actions})
        try:
            msg = protocol.recv_msg(self._conn)
        except protocol.ProtocolError:
            # Engine closed the connection without a terminal obs — game ended
            # (e.g. commander died, time-limit, etc.). Return the last known obs
            # marked as done so callers don't need to handle the exception.
            last = dict(self._last_obs or {})
            last.update({"done": True, "result": "disconnected"})
            return last, True
        if msg.get("type") != "obs":
            raise protocol.ProtocolError(f"expected 'obs', got {msg}")
        self._last_obs = msg
        return msg, bool(msg.get("done"))

    def quit(self) -> None:
        """Tell the widget to quit the engine. Safe to call multiple times."""
        if self._conn is not None:
            try:
                protocol.send_msg(self._conn, {"type": "quit"})
            except Exception:
                pass

    def demo_path(self) -> Path | None:
        """Path to the .sdfz replay produced by this match, if any.

        Only meaningful after the engine has finished writing it -- typically
        call this AFTER close() returns. The file lives under the run's
        write-dir (defaults to D:\\BAR_Replays on Windows when D: is present).
        """
        if self._run is None and self._last_run_demo is not None:
            return self._last_run_demo
        if self._run is not None:
            return self._run.find_demo()
        return None

    def close(self) -> None:
        if self._closed:
            return
        self.quit()
        # Capture the demo path before _cleanup_match wipes self._run.
        self._last_run_demo = self._run.find_demo() if self._run else None
        self._cleanup_match(quit_engine=True)
        # After engine shutdown, the demo may have been finalized; re-check.
        if self._last_run_demo is None and self._last_write_dir is not None:
            demos = sorted((self._last_write_dir / "demos").glob("*.sdfz")) \
                if (self._last_write_dir / "demos").is_dir() else []
            if demos:
                self._last_run_demo = demos[-1]
        self._closed = True

    # ----- internal -----

    def _cleanup_match(self, *, quit_engine: bool) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        if self._run is not None and quit_engine:
            # The widget handles quit in two phases: Spring.GameOver to fire
            # the gameover gadgets + flush the demo, then quitforce to exit.
            # On a long game with >1GB Lua RAM the cleanup can genuinely take
            # 30+ seconds, so give it 60s before SIGTERM-ing.
            if self._run.wait(timeout=60) is None:
                self._run.terminate(timeout=10)
            self._run = None

    def __enter__(self) -> "BarEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
