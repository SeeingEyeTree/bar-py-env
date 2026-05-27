# bar-py-env

A pysc2-style Python learning environment for **Beyond All Reason** (BAR), built on the **Recoil engine**.

> v0.0.1 — proof-of-concept end-to-end loop only. A random agent connects to a headless BAR game over TCP and issues random move orders. Real observation spaces, feature layers, and a proper Gym/Gymnasium env come next.

## Architecture

```
+--------------------+        TCP        +-----------------------+
|  Python process    | <---------------> |  Recoil widget        |
|  - BarEnv          |   length-prefix   |  - reads Spring.* API |
|  - reset/step/close|   JSON            |  - issues orders      |
+--------------------+                   |  - runs in spring-    |
                                         |    headless           |
                                         +-----------------------+
```

Python is the TCP server. The widget connects to it from inside `spring-headless` once the game starts.

## v1 MVP requirements

- Beyond All Reason already installed at the default location (`%LOCALAPPDATA%\Programs\Beyond-All-Reason\data\`). The package uses your existing BAR data dir for game / map content.
- Python 3.11+.
- Windows (Linux support comes later — same code, different binary).

## First-time setup

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Download the headless engine (~200 MB, one-time):
python scripts/fetch_engine.py
```

`fetch_engine.py` reads `https://launcher-config.beyondallreason.dev/config.json`, finds the latest stable Windows engine, downloads the 7z release from the RecoilEngine GitHub, and extracts it to `./engine/recoil_<version>/`. It then verifies that `spring-headless.exe` is present.

> 7-Zip is required for extraction. The script tries `7z`, `7za`, and `7z.exe` on PATH; install [7-Zip](https://www.7-zip.org/) if it can't find one.

## Smoke test (random agent end-to-end)

```powershell
python examples/random_agent.py
```

What this does:

1. Binds a TCP server on `127.0.0.1:8765`.
2. Generates a startscript for Comet Catcher Remake 1.8 (your team vs a NullAI).
3. Launches `spring-headless.exe` with that startscript and our bridge widget pre-installed in an isolated write-dir.
4. The widget connects back to the Python server.
5. Python runs 10 observation/action steps, sending random `MOVE` orders to all owned units.
6. Python sends `quit`; widget quits the engine; subprocess exits.
7. Python prints `OK` and returns.

If you see `OK` at the end, the full Python ↔ headless engine ↔ Lua widget loop is working.

## Package layout

```
bar-py-env/
├── pyproject.toml
├── README.md
├── bar_env/
│   ├── config.py          # paths, defaults
│   ├── protocol.py        # wire format (length-prefixed JSON)
│   ├── start_script.py    # programmatic startscript.txt generation
│   ├── headless.py        # spring-headless subprocess management
│   └── env.py             # BarEnv: reset / step / close
├── lua/
│   └── bridge_widget.lua  # in-game TCP client widget
├── scripts/
│   └── fetch_engine.py    # one-time engine download
└── examples/
    └── random_agent.py    # end-to-end smoke test
```

## Status

- [x] Feasibility doc
- [x] Engine fetcher
- [x] Wire protocol (JSON)
- [x] Lua bridge widget (unsynced, fog-of-war)
- [x] Python env: launch headless, reset/step/close
- [x] Random agent smoke test
- [ ] Gymnasium-compatible observation/action spaces
- [ ] Numpy feature layers
- [ ] God's-eye (synced gadget) mode
- [ ] Vectorized envs
- [ ] Replay parsing (separate package or submodule)

## License

MIT.
