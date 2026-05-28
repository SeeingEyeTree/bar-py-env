#!/usr/bin/env python3
"""Render a BAR replay JSONL file to an .mp4 video.

Each frame is drawn as a top-down map view with units colored by cluster type,
team identity shown via a colored outer ring, and a resource HUD at the top.

Usage
-----
    python examples/visualize_replay.py --jsonl path/to/replay.jsonl
    python examples/visualize_replay.py --jsonl path/to/replay.jsonl --grid
    python examples/visualize_replay.py --jsonl path/to/replay.jsonl --fog --perspective 0

Options
-------
    --jsonl PATH        Input JSONL replay file [required]
    --output PATH       Output .mp4 file [default: <jsonl_stem>.mp4 beside the jsonl]
    --fps INT           Video framerate [default: 15]
    --width INT         Render width in pixels [default: 1024]
    --fog               Enable approximate fog-of-war circles
    --perspective INT   Team PoV for fog: 0 or 1. -1 = god's-eye [default: -1]
    --grid              Overlay semi-transparent grid cells (unit aggregation view)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import cv2  # noqa: E402

from bar_env.replay import iter_jsonl          # noqa: E402
from bar_env.vis import ReplayRenderer, draw_grid_overlay  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--jsonl", type=Path, required=True,
                   help="Input JSONL replay file.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .mp4 path. Default: <jsonl_stem>.mp4 in same folder.")
    p.add_argument("--fps", type=int, default=15,
                   help="Video framerate (default: 15).")
    p.add_argument("--width", type=int, default=1024,
                   help="Render width in pixels (default: 1024).")
    p.add_argument("--fog", action="store_true",
                   help="Enable approximate fog-of-war circles (black background).")
    p.add_argument("--perspective", type=int, default=-1,
                   help="Team index (0 or 1) for fog PoV. -1 = god's-eye (default).")
    p.add_argument("--grid", action="store_true",
                   help="Overlay semi-transparent 64x48 grid cell aggregation view.")
    return p.parse_args()


def _get_map_size(jsonl_path: Path) -> tuple[float, float]:
    """Read the first obs that has a 'map' field and return (size_x, size_z)."""
    for obs in iter_jsonl(jsonl_path):
        m = obs.get("map")
        if m and m.get("size_x"):
            return float(m["size_x"]), float(m["size_z"])
    return 8192.0, 6144.0  # BAR default fallback


def main() -> int:
    args = parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: JSONL file not found: {args.jsonl}", file=sys.stderr)
        return 1

    output = args.output or args.jsonl.with_suffix(".mp4")
    img_w = args.width
    img_h = int(img_w * 0.75)

    print(f"Loading map size from: {args.jsonl.name}")
    map_w, map_h = _get_map_size(args.jsonl)
    print(f"  Map: {map_w:.0f} x {map_h:.0f} Spring units")
    print(f"  Output: {output}")
    print(f"  Render: {img_w}x{img_h}  fps={args.fps}"
          + ("  fog=on" if args.fog else "")
          + (f"  perspective=team{args.perspective}" if args.fog and args.perspective >= 0 else "")
          + ("  grid=on" if args.grid else ""))
    print()

    renderer = ReplayRenderer(
        map_w, map_h,
        img_w=img_w, img_h=img_h,
        use_fog=args.fog,
        perspective_team=args.perspective,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, float(args.fps), (img_w, img_h))
    if not writer.isOpened():
        print(f"ERROR: could not open VideoWriter for {output}", file=sys.stderr)
        return 1

    n_frames = 0
    try:
        for obs in iter_jsonl(args.jsonl):
            if obs.get("type") == "header":
                continue
            if obs.get("type") != "obs" and "teams" not in obs:
                continue

            frame_img = renderer.render_frame(obs)

            if args.grid:
                draw_grid_overlay(frame_img, obs, map_w, map_h)

            writer.write(frame_img)
            n_frames += 1

            if n_frames % 50 == 0:
                frame_num = obs.get("frame", 0)
                secs = int(frame_num) // 30
                print(f"  rendered {n_frames} frames  (game time {secs//60:02d}:{secs%60:02d})")

    finally:
        writer.release()

    duration_s = n_frames / args.fps
    print()
    print(f"Done. {n_frames} frames -> {duration_s:.1f}s video @ {args.fps}fps")
    print(f"Output: {output}  ({output.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
