#!/usr/bin/env python3
"""Generate bar_env/data/unitdef_names.json and bar_env/data/unitdef_costs.json
from the BAR source tree.

Spring/Recoil assigns unitDefIDs by sorting ALL unit def names alphabetically
(case-insensitive) and numbering them 1..N. This script replicates that by
scanning the units/ folder of the BAR project.

Outputs:
    unitdef_names.json  -- { "armflash": 47, "cormex": 312, ... }
    unitdef_costs.json  -- { "armflash": {"metal": 110, "energy": 900, "equiv": 125.0}, ... }

Metal-equivalent cost = metalcost + energycost / 60
(BAR's approximate metal:energy conversion at an efficient economy is ~1:60.)

Usage:
    python scripts/gen_unitdef_map.py
    python scripts/gen_unitdef_map.py --bar-src "C:/path/to/Beyond-All-Reason-project"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_BAR_SRC = Path(
    r"C:\Users\malco\OneDrive\Documents\GitHub\Beyond-All-Reason-project"
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "bar_env" / "data"

EXCLUDE_STEMS = {"buildinggrounddecal_processor"}

ENERGY_TO_METAL_RATIO = 60.0  # 60 energy = 1 metal (approximate BAR conversion)


def collect_defs(bar_src: Path) -> tuple[list[str], dict[str, dict]]:
    """Return (sorted_names, costs_by_name).

    costs_by_name: { name: {"metal": int, "energy": int, "equiv": float} }
    """
    units_dir = bar_src / "units"
    if not units_dir.is_dir():
        raise FileNotFoundError(f"units/ folder not found under {bar_src}")

    names: list[str] = []
    costs: dict[str, dict] = {}

    for lua in units_dir.rglob("*.lua"):
        stem = lua.stem.lower()
        if stem in EXCLUDE_STEMS or stem.startswith("buildingground"):
            continue
        names.append(stem)

        try:
            text = lua.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        m_m = re.search(r"\bmetalcost\s*=\s*([0-9]+)", text)
        m_e = re.search(r"\benergycost\s*=\s*([0-9]+)", text)
        metal  = int(m_m.group(1)) if m_m else 0
        energy = int(m_e.group(1)) if m_e else 0
        if metal > 0 or energy > 0:
            costs[stem] = {
                "metal":  metal,
                "energy": energy,
                "equiv":  round(metal + energy / ENERGY_TO_METAL_RATIO, 1),
            }

    names.sort()
    return names, costs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bar-src", type=Path, default=DEFAULT_BAR_SRC,
                   help=f"Path to BAR source tree. Default: {DEFAULT_BAR_SRC}")
    args = p.parse_args()

    print(f"Scanning: {args.bar_src / 'units'}")
    names, costs = collect_defs(args.bar_src)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write names -> def_id mapping
    names_path = DATA_DIR / "unitdef_names.json"
    mapping = {name: idx + 1 for idx, name in enumerate(names)}
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Written {len(mapping)} entries -> {names_path}")

    # Write costs mapping (keyed by def_id for fast lookup at runtime)
    costs_by_id = {mapping[name]: costs[name] for name in costs if name in mapping}
    costs_path = DATA_DIR / "unitdef_costs.json"
    with open(costs_path, "w", encoding="utf-8") as f:
        json.dump(costs_by_id, f, indent=2, sort_keys=True)
    print(f"Written {len(costs_by_id)} unit costs -> {costs_path}")

    # Sanity check
    print()
    print(f"  {'unit':<12} {'def_id':>7}  {'metal':>6}  {'energy':>7}  {'equiv':>7}")
    for name in ("armflash", "corraid", "armcom", "corcom", "armvp", "corvp",
                 "armmex", "cormex"):
        did = mapping.get(name, "?")
        c = costs.get(name, {})
        print(f"  {name:<12} {str(did):>7}  {c.get('metal', 0):>6}  "
              f"{c.get('energy', 0):>7}  {c.get('equiv', 0):>7.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
