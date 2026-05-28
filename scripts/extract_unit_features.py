#!/usr/bin/env python3
"""Extract a numeric feature row per unit, ready for unsupervised clustering.

Reads:
  bar_env/units/real_bar_units.json   - whitelist of unit codes to keep
  <bar_root>/units/**/*.lua           - the unit defs themselves

Writes:
  bar_env/units/unit_features.csv     - one row per unit, columns:
    code, name, faction, folder, tier, unitgroup,
    buildtime, metalcost, energycost, health,
    speed, maxacc, maxdec, maxslope, maxwaterdepth,
    sightdistance, turnrate, turninplace, canmove, techlevel,
    n_weapons, best_dps, total_dps, max_range, min_range,
    max_aoe, has_aa, has_anti_sub, has_paralyzer,
    wt_cannon, wt_beamlaser, wt_lasercannon, wt_missilelauncher,
    wt_starburstlauncher, wt_torpedolauncher, wt_emgcannon,
    wt_aircraftbomb, wt_flame, wt_dgun, wt_shield, wt_other

Lua execution: uses `lupa` to actually run the unit's `return {...}` table.
Install: pip install lupa  (already pinned in pyproject.toml).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import lupa


# Spring/Recoil weapon types we one-hot. Anything else → wt_other.
WEAPON_TYPES = [
    "cannon", "beamlaser", "lasercannon", "missilelauncher",
    "starburstlauncher", "torpedolauncher", "emgcannon",
    "aircraftbomb", "flame", "dgun", "shield",
]


def lua_to_py(v):
    """Convert a lupa-returned value (Lua table / scalar) to a Python value."""
    if v is None:
        return None
    t = lupa.lua_type(v)
    if t == "table":
        d = {}
        # Iterate the table's pairs. Lua keys can be ints, strings, or arbitrary.
        for k, vv in v.items():
            d[k] = lua_to_py(vv)
        # If keys are 1..n (Lua array), return a list.
        if d and all(isinstance(k, int) for k in d):
            keys = sorted(d)
            if keys == list(range(1, len(keys) + 1)):
                return [d[k] for k in keys]
        return d
    return v


def parse_unit(path: Path) -> tuple[str, dict] | None:
    """Run the unit's return-statement and return (code, defs_dict)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    try:
        tbl = lua.execute(text)
    except Exception:  # noqa: BLE001
        return None
    if tbl is None:
        return None
    py = lua_to_py(tbl)
    if not isinstance(py, dict) or len(py) != 1:
        return None
    code, defs = next(iter(py.items()))
    if not isinstance(defs, dict):
        return None
    return code, defs


def faction_of(code: str) -> str:
    c = code.lower()
    if c.startswith("arm"):
        return "Armada"
    if c.startswith("cor"):
        return "Cortex"
    if c.startswith("leg"):
        return "Legion"
    return "Other"


def folder_of(rel_path: str) -> str:
    return rel_path.replace("\\", "/").split("/", 1)[0]


def tier_of(rel_path: str, cp_techlevel) -> int:
    """Prefer customparams.techlevel; fall back to folder hint."""
    if cp_techlevel is not None:
        try:
            return int(float(cp_techlevel))
        except (TypeError, ValueError):
            pass
    p = rel_path.replace("\\", "/").lower()
    if "/t3" in p or "experimental" in p:
        return 3
    if "/t2" in p or "advanced" in p:
        return 2
    return 1


def best_speed(defs: dict) -> float | None:
    """Spring uses different field names depending on unit kind."""
    for k in ("speed", "maxvelocity", "cruisespeed"):
        v = defs.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def num(defs: dict, key: str) -> float | None:
    v = defs.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def boolean(defs: dict, key: str) -> bool:
    v = defs.get(key)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


def max_damage_default(damage_tbl) -> float | None:
    """Pick the largest 'damage.<category>' value (or just default if present).

    Spring lets weapons deal different damage to different unit categories, so
    a weapon's intrinsic punch is best summarised as 'max across categories'.
    """
    if not isinstance(damage_tbl, dict):
        return None
    best = None
    for v in damage_tbl.values():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if best is None or x > best:
            best = x
    return best


def is_aa_weapon(weapondef: dict, mount: dict | None) -> bool:
    """Heuristic: a weapon is AA if it can only target air, OR has flak hints."""
    # The mount's onlytargetcategory wins -- it's how Spring actually filters.
    if isinstance(mount, dict):
        only = (mount.get("onlytargetcategory") or "").upper()
        if "VTOL" in only and "GROUND" not in only and "SURFACE" not in only:
            return True
    # Some weapons declare it on the def too.
    wt = (weapondef.get("weapontargetcategory") or "").upper()
    if "VTOL" in wt and "GROUND" not in wt:
        return True
    # Customparams hint
    cp = weapondef.get("customparams") or {}
    if isinstance(cp, dict) and (cp.get("istargetableonlybyair") or cp.get("targetair")):
        return True
    return False


def is_anti_sub_weapon(weapondef: dict, mount: dict | None) -> bool:
    wtype = (weapondef.get("weapontype") or "").lower()
    if "torpedo" in wtype:
        return True
    if weapondef.get("waterweapon"):
        return True
    if isinstance(mount, dict):
        only = (mount.get("onlytargetcategory") or "").upper()
        if "UNDERWATER" in only:
            return True
    return False


def is_paralyzer(weapondef: dict) -> bool:
    if weapondef.get("paralyzer"):
        return True
    cp = weapondef.get("customparams") or {}
    if isinstance(cp, dict) and (cp.get("damagetype") or "").lower() in {"emp", "paralyzer", "paralyze"}:
        return True
    if "emp" in (weapondef.get("name") or "").lower():
        return True
    return False


def weapon_features(defs: dict) -> dict:
    """Aggregate every weapon mount to fixed-size numeric features."""
    weapondefs = defs.get("weapondefs") or {}
    mounts = defs.get("weapons") or []
    if not isinstance(weapondefs, dict):
        weapondefs = {}
    # Normalize mounts to a list of dicts.
    if isinstance(mounts, dict):
        mounts = [v for _, v in sorted(mounts.items())]
    if not isinstance(mounts, list):
        mounts = []

    # Index weapondefs by lower-cased name so we can resolve mount.def lookup.
    def_by_lower = {str(k).lower(): v for k, v in weapondefs.items()
                    if isinstance(v, dict)}

    # Build the list of (weapondef, mount) pairs the engine actually uses.
    # When a unit has weapondefs but no `weapons` block, the engine still uses
    # all defined weapons -- so we fall back to all weapondefs.
    pairs: list[tuple[dict, dict | None]] = []
    if mounts:
        for m in mounts:
            if not isinstance(m, dict):
                continue
            dname = str(m.get("def") or "").lower()
            wdef = def_by_lower.get(dname)
            if wdef is None and def_by_lower:
                # Some files don't reference defs by name; if only one weapondef
                # exists, that's it.
                if len(def_by_lower) == 1:
                    wdef = next(iter(def_by_lower.values()))
            if wdef is not None:
                pairs.append((wdef, m))
    if not pairs and def_by_lower:
        pairs = [(w, None) for w in def_by_lower.values()]

    n_weapons = len(pairs)
    if n_weapons == 0:
        out = {
            "n_weapons": 0,
            "best_dps": 0.0, "total_dps": 0.0,
            "max_range": 0.0, "min_range": 0.0,
            "max_aoe": 0.0,
            "has_aa": 0, "has_anti_sub": 0, "has_paralyzer": 0,
        }
        for wt in WEAPON_TYPES:
            out[f"wt_{wt}"] = 0
        out["wt_other"] = 0
        return out

    best_dps = 0.0
    total_dps = 0.0
    ranges, aoes = [], []
    has_aa = has_anti_sub = has_paralyzer = 0
    type_counts = {wt: 0 for wt in WEAPON_TYPES}
    type_counts["other"] = 0
    for wdef, mount in pairs:
        dmg = max_damage_default(wdef.get("damage"))
        rt = wdef.get("reloadtime")
        try:
            rt_f = float(rt) if rt is not None else 1.0
        except (TypeError, ValueError):
            rt_f = 1.0
        if rt_f <= 0:
            rt_f = 1.0
        if dmg is not None:
            dps = dmg / rt_f
            # D-gun's `disintegrator` is 99999 damage and would dominate every
            # downstream feature scaler. Cap at 60000, which keeps T2+ heavy
            # arty (10k+) and crawling bombs (~30k) in the picture but drops
            # the commander's instakill.
            if dmg < 60000:
                total_dps += dps
                if dps > best_dps:
                    best_dps = dps
        r = wdef.get("range")
        try:
            ranges.append(float(r))
        except (TypeError, ValueError):
            pass
        a = wdef.get("areaofeffect")
        try:
            aoes.append(float(a))
        except (TypeError, ValueError):
            pass
        if is_aa_weapon(wdef, mount):
            has_aa = 1
        if is_anti_sub_weapon(wdef, mount):
            has_anti_sub = 1
        if is_paralyzer(wdef):
            has_paralyzer = 1
        wt = (wdef.get("weapontype") or "").lower()
        if wt in type_counts:
            type_counts[wt] += 1
        else:
            type_counts["other"] += 1

    out = {
        "n_weapons": n_weapons,
        "best_dps": round(best_dps, 2),
        "total_dps": round(total_dps, 2),
        "max_range": max(ranges) if ranges else 0.0,
        "min_range": min(ranges) if ranges else 0.0,
        "max_aoe": max(aoes) if aoes else 0.0,
        "has_aa": has_aa, "has_anti_sub": has_anti_sub,
        "has_paralyzer": has_paralyzer,
    }
    for wt in WEAPON_TYPES:
        out[f"wt_{wt}"] = type_counts[wt]
    out["wt_other"] = type_counts["other"]
    return out


def base_features(defs: dict, rel_path: str, name: str) -> dict:
    cp = defs.get("customparams") or {}
    if not isinstance(cp, dict):
        cp = {}
    return {
        "name": name,
        "folder": folder_of(rel_path),
        "unitgroup": cp.get("unitgroup", ""),
        "buildtime": num(defs, "buildtime"),
        "metalcost": num(defs, "metalcost") or num(defs, "buildcostmetal"),
        "energycost": num(defs, "energycost") or num(defs, "buildcostenergy"),
        "health": num(defs, "health") or num(defs, "maxdamage"),
        "speed": best_speed(defs) or 0.0,
        "maxacc": num(defs, "maxacc") or 0.0,
        "maxdec": num(defs, "maxdec") or 0.0,
        "maxslope": num(defs, "maxslope") or 0.0,
        "maxwaterdepth": num(defs, "maxwaterdepth") or 0.0,
        "sightdistance": num(defs, "sightdistance") or 0.0,
        "turnrate": num(defs, "turnrate") or 0.0,
        "turninplace": 1 if boolean(defs, "turninplace") else 0,
        "canmove": 1 if boolean(defs, "canmove") else 0,
        "techlevel": tier_of(rel_path, cp.get("techlevel")),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect(bar_root: Path, whitelist: set[str], names: dict[str, str]) -> list[dict]:
    units_dir = bar_root / "units"
    by_code: dict[str, dict] = {}
    skipped = 0
    failed = 0
    for path in units_dir.rglob("*.lua"):
        rec = parse_unit(path)
        if rec is None:
            failed += 1
            continue
        code, defs = rec
        if code not in whitelist:
            skipped += 1
            continue
        rel = path.relative_to(units_dir).as_posix()
        row = {"code": code, "faction": faction_of(code)}
        row.update(base_features(defs, rel, names.get(code, "")))
        row.update(weapon_features(defs))
        by_code[code] = row  # last-wins; whitelisted units only appear once anyway
    missing = sorted(whitelist - by_code.keys())
    print(f"  wrote {len(by_code)} units; {failed} parse failures; "
          f"{skipped} parsed but not in whitelist; "
          f"{len(missing)} whitelist entries with no lua file found")
    if missing[:5]:
        print(f"  missing first 5: {missing[:5]}")
    return sorted(by_code.values(), key=lambda r: (r["faction"], r["folder"], r["code"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bar-root", type=Path,
        default=Path("C:/Users/malco/OneDrive/Documents/GitHub/Beyond-All-Reason-project"),
    )
    p.add_argument(
        "--whitelist", type=Path,
        default=Path(__file__).resolve().parent.parent / "bar_env" / "units" / "real_bar_units.json",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "bar_env" / "units" / "unit_features.csv",
    )
    args = p.parse_args()

    if not (args.bar_root / "units").is_dir():
        sys.stderr.write(f"--bar-root has no units/ subfolder: {args.bar_root}\n")
        return 1
    if not args.whitelist.exists():
        sys.stderr.write(f"--whitelist not found: {args.whitelist}\n")
        return 1

    data = json.loads(args.whitelist.read_text(encoding="utf-8"))
    codes = set(data.get("codes") or [])
    print(f"Whitelist size: {len(codes)}")

    names: dict[str, str] = {}
    lang = args.bar_root / "language" / "en" / "units.json"
    if lang.exists():
        try:
            names = ((json.loads(lang.read_text(encoding="utf-8")) or {})
                     .get("units", {}).get("names", {}) or {})
        except Exception:  # noqa: BLE001
            pass

    rows = collect(args.bar_root, codes, names)
    if not rows:
        sys.stderr.write("No rows extracted.\n"); return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.out}  ({len(rows)} rows, {len(fieldnames)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
