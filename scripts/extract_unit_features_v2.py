"""V2 feature extractor: adds movementclass + derived features.

New columns compared to v1:
  movementclass (raw string)
  mc_BOT, mc_ABOT, mc_TANK, mc_ATANK, mc_HOVER (family one-hots, slightly weighted in clustering)
  weapon_velocity                (max across weapons)
  damage_vs_air                  (max damage.vtol / damage.default ratio across weapons)
  trajectory_high                (any weapon highTrajectory or StarburstLauncher)
  range_per_speed                (max_range / max(speed, 1))
  dps_per_metal                  (best_dps / max(metalcost, 1))
  health_per_metal               (health / max(metalcost, 1))
  range_tier                     (0=short<400, 1=mid 400-900, 2=long 900-2000, 3=siege>2000)
  weapon_velocity_low            (1 if weapon_velocity > 0 and < 600 — arcing/ballistic)

Dropped from clustering (but kept in CSV for tooltips):
  turnrate
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import lupa

WEAPON_TYPES = ["cannon","beamlaser","lasercannon","missilelauncher",
    "starburstlauncher","torpedolauncher","emgcannon",
    "aircraftbomb","flame","dgun","shield"]

# Family grouping for movementclass. Anything not matched → "OTHER".
MC_FAMILY = {
    # bots (regular kbots)
    **{n: "BOT" for n in ["BOT2","BOT3","HBOT4","HBOT7","HTBOT6","TBOT3","VBOT6","SBOT2","KBOT2","KBOT3","KBOT4"]},
    # amphibious bots / suicide amphibs
    **{n: "ABOT" for n in ["ABOT3","AKBOT3","ABOTBOMB2","HABOT5","AKBOT2"]},
    # tanks (all sizes)
    **{n: "TANK" for n in ["TANK2","TANK3","MTANK3","HTANK4","HTANK7","VTANK4","VTANK6","VTANK7"]},
    # amphibious tanks
    **{n: "ATANK" for n in ["ATANK3","ATANK4"]},
    # hover (treat amphibious-hover the same; hovers already cross water by nature)
    **{n: "HOVER" for n in ["HOVER2","HOVER3","HHOVER4","AHOVER2","HOVER4"]},
    # commanders / specials — usually filtered out, but bucket them
    "COMMANDERBOT": "COMMANDER",
}
MC_FAMILIES = ["BOT","ABOT","TANK","ATANK","HOVER","COMMANDER","OTHER"]


def lua_to_py(v):
    if v is None: return None
    if lupa.lua_type(v) == "table":
        d = {}
        for k, vv in v.items(): d[k] = lua_to_py(vv)
        if d and all(isinstance(k, int) for k in d):
            ks = sorted(d)
            if ks == list(range(1, len(ks)+1)): return [d[k] for k in ks]
        return d
    return v


def parse_unit(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    try: tbl = lupa.LuaRuntime(unpack_returned_tuples=True).execute(text)
    except Exception: return None
    if tbl is None: return None
    py = lua_to_py(tbl)
    if not isinstance(py, dict) or len(py) != 1: return None
    code, defs = next(iter(py.items()))
    return (code, defs) if isinstance(defs, dict) else None


def faction_of(code):
    c = code.lower()
    if c.startswith("arm"): return "Armada"
    if c.startswith("cor"): return "Cortex"
    return "Other"

def folder_of(rel): return rel.replace("\\","/").split("/",1)[0]

def tier_of(rel, cp_tech):
    if cp_tech is not None:
        try: return int(float(cp_tech))
        except (TypeError, ValueError): pass
    p = rel.replace("\\","/").lower()
    if "/t3" in p or "experimental" in p: return 3
    if "/t2" in p or "advanced" in p: return 2
    return 1

def num(d, k):
    v = d.get(k)
    if v is None: return None
    try: return float(v)
    except (TypeError, ValueError): return None

def best_speed(d):
    for k in ("speed","maxvelocity","cruisespeed"):
        v = num(d, k)
        if v is not None: return v
    return None

def boolean(d, k):
    v = d.get(k)
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in {"1","true","yes"}
    return False

def damage_max_default(dmg):
    if not isinstance(dmg, dict): return None
    best = None
    for k, v in dmg.items():
        if str(k).lower() == "vtol": continue   # tracked separately
        try: x = float(v)
        except (TypeError, ValueError): continue
        if best is None or x > best: best = x
    return best

def is_aa(wdef, mount):
    if isinstance(mount, dict):
        only = str(mount.get("onlytargetcategory") or "").upper()
        if "VTOL" in only and "GROUND" not in only and "SURFACE" not in only: return True
    wt = str(wdef.get("weapontargetcategory") or "").upper()
    if "VTOL" in wt and "GROUND" not in wt: return True
    cp = wdef.get("customparams") or {}
    if isinstance(cp, dict) and (cp.get("istargetableonlybyair") or cp.get("targetair")): return True
    return False

def is_anti_sub(wdef, mount):
    if "torpedo" in str(wdef.get("weapontype") or "").lower(): return True
    if wdef.get("waterweapon"): return True
    if isinstance(mount, dict) and "UNDERWATER" in str(mount.get("onlytargetcategory") or "").upper(): return True
    return False

def is_para(wdef):
    if wdef.get("paralyzer"): return True
    cp = wdef.get("customparams") or {}
    if isinstance(cp, dict) and str(cp.get("damagetype") or "").lower() in {"emp","paralyzer","paralyze"}: return True
    if "emp" in str(wdef.get("name") or "").lower(): return True
    return False

def weapon_features(defs):
    wd = defs.get("weapondefs") or {}
    mounts = defs.get("weapons") or []
    if not isinstance(wd, dict): wd = {}
    if isinstance(mounts, dict): mounts = [v for _, v in sorted(mounts.items())]
    if not isinstance(mounts, list): mounts = []
    by_lower = {str(k).lower(): v for k, v in wd.items() if isinstance(v, dict)}

    pairs = []
    if mounts:
        for m in mounts:
            if not isinstance(m, dict): continue
            dname = str(m.get("def") or "").lower()
            wdef = by_lower.get(dname) or (next(iter(by_lower.values())) if len(by_lower)==1 else None)
            if wdef is not None: pairs.append((wdef, m))
    if not pairs and by_lower:
        pairs = [(w, None) for w in by_lower.values()]

    out = {"n_weapons": len(pairs), "best_dps": 0.0, "total_dps": 0.0,
           "max_range": 0.0, "min_range": 0.0, "max_aoe": 0.0,
           "has_aa": 0, "has_anti_sub": 0, "has_paralyzer": 0,
           "weapon_velocity": 0.0, "damage_vs_air": 0.0, "trajectory_high": 0}
    for wt in WEAPON_TYPES: out[f"wt_{wt}"] = 0
    out["wt_other"] = 0
    if not pairs: return out

    bd, td = 0.0, 0.0
    aa_dps_total = 0.0
    ranges, aoes, vels = [], [], []
    for wdef, mount in pairs:
        dmg = wdef.get("damage")
        default_dmg = damage_max_default(dmg)
        try: rt = float(wdef.get("reloadtime") or 1.0)
        except (TypeError, ValueError): rt = 1.0
        if rt <= 0: rt = 1.0
        if default_dmg is not None and default_dmg < 60000:
            dps = default_dmg / rt
            td += dps
            if dps > bd: bd = dps
        # vtol-damage ratio
        vtol = None
        if isinstance(dmg, dict):
            try: vtol = float(dmg.get("vtol")) if dmg.get("vtol") is not None else None
            except (TypeError, ValueError): vtol = None
        # Per-weapon AA contribution -- summed below into aa_dps.
        wpn_default = default_dmg if (default_dmg and default_dmg < 60000) else 0.0
        wpn_vtol = vtol if (vtol and vtol > 0) else 0.0
        # Decide what this weapon contributes to total-DPS and aa-DPS.
        if wpn_default > 0 and wpn_vtol == 0:
            # ground-only weapon
            pass  # already counted toward total in bd/td
        elif wpn_vtol > 0 and wpn_default == 0:
            # pure AA weapon -- count its DPS toward aa_dps too
            aa_dps_total += wpn_vtol / rt
            td += wpn_vtol / rt   # also count it as total damage output
            if wpn_vtol / rt > bd: bd = wpn_vtol / rt
        elif wpn_vtol > 0 and wpn_default > 0:
            # dual-purpose -- count *vtol* damage as the AA portion
            aa_dps_total += wpn_vtol / rt
        try: ranges.append(float(wdef.get("range")))
        except (TypeError, ValueError): pass
        try: aoes.append(float(wdef.get("areaofeffect")))
        except (TypeError, ValueError): pass
        try:
            v = float(wdef.get("weaponvelocity") or 0)
            if v > 0: vels.append(v)
        except (TypeError, ValueError): pass
        if is_aa(wdef, mount): out["has_aa"] = 1
        if is_anti_sub(wdef, mount): out["has_anti_sub"] = 1
        if is_para(wdef): out["has_paralyzer"] = 1
        wt = str(wdef.get("weapontype") or "").lower()
        if wt in WEAPON_TYPES: out[f"wt_{wt}"] += 1
        else: out["wt_other"] += 1
        # trajectory: BAR uses multiple field names for "arcing shot"
        if wdef.get("highTrajectory") or wdef.get("hightrajectory"):
            out["trajectory_high"] = 1
        try:
            if float(wdef.get("trajectoryheight") or 0) > 0:
                out["trajectory_high"] = 1
        except (TypeError, ValueError): pass
        if "starburst" in wt:
            out["trajectory_high"] = 1

    out["best_dps"] = round(bd, 2); out["total_dps"] = round(td, 2)
    out["max_range"] = max(ranges) if ranges else 0.0
    out["min_range"] = min(ranges) if ranges else 0.0
    out["max_aoe"] = max(aoes) if aoes else 0.0
    out["weapon_velocity"] = max(vels) if vels else 0.0
    out["damage_vs_air"] = round(aa_dps_total / max(td, 1.0), 3)  # fraction of DPS dedicated to AA
    return out

def base_features(defs, rel, name):
    cp = defs.get("customparams") or {}
    if not isinstance(cp, dict): cp = {}
    mc = (defs.get("movementclass") or "").upper()
    family = MC_FAMILY.get(mc, "OTHER")
    base = {"name": name, "folder": folder_of(rel),
        "unitgroup": cp.get("unitgroup",""),
        "movementclass": mc,
        "buildtime": num(defs, "buildtime") or 0,
        "metalcost": num(defs, "metalcost") or num(defs, "buildcostmetal") or 0,
        "energycost": num(defs, "energycost") or num(defs, "buildcostenergy") or 0,
        "health": num(defs, "health") or num(defs, "maxdamage") or 0,
        "speed": best_speed(defs) or 0,
        "maxacc": num(defs, "maxacc") or 0, "maxdec": num(defs, "maxdec") or 0,
        "maxslope": num(defs, "maxslope") or 0,
        "maxwaterdepth": num(defs, "maxwaterdepth") or 0,
        "sightdistance": num(defs, "sightdistance") or 0,
        "turnrate": num(defs, "turnrate") or 0,
        "turninplace": 1 if boolean(defs, "turninplace") else 0,
        "canmove": 1 if boolean(defs, "canmove") else 0,
        "techlevel": tier_of(rel, cp.get("techlevel"))}
    # one-hot movement family
    for fam in MC_FAMILIES:
        base[f"mc_{fam}"] = 1 if family == fam else 0
    return base

def derived_features(row):
    """Compute features that depend on already-extracted base+weapon stats."""
    speed = max(row.get("speed", 0) or 0, 1)
    metal = max(row.get("metalcost", 0) or 0, 1)
    range_ = row.get("max_range", 0) or 0
    dps = row.get("best_dps", 0) or 0
    hp = row.get("health", 0) or 0
    wvel = row.get("weapon_velocity", 0) or 0
    # ratios
    row["range_per_speed"] = round(range_ / speed, 3)
    row["dps_per_metal"]   = round(dps / metal, 4)
    row["health_per_metal"] = round(hp / metal, 3)
    # range tier (ordinal)
    if range_ < 400:        row["range_tier"] = 0
    elif range_ < 900:      row["range_tier"] = 1
    elif range_ < 2000:     row["range_tier"] = 2
    else:                   row["range_tier"] = 3
    # weapon velocity low (arcing/ballistic) -- 0 if no weapon
    row["weapon_velocity_low"] = 1 if (0 < wvel < 600) else 0
    return row

def main():
    root = Path("/sessions/relaxed-nice-maxwell/mnt/Beyond-All-Reason-project")
    whitelist = set(json.loads(Path("/sessions/relaxed-nice-maxwell/mnt/bar-py-env/bar_env/units/real_bar_units.json").read_text())["codes"])
    names_path = root / "language" / "en" / "units.json"
    names = json.loads(names_path.read_text(encoding="utf-8"))["units"]["names"] if names_path.exists() else {}

    by_code = {}
    for path in (root/"units").rglob("*.lua"):
        rec = parse_unit(path)
        if rec is None: continue
        code, defs = rec
        if code not in whitelist: continue
        rel = path.relative_to(root/"units").as_posix()
        row = {"code": code, "faction": faction_of(code)}
        row.update(base_features(defs, rel, names.get(code, "")))
        row.update(weapon_features(defs))
        derived_features(row)
        by_code[code] = row

    rows = sorted(by_code.values(), key=lambda r: (r["faction"], r["folder"], r["code"]))
    out = Path("/sessions/relaxed-nice-maxwell/mnt/outputs/unit_features_v2.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows, {len(rows[0])} cols)")
    return rows

if __name__ == "__main__": main()
