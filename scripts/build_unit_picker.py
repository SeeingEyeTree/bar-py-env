#!/usr/bin/env python3
"""Generate a single-file HTML picker over every unit in the BAR repo.

The result is a static HTML page with a checkbox per unit, faction/tier/folder
filters, a search box, and Download/Load Whitelist JSON buttons. Save the page
locally and open it in any browser -- selection persists via localStorage, so
you can tab away and come back.

Usage:
    python scripts/build_unit_picker.py \\
        --bar-root C:/Users/malco/OneDrive/Documents/GitHub/Beyond-All-Reason-project \\
        --out      tools/unit_picker.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Top-level Lua-table field extractor. Targets keys at exactly one tab/2-spaces
# of indent, i.e. the unit's own fields, NOT fields inside weapondefs/customparams.
_FIELD_RE_CACHE: dict[str, re.Pattern] = {}


def field(text: str, key: str) -> str | None:
    """Find a unit-level field (one indent inside `armcode = { ... }`).

    BAR's unit defs are consistently tab-indented, so the unit's own fields sit
    at exactly two tabs of indent (one for the `return {` table, one for the
    `armcode = {` sub-table). Field names that appear deeper (inside
    customparams or weapondefs) are matched at three or more tabs, so this
    regex won't pick them up.
    """
    pat = _FIELD_RE_CACHE.get(key)
    if pat is None:
        pat = re.compile(
            r"(?:^|\n)(?:\t\t|        )" + re.escape(key)
            + r"\s*=\s*([^,\r\n]+)",
            re.IGNORECASE,
        )
        _FIELD_RE_CACHE[key] = pat
    m = pat.search(text)
    if m is None:
        return None
    v = m.group(1).strip().rstrip(",").strip().rstrip("\r")
    return v.strip('"').strip("'")


def num(text: str, key: str) -> float | None:
    v = field(text, key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def boolean(text: str, key: str) -> bool | None:
    v = field(text, key)
    if v is None:
        return None
    return v.lower() == "true"


def faction_of(code: str) -> str:
    c = code.lower()
    if c.startswith("arm"):
        return "Armada"
    if c.startswith("cor"):
        return "Cortex"
    if c.startswith("leg"):
        return "Legion"
    if c.startswith("scav"):
        return "Scavenger"
    if c.startswith("rapt") or c.startswith("chick"):
        return "Raptor"
    if c.startswith("xmas") or c.startswith("ufo"):
        return "Joke"
    return "Other"


def tier_of(rel_path: str) -> str:
    p = rel_path.replace("\\", "/")
    if "/T3" in p or "T3 " in p:
        return "T3"
    if "/T2" in p or "T2 " in p:
        return "T2"
    return "T1"


def category_of(rel_path: str) -> str:
    """Top-level folder, e.g. ArmBots / Legion / other."""
    return rel_path.replace("\\", "/").split("/", 1)[0]


def parse_unit_file(path: Path, rel_path: str) -> dict | None:
    # BAR's unit files are CRLF; normalize so the regexes don't pick up `\r`
    # in captured values (which then break `float()`).
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    m = re.search(r"return\s*\{\s*([a-zA-Z0-9_]+)\s*=", text)
    if not m:
        return None
    code = m.group(1)
    hp = num(text, "health")
    if hp is None:
        hp = num(text, "maxdamage")
    metalcost = num(text, "metalcost")
    if metalcost is None:
        metalcost = num(text, "buildcostmetal")
    energycost = num(text, "energycost")
    if energycost is None:
        energycost = num(text, "buildcostenergy")
    speed = num(text, "speed")
    if speed is None:
        speed = num(text, "maxvelocity")
    workertime = num(text, "workertime")
    return {
        "code": code,
        "path": rel_path,
        "faction": faction_of(code),
        "tier": tier_of(rel_path),
        "category": category_of(rel_path),
        "hp": hp,
        "metal": metalcost,
        "energy": energycost,
        "buildtime": num(text, "buildtime"),
        "speed": speed,
        "canfly": boolean(text, "canfly") or False,
        "canmove": boolean(text, "canmove") or False,
        "builder": boolean(text, "builder") or False,
        "workertime": workertime,
        "weapon": "weapondefs" in text.lower(),
    }


def collect_units(bar_root: Path) -> list[dict]:
    units_dir = bar_root / "units"
    lang_file = bar_root / "language" / "en" / "units.json"
    names: dict[str, str] = {}
    descs: dict[str, str] = {}
    if lang_file.exists():
        try:
            data = json.loads(lang_file.read_text(encoding="utf-8"))
            units = (data or {}).get("units", {})
            names = units.get("names", {}) or {}
            descs = units.get("descriptions", {}) or {}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not read {lang_file}: {e}", file=sys.stderr)

    records: list[dict] = []
    skipped = 0
    for path in sorted(units_dir.rglob("*.lua")):
        rel = path.relative_to(units_dir).as_posix()
        try:
            rec = parse_unit_file(path, rel)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if rec is None:
            skipped += 1
            continue
        rec["name"] = names.get(rec["code"], "")
        rec["desc"] = descs.get(rec["code"], "")
        records.append(rec)
    print(f"  parsed {len(records)} units; skipped {skipped}")
    return records


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BAR Unit Picker</title>
<style>
  :root {
    --bg: #14161a;
    --card: #1c2128;
    --card-on: #234d3a;
    --border: #2f343c;
    --text: #e6e7eb;
    --muted: #9aa1ac;
    --accent: #4dabf7;
    --arm: #4dabf7;
    --cor: #ff6b6b;
    --leg: #51cf66;
    --scav: #b197fc;
    --rapt: #ffa94d;
    --joke: #868e96;
    --other: #adb5bd;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
          Ubuntu, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
  }
  h1 { margin: 0 0 8px; font-size: 18px; font-weight: 600; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center;
  }
  .controls input[type="text"],
  .controls select {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font: inherit;
  }
  .controls input[type="text"] { min-width: 200px; }
  button {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 12px; font: inherit; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary {
    background: var(--accent); color: #14161a;
    border-color: var(--accent); font-weight: 600;
  }
  .stats { color: var(--muted); margin-left: auto; }
  main {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 8px; padding: 16px 20px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; cursor: pointer;
    transition: background 80ms ease, border-color 80ms ease;
    user-select: none;
  }
  .card:hover { border-color: var(--accent); }
  .card.on { background: var(--card-on); border-color: #51cf66; }
  .row1 { display: flex; align-items: baseline; gap: 8px; }
  .name { font-weight: 600; font-size: 14px; flex: 1;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .code { color: var(--muted); font-family: ui-monospace, Menlo, monospace;
          font-size: 11px; }
  .badges { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
  .b {
    display: inline-block; padding: 1px 6px; border-radius: 10px;
    font-size: 10px; line-height: 16px; background: #2a2f37;
    color: var(--muted); border: 1px solid var(--border);
  }
  .b.arm  { color: var(--arm);   border-color: var(--arm); }
  .b.cor  { color: var(--cor);   border-color: var(--cor); }
  .b.leg  { color: var(--leg);   border-color: var(--leg); }
  .b.scav { color: var(--scav);  border-color: var(--scav); }
  .b.rapt { color: var(--rapt);  border-color: var(--rapt); }
  .b.joke { color: var(--joke);  border-color: var(--joke); }
  .stats-row { color: var(--muted); font-size: 12px; }
  .desc { color: var(--muted); font-size: 12px; margin-top: 4px;
          line-height: 1.35; min-height: 1.35em; }
  details { background: var(--card); border: 1px solid var(--border);
            border-radius: 6px; padding: 6px 10px; margin-right: 8px; }
  details > summary { cursor: pointer; color: var(--muted); list-style: none; }
  details[open] > summary { color: var(--text); }
  details .opts { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
  details .opts label { display: inline-flex; align-items: center; gap: 4px;
                        font-size: 12px; color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>BAR Unit Picker  &middot;  pick which units to include in your whitelist</h1>
  <div class="controls">
    <input id="q" type="text" placeholder="Search name, code or description…">
    <select id="faction">
      <option value="">All factions</option>
      <option>Armada</option>
      <option>Cortex</option>
      <option>Legion</option>
      <option>Scavenger</option>
      <option>Raptor</option>
      <option>Joke</option>
      <option>Other</option>
    </select>
    <select id="tier">
      <option value="">All tiers</option>
      <option>T1</option><option>T2</option><option>T3</option>
    </select>
    <details>
      <summary>Folder</summary>
      <div id="folders" class="opts"></div>
    </details>
    <details>
      <summary>Type</summary>
      <div class="opts">
        <label><input type="checkbox" id="weaponOnly"> has weapon</label>
        <label><input type="checkbox" id="builderOnly"> is builder</label>
        <label><input type="checkbox" id="immobileOnly"> immobile</label>
        <label><input type="checkbox" id="moverOnly"> mobile</label>
        <label><input type="checkbox" id="flyerOnly"> flying</label>
      </div>
    </details>
    <button id="selVis">Select visible</button>
    <button id="desVis">Deselect visible</button>
    <button id="invert">Invert visible</button>
    <button id="clear">Clear all</button>
    <button class="primary" id="download">Download whitelist…</button>
    <button id="load">Load whitelist…</button>
    <input id="loadFile" type="file" accept="application/json" hidden>
    <span class="stats" id="stats"></span>
  </div>
</header>
<main id="grid"></main>

<script>
const UNITS = __UNITS_JSON__;
const STORAGE_KEY = "bar_unit_picker_v1";

// Selection state: Set of unit codes.
let selected = new Set();
try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  if (Array.isArray(saved)) selected = new Set(saved);
} catch {}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...selected]));
}

function factionClass(f) {
  return { "Armada": "arm", "Cortex": "cor", "Legion": "leg",
           "Scavenger": "scav", "Raptor": "rapt", "Joke": "joke" }[f] || "";
}

function fmtNum(n) {
  if (n == null) return "";
  if (n >= 1000) return Math.round(n).toLocaleString();
  if (n === Math.round(n)) return String(n);
  return n.toFixed(1);
}

function cardHtml(u) {
  const fc = factionClass(u.faction);
  const stats = [];
  if (u.hp) stats.push(`HP ${fmtNum(u.hp)}`);
  if (u.metal) stats.push(`M ${fmtNum(u.metal)}`);
  if (u.energy) stats.push(`E ${fmtNum(u.energy)}`);
  if (u.buildtime) stats.push(`bt ${fmtNum(u.buildtime)}`);
  if (u.speed) stats.push(`spd ${fmtNum(u.speed)}`);
  if (u.workertime) stats.push(`BP ${fmtNum(u.workertime)}`);
  const role = [];
  if (u.builder)  role.push("builder");
  if (u.weapon)   role.push("armed");
  if (u.canfly)   role.push("air");
  else if (u.canmove) role.push("mobile");
  else role.push("static");
  return `
    <div class="card${selected.has(u.code) ? " on" : ""}" data-code="${u.code}">
      <div class="row1">
        <span class="name">${u.name || u.code}</span>
        <span class="code">${u.code}</span>
      </div>
      <div class="badges">
        <span class="b ${fc}">${u.faction}</span>
        <span class="b">${u.tier}</span>
        <span class="b">${u.category}</span>
        ${role.map(r => `<span class="b">${r}</span>`).join("")}
      </div>
      <div class="stats-row">${stats.join("  ·  ")}</div>
      <div class="desc">${u.desc || ""}</div>
    </div>`;
}

// Filter state
const q = document.getElementById("q");
const fSel = document.getElementById("faction");
const tSel = document.getElementById("tier");
const wOnly = document.getElementById("weaponOnly");
const bOnly = document.getElementById("builderOnly");
const iOnly = document.getElementById("immobileOnly");
const mOnly = document.getElementById("moverOnly");
const flyOnly = document.getElementById("flyerOnly");
const grid = document.getElementById("grid");
const stats = document.getElementById("stats");
const foldersDiv = document.getElementById("folders");

// Build the folder filter dynamically.
const folders = [...new Set(UNITS.map(u => u.category))].sort();
const activeFolders = new Set(folders);   // start: all enabled
foldersDiv.innerHTML = folders.map(f => `
  <label><input type="checkbox" data-folder="${f}" checked> ${f}</label>
`).join("");

function applyFilters() {
  const ql = q.value.trim().toLowerCase();
  const f = fSel.value;
  const t = tSel.value;
  const wantWeapon = wOnly.checked;
  const wantBuilder = bOnly.checked;
  const wantImmobile = iOnly.checked;
  const wantMover = mOnly.checked;
  const wantFly = flyOnly.checked;
  const filtered = UNITS.filter(u => {
    if (f && u.faction !== f) return false;
    if (t && u.tier !== t) return false;
    if (!activeFolders.has(u.category)) return false;
    if (wantWeapon && !u.weapon) return false;
    if (wantBuilder && !u.builder) return false;
    if (wantImmobile && u.canmove) return false;
    if (wantMover && !u.canmove) return false;
    if (wantFly && !u.canfly) return false;
    if (ql) {
      const hay = ((u.name||"") + " " + u.code + " " + (u.desc||"")).toLowerCase();
      if (!hay.includes(ql)) return false;
    }
    return true;
  });
  grid.innerHTML = filtered.map(cardHtml).join("");
  refreshStats(filtered);
  return filtered;
}

function refreshStats(filteredSubset) {
  const total = UNITS.length;
  const visible = filteredSubset ? filteredSubset.length :
    grid.querySelectorAll(".card").length;
  stats.textContent = `${selected.size} selected  ·  ${visible} visible  ·  ${total} total`;
}

grid.addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  const code = card.dataset.code;
  if (selected.has(code)) { selected.delete(code); card.classList.remove("on"); }
  else                    { selected.add(code);    card.classList.add("on"); }
  persist(); refreshStats();
});

[q, fSel, tSel, wOnly, bOnly, iOnly, mOnly, flyOnly].forEach(el => {
  el.addEventListener("input", applyFilters);
});
foldersDiv.addEventListener("change", (e) => {
  const cb = e.target;
  if (cb && cb.dataset && cb.dataset.folder) {
    if (cb.checked) activeFolders.add(cb.dataset.folder);
    else activeFolders.delete(cb.dataset.folder);
    applyFilters();
  }
});

document.getElementById("selVis").onclick = () => {
  for (const c of grid.querySelectorAll(".card")) {
    selected.add(c.dataset.code); c.classList.add("on");
  }
  persist(); refreshStats();
};
document.getElementById("desVis").onclick = () => {
  for (const c of grid.querySelectorAll(".card")) {
    selected.delete(c.dataset.code); c.classList.remove("on");
  }
  persist(); refreshStats();
};
document.getElementById("invert").onclick = () => {
  for (const c of grid.querySelectorAll(".card")) {
    const code = c.dataset.code;
    if (selected.has(code)) { selected.delete(code); c.classList.remove("on"); }
    else                    { selected.add(code);    c.classList.add("on"); }
  }
  persist(); refreshStats();
};
document.getElementById("clear").onclick = () => {
  if (!confirm(`Clear all ${selected.size} selections?`)) return;
  selected.clear(); persist();
  for (const c of grid.querySelectorAll(".card")) c.classList.remove("on");
  refreshStats();
};
document.getElementById("download").onclick = () => {
  const payload = {
    version: 1,
    generated: new Date().toISOString(),
    count: selected.size,
    codes: [...selected].sort(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                       { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "bar_unit_whitelist.json";
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(a.href);
};
document.getElementById("load").onclick = () =>
  document.getElementById("loadFile").click();
document.getElementById("loadFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const txt = await file.text();
    const data = JSON.parse(txt);
    const codes = Array.isArray(data) ? data : (data.codes || []);
    selected = new Set(codes);
    persist(); applyFilters();
    alert(`Loaded ${selected.size} selected units.`);
  } catch (err) {
    alert("Couldn't read that file: " + err.message);
  }
});

applyFilters();
</script>
</body>
</html>
"""


def render_html(records: list[dict]) -> str:
    payload = json.dumps(records, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__UNITS_JSON__", payload)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bar-root", type=Path,
        default=Path("C:/Users/malco/OneDrive/Documents/GitHub/Beyond-All-Reason-project"),
        help="Path to the Beyond-All-Reason repo clone.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "tools" / "unit_picker.html",
        help="Where to write the HTML page.",
    )
    args = p.parse_args()

    if not (args.bar_root / "units").is_dir():
        sys.stderr.write(
            f"--bar-root does not look like a BAR clone "
            f"(no `units/` folder at {args.bar_root}).\n"
        )
        return 1
    print(f"Scanning units under: {args.bar_root}")
    records = collect_units(args.bar_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(records), encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out}  ({size_kb:,.0f} KB)")
    print("Open it in any browser to pick units, then click "
          "'Download whitelist…' to save your selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
