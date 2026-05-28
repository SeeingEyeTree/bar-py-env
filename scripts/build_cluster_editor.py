"""V4 cluster viewer: editable. Drag units between clusters, rename clusters,
add/remove clusters, save/load curated state as JSON."""
import csv, json, math
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

LOG_COLS = ["buildtime","metalcost","energycost","health","best_dps","total_dps",
            "max_range","max_aoe","sightdistance","weapon_velocity",
            "range_per_speed","dps_per_metal","health_per_metal"]
LINEAR_COLS = ["speed","maxacc","maxdec","maxslope","maxwaterdepth",
               "min_range","n_weapons","techlevel","damage_vs_air","range_tier"]
FLAG_COLS = ["canmove","turninplace","has_aa","has_anti_sub","has_paralyzer",
             "trajectory_high","weapon_velocity_low"]
WT_COLS = ["wt_cannon","wt_beamlaser","wt_lasercannon","wt_missilelauncher",
           "wt_emgcannon","wt_flame","wt_other"]
MC_COLS = ["mc_BOT","mc_ABOT","mc_TANK","mc_ATANK","mc_HOVER","mc_COMMANDER","mc_OTHER"]
WEIGHTS = {
    "max_range":2.0,"best_dps":2.0,"has_paralyzer":2.0,
    "range_per_speed":2.0,"dps_per_metal":2.0,"damage_vs_air":2.0,
    "trajectory_high":2.0,"range_tier":3.0,"weapon_velocity_low":2.0,
    "has_aa":1.0,"speed":1.5,
    "mc_BOT":1.5,"mc_ABOT":1.5,"mc_TANK":1.5,"mc_ATANK":1.5,
    "mc_HOVER":1.5,"mc_COMMANDER":1.5,"mc_OTHER":1.5,
}
FEATURE_COLS = LOG_COLS + LINEAR_COLS + FLAG_COLS + WT_COLS + MC_COLS

# Default cluster names based on what each v3 cluster contains. The user can
# rename in-page; these are just a starting point.
DEFAULT_NAMES = {
    100: "T1 light raiders",
    101: "T1 anti-air",
    102: "T1 short artillery",
    103: "T1 mid mobiles",
    104: "T1 rocket bots",
    105: "T1 amphibious AA bots",
    106: "T1 heavy assault",
    200: "T2 EMP",
    201: "T2 mid mobiles",
    202: "T2 anti-air",
    203: "Crawling bombs",
    204: "Mobile anti-nuke",
    205: "T2 heavy assault",
    206: "T2 amphibious tanks",
    207: "T2 hovercraft AA-capable",
    208: "T2 long-range artillery",
    300: "T3 experimental / EMP",
    301: "T3 heavy mobile",
    302: "T3 long-range",
}


def build_X(rows):
    feats = []
    for c in LOG_COLS:    feats.append([math.log1p(r[c]) for r in rows])
    for c in LINEAR_COLS: feats.append([r[c] for r in rows])
    for c in FLAG_COLS:   feats.append([r[c] for r in rows])
    for c in WT_COLS:     feats.append([min(r[c], 4.0) for r in rows])
    for c in MC_COLS:     feats.append([r[c] for r in rows])
    return np.array(feats).T.astype(float)


def main():
    feat = "/sessions/relaxed-nice-maxwell/mnt/outputs/unit_features_v2_land_combat.csv"
    cats = "/sessions/relaxed-nice-maxwell/mnt/outputs/unit_categories_v3.json"
    rows = list(csv.DictReader(open(feat)))
    assignments = json.loads(Path(cats).read_text())["assignments"]
    for r in rows:
        for k in FEATURE_COLS:
            v = r.get(k)
            try: r[k] = float(v) if v not in (None,"") else 0.0
            except ValueError: r[k] = 0.0
        r["cluster"] = int(assignments[r["code"]])

    X = StandardScaler().fit_transform(build_X(rows))
    weights = np.array([WEIGHTS.get(c, 1.0) for c in FEATURE_COLS])
    Xw = X * weights
    pca = PCA(n_components=2, random_state=42).fit(Xw)
    XP = pca.transform(Xw)
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, init="pca",
                learning_rate="auto").fit_transform(Xw)
    def norm(a):
        lo, hi = a.min(0), a.max(0)
        return (a - lo) / np.where(hi-lo==0, 1, hi-lo)
    XP_n = norm(XP); tsne_n = norm(tsne)

    units = []
    for i, r in enumerate(rows):
        units.append({
            "code": r["code"], "name": r.get("name") or r["code"],
            "faction": r["faction"], "folder": r["folder"],
            "movementclass": r.get("movementclass",""),
            "tier": int(float(r["techlevel"])),
            "hp": int(r["health"]), "metal": int(r["metalcost"]),
            "energy": int(r["energycost"]), "buildtime": int(r["buildtime"]),
            "speed": round(r["speed"], 1), "dps": round(r["best_dps"], 1),
            "range": int(r["max_range"]), "aoe": int(r["max_aoe"]),
            "n_weapons": int(r["n_weapons"]),
            "has_aa": int(r["has_aa"]), "has_sub": int(r["has_anti_sub"]),
            "has_par": int(r["has_paralyzer"]),
            "aa_frac": round(r["damage_vs_air"], 2),
            "traj": int(r["trajectory_high"]),
            "rngT": int(r["range_tier"]),
            "default_cluster": int(r["cluster"]),
            "pca_x": round(float(XP_n[i,0]), 4), "pca_y": round(float(XP_n[i,1]), 4),
            "tsne_x": round(float(tsne_n[i,0]), 4), "tsne_y": round(float(tsne_n[i,1]), 4),
        })

    payload = {
        "units": units,
        "default_names": {str(k): v for k, v in DEFAULT_NAMES.items()},
        "pca_var": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    }
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",",":")))
    out = Path("/sessions/relaxed-nice-maxwell/mnt/outputs/cluster_editor.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>BAR Cluster Editor</title>
<style>
  :root{--bg:#14161a;--panel:#1c2128;--border:#2f343c;--text:#e6e7eb;--muted:#9aa1ac;--accent:#4dabf7;--green:#51cf66;--red:#ff6b6b;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Ubuntu,sans-serif;}
  header{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:10px;align-items:center;position:sticky;top:0;background:var(--bg);z-index:20;}
  header h1{margin:0;font-size:15px;font-weight:600;}
  header .sub{color:var(--muted);font-size:11px;font-weight:400;margin-left:4px;}
  .stats{color:var(--muted);margin-left:auto;font-size:12px;font-family:ui-monospace,monospace;}
  button,select,input[type="text"],input[type="file"]{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:5px 10px;font:inherit;cursor:pointer;}
  button:hover{border-color:var(--accent);}
  button.active{background:#234d3a;border-color:var(--green);}
  button.primary{background:var(--accent);border-color:var(--accent);color:#14161a;font-weight:600;}
  button.danger{color:var(--red);}
  .row{display:grid;grid-template-columns:1fr 300px;min-height:48vh;border-bottom:1px solid var(--border);}
  #plot{position:relative;background:#0e1014;} #plot svg{width:100%;height:100%;display:block;}
  #plot circle{transition:r 100ms,stroke-width 100ms;cursor:pointer;}
  #plot circle.dim{opacity:.12;} #plot circle.hi{stroke:#fff;stroke-width:2;}
  .legend{padding:10px;overflow-y:auto;border-left:1px solid var(--border);background:var(--panel);}
  .legend h3{margin:0 0 6px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
  .legend-item{display:flex;align-items:center;gap:6px;padding:4px 6px;margin-bottom:2px;cursor:pointer;border-radius:4px;user-select:none;font-size:12px;}
  .legend-item:hover{background:#2a2f37;} .legend-item.off{opacity:.35;}
  .swatch{width:12px;height:12px;border-radius:50%;flex-shrink:0;border:1px solid rgba(255,255,255,.15);}
  .legend-item .lbl{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .legend-item .n{color:var(--muted);font-size:11px;font-family:ui-monospace,monospace;}
  .tooltip{position:fixed;pointer-events:none;z-index:50;background:rgba(20,22,26,.96);border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-size:12px;max-width:300px;box-shadow:0 4px 16px rgba(0,0,0,.4);display:none;}
  .tooltip h4{margin:0 0 4px;font-size:13px;}
  .tooltip .tag{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;line-height:14px;background:#2a2f37;margin-right:3px;color:var(--muted);}
  .tooltip .stats-row{color:var(--muted);margin-top:3px;font-family:ui-monospace,monospace;font-size:11px;}
  main{padding:14px 16px;}
  .toolbar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;}
  .toolbar .save-status{margin-left:auto;color:var(--muted);font-size:11px;font-family:ui-monospace,monospace;}
  .cluster-block{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin-bottom:8px;transition:border-color 100ms,background 100ms;}
  .cluster-block.drag-over{border-color:var(--accent);background:#1e2a35;}
  .cluster-block.unassigned{border-color:#fa5252;background:#2a1b1e;}
  .cluster-head{display:flex;gap:10px;align-items:center;margin-bottom:6px;}
  .cluster-head .swatch{width:14px;height:14px;}
  .cluster-head .name-input{background:transparent;border:1px solid transparent;color:var(--text);font-weight:600;font-size:13px;flex:1;padding:2px 4px;border-radius:3px;min-width:120px;}
  .cluster-head .name-input:hover{border-color:var(--border);}
  .cluster-head .name-input:focus{border-color:var(--accent);background:var(--bg);outline:none;}
  .cluster-head .summary{color:var(--muted);font-size:11px;font-family:ui-monospace,monospace;}
  .cluster-head .actions{display:flex;gap:4px;}
  .cluster-head .actions button{padding:2px 8px;font-size:11px;}
  .members{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:5px;min-height:30px;}
  .unit{background:#14161a;border:1px solid var(--border);border-radius:6px;padding:5px 8px;font-size:11px;cursor:grab;user-select:none;transition:transform 80ms,opacity 80ms,border-color 80ms;}
  .unit:hover{border-color:var(--accent);}
  .unit:active{cursor:grabbing;}
  .unit.dragging{opacity:.3;transform:scale(.97);}
  .unit .nm{font-weight:600;}
  .unit .cd{color:var(--muted);font-family:ui-monospace,monospace;font-size:10px;margin-left:6px;}
  .unit .row2{color:var(--muted);font-size:10px;font-family:ui-monospace,monospace;margin-top:2px;}
  .empty-hint{color:var(--muted);font-style:italic;padding:6px 4px;font-size:11px;}
</style></head><body>
<header>
  <h1>BAR Cluster Editor <span class="sub">— drag units, rename clusters, save your curation</span></h1>
  <div>View:
    <button id="btn-tsne" class="active">t-SNE</button>
    <button id="btn-pca">PCA</button>
  </div>
  <span class="stats" id="stats"></span>
</header>
<div class="row">
  <div id="plot"></div>
  <div class="legend"><h3>clusters · click to fade</h3><div id="legend"></div></div>
</div>
<div id="tt" class="tooltip"></div>
<main>
  <div class="toolbar">
    <button id="btn-add" class="primary">+ New cluster</button>
    <button id="btn-reset">Reset to defaults</button>
    <button id="btn-export" class="primary">Save curation…</button>
    <button id="btn-import">Load curation…</button>
    <input id="file-import" type="file" accept="application/json" hidden>
    <span class="save-status" id="save-status">localStorage: synced</span>
  </div>
  <div id="clusters"></div>
</main>
<script>
const DATA = __PAYLOAD__;
const STORAGE_KEY = "bar_cluster_curation_v1";

// ---- state ----
// clusters: Array of {id, name, codes: Set<string>}
// We keep them as arrays so order is preserved and the user can re-arrange.
let clusters = [];
const unitByCode = new Map(DATA.units.map(u => [u.code, u]));
let unitToCluster = new Map(); // code -> cluster id (computed)

// ---- color palette (mirrors v3) ----
function clusterColor(id) {
  // ID convention: tier*100 + sub. For custom clusters >= 900 we use a fallback.
  const tier = Math.floor(id / 100);
  const sub  = id % 100;
  const pals = {
    1: ["#4dabf7","#74c0fc","#22b8cf","#15aabf","#63e6be","#a9e34b","#94d82d","#37b24d"],
    2: ["#ffa94d","#ffd43b","#fcc419","#ff8787","#f783ac","#e599f7","#b197fc","#9775fa","#845ef7"],
    3: ["#ff6b6b","#fa5252","#cc5de8","#ae3ec9","#7950f2"],
    9: ["#868e96","#adb5bd","#ced4da","#dee2e6"],  // custom user-added clusters
  };
  const p = pals[tier] || pals[9];
  return p[sub % p.length];
}

function defaultClusterName(id) {
  return DATA.default_names[String(id)] || `cluster ${id}`;
}

// ---- initialize from defaults or localStorage ----
function loadFromDefaults() {
  const byId = new Map();
  for (const u of DATA.units) {
    const cid = u.default_cluster;
    if (!byId.has(cid)) byId.set(cid, { id: cid, name: defaultClusterName(cid), codes: new Set() });
    byId.get(cid).codes.add(u.code);
  }
  clusters = [...byId.values()].sort((a, b) => a.id - b.id);
  rebuildIndex();
}

function loadFromState(state) {
  clusters = (state.clusters || []).map(c => ({
    id: c.id, name: c.name, codes: new Set(c.codes || [])
  }));
  // Make sure every unit is assigned to some cluster -- orphans go to "Unassigned"
  const assigned = new Set();
  for (const c of clusters) for (const code of c.codes) assigned.add(code);
  const orphans = DATA.units.filter(u => !assigned.has(u.code));
  if (orphans.length) {
    let unassigned = clusters.find(c => c.id === 999);
    if (!unassigned) {
      unassigned = { id: 999, name: "Unassigned", codes: new Set() };
      clusters.push(unassigned);
    }
    for (const u of orphans) unassigned.codes.add(u.code);
  }
  rebuildIndex();
}

function rebuildIndex() {
  unitToCluster.clear();
  for (const c of clusters) for (const code of c.codes) unitToCluster.set(code, c.id);
}

function persist() {
  const state = {
    version: 1,
    saved_at: new Date().toISOString(),
    clusters: clusters.map(c => ({ id: c.id, name: c.name, codes: [...c.codes].sort() })),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  const el = document.getElementById("save-status");
  el.textContent = "localStorage: saved " + new Date().toLocaleTimeString();
}

// ---- pick up where we left off ----
try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  if (saved && saved.clusters) loadFromState(saved);
  else loadFromDefaults();
} catch {
  loadFromDefaults();
}

// ---- rendering ----
let proj = "tsne";
const offClusters = new Set();

function renderPlot() {
  const plot = document.getElementById("plot");
  const rect = plot.getBoundingClientRect();
  const W = rect.width, H = rect.height, pad = 30;
  const xf = proj === "tsne" ? "tsne_x" : "pca_x";
  const yf = proj === "tsne" ? "tsne_y" : "pca_y";
  const svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${
    DATA.units.map((u, i) => {
      const cx = pad + u[xf] * (W - 2*pad);
      const cy = pad + (1 - u[yf]) * (H - 2*pad);
      const cid = unitToCluster.get(u.code);
      const color = cid != null ? clusterColor(cid) : "#666";
      const dim = (cid != null && offClusters.has(cid)) ? "dim" : "";
      return `<circle data-code="${u.code}" class="${dim}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5" fill="${color}" fill-opacity="0.82" stroke="#0e1014" stroke-width="1"/>`;
    }).join("")
  }</svg>`;
  plot.innerHTML = svg;
}

function renderLegend() {
  const root = document.getElementById("legend");
  root.innerHTML = clusters.map(c => `
    <div class="legend-item${offClusters.has(c.id) ? " off" : ""}" data-cluster="${c.id}">
      <div class="swatch" style="background:${clusterColor(c.id)}"></div>
      <div class="lbl">c${c.id}: ${c.name}</div>
      <div class="n">${c.codes.size}</div>
    </div>`).join("");
}

function renderClusters() {
  const root = document.getElementById("clusters");
  root.innerHTML = clusters.map(c => clusterBlockHtml(c)).join("");
  // Wire up drag-and-drop on every block.
  for (const block of root.querySelectorAll(".cluster-block")) wireBlock(block);
  for (const card of root.querySelectorAll(".unit")) wireCard(card);
  // And inline rename, delete, etc.
  for (const inp of root.querySelectorAll(".name-input")) {
    inp.addEventListener("blur", commitRename);
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); inp.blur(); }
      if (e.key === "Escape") { inp.value = clusterById(parseInt(inp.dataset.cid, 10)).name; inp.blur(); }
    });
  }
  for (const btn of root.querySelectorAll(".btn-del")) {
    btn.addEventListener("click", () => deleteCluster(parseInt(btn.dataset.cid, 10)));
  }
}

function clusterBlockHtml(c) {
  const members = [...c.codes].map(code => unitByCode.get(code)).filter(Boolean);
  members.sort((a, b) => b.hp - a.hp || a.code.localeCompare(b.code));
  const avg = (k) => members.length ? Math.round(members.reduce((s, u) => s + u[k], 0) / members.length) : 0;
  return `
    <div class="cluster-block${c.id===999 ? " unassigned" : ""}" data-cid="${c.id}">
      <div class="cluster-head">
        <div class="swatch" style="background:${clusterColor(c.id)}"></div>
        <input class="name-input" data-cid="${c.id}" value="${(c.name || "").replace(/"/g, "&quot;")}">
        <div class="summary">id ${c.id} · n=${c.codes.size}${members.length ? ` · hp ${avg("hp")} · dps ${avg("dps")} · rng ${avg("range")} · spd ${avg("speed")}` : ""}</div>
        <div class="actions">
          <button class="btn-del danger" data-cid="${c.id}" title="Delete cluster (units go to Unassigned)">delete</button>
        </div>
      </div>
      <div class="members" data-cid="${c.id}">
        ${members.length === 0
          ? `<div class="empty-hint">empty cluster — drag a unit here</div>`
          : members.map(u => unitCardHtml(u)).join("")}
      </div>
    </div>`;
}

function unitCardHtml(u) {
  return `
    <div class="unit" draggable="true" data-code="${u.code}">
      <div><span class="nm">${u.name || u.code}</span><span class="cd">${u.code}</span></div>
      <div class="row2">${u.faction[0]} · T${u.tier} · ${u.movementclass} · hp ${u.hp} · dps ${u.dps} · rng ${u.range}${u.has_aa?' · AA':''}${u.has_par?' · EMP':''}${u.traj?' · arc':''}</div>
    </div>`;
}

function clusterById(id) { return clusters.find(c => c.id === id); }

// ---- drag and drop ----
let dragging = null;     // currently-dragging code
function wireCard(card) {
  card.addEventListener("dragstart", (e) => {
    dragging = card.dataset.code;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", dragging);
    card.classList.add("dragging");
  });
  card.addEventListener("dragend", () => {
    card.classList.remove("dragging");
    dragging = null;
    document.querySelectorAll(".cluster-block.drag-over").forEach(b => b.classList.remove("drag-over"));
  });
}
function wireBlock(block) {
  block.addEventListener("dragover", (e) => {
    if (!dragging) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    block.classList.add("drag-over");
  });
  block.addEventListener("dragleave", (e) => {
    if (e.target === block) block.classList.remove("drag-over");
  });
  block.addEventListener("drop", (e) => {
    e.preventDefault();
    block.classList.remove("drag-over");
    const code = dragging || e.dataTransfer.getData("text/plain");
    if (!code) return;
    const targetId = parseInt(block.dataset.cid, 10);
    moveUnit(code, targetId);
  });
}

function moveUnit(code, targetId) {
  const sourceId = unitToCluster.get(code);
  if (sourceId === targetId) return;
  if (sourceId != null) {
    const src = clusterById(sourceId);
    if (src) src.codes.delete(code);
  }
  const tgt = clusterById(targetId);
  if (tgt) tgt.codes.add(code);
  // If source cluster is now empty and was a user-added one (id >= 900), prune it.
  if (sourceId != null && sourceId >= 900) {
    const src = clusterById(sourceId);
    if (src && src.codes.size === 0 && src.id !== 999) {
      clusters = clusters.filter(c => c.id !== sourceId);
    }
  }
  rebuildIndex();
  persist();
  renderAll();
}

// ---- add / delete clusters ----
function addCluster() {
  // Use 900+ for user-added clusters so they don't collide with tier IDs.
  let nextId = 900;
  for (const c of clusters) if (c.id >= 900 && c.id < 999) nextId = Math.max(nextId, c.id + 1);
  clusters.push({ id: nextId, name: `New cluster ${nextId - 899}`, codes: new Set() });
  persist();
  renderAll();
  // Focus the new cluster's name field.
  setTimeout(() => {
    const inp = document.querySelector(`.name-input[data-cid="${nextId}"]`);
    if (inp) { inp.focus(); inp.select(); }
  }, 0);
}

function deleteCluster(id) {
  const c = clusterById(id);
  if (!c) return;
  if (c.codes.size > 0) {
    if (!confirm(`Cluster "${c.name}" has ${c.codes.size} units. Move them to "Unassigned"?`)) return;
    let un = clusterById(999);
    if (!un) { un = { id: 999, name: "Unassigned", codes: new Set() }; clusters.push(un); }
    for (const code of c.codes) un.codes.add(code);
  }
  clusters = clusters.filter(x => x.id !== id);
  rebuildIndex(); persist(); renderAll();
}

function commitRename(e) {
  const inp = e.target;
  const cid = parseInt(inp.dataset.cid, 10);
  const c = clusterById(cid);
  if (!c) return;
  const v = inp.value.trim() || `cluster ${cid}`;
  if (v !== c.name) {
    c.name = v;
    persist();
    renderLegend();
  }
}

function resetToDefaults() {
  if (!confirm("Discard your edits and restore the original cluster assignments?")) return;
  localStorage.removeItem(STORAGE_KEY);
  loadFromDefaults();
  persist();
  renderAll();
}

// ---- save / load ----
document.getElementById("btn-export").onclick = () => {
  const payload = {
    version: 1,
    generated: new Date().toISOString(),
    n_units: DATA.units.length,
    n_clusters: clusters.length,
    clusters: clusters.map(c => ({ id: c.id, name: c.name, codes: [...c.codes].sort() })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "bar_unit_clusters.json";
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(a.href);
};
document.getElementById("btn-import").onclick = () => document.getElementById("file-import").click();
document.getElementById("file-import").addEventListener("change", async (e) => {
  const file = e.target.files[0]; if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!data.clusters || !Array.isArray(data.clusters)) throw new Error("missing clusters[]");
    loadFromState(data); persist(); renderAll();
    alert(`Loaded ${clusters.length} clusters.`);
  } catch (err) { alert("Couldn't read that file: " + err.message); }
});
document.getElementById("btn-add").onclick = addCluster;
document.getElementById("btn-reset").onclick = resetToDefaults;

// ---- legend & plot interactions ----
document.getElementById("legend").addEventListener("click", (e) => {
  const item = e.target.closest(".legend-item"); if (!item) return;
  const cid = parseInt(item.dataset.cluster, 10);
  if (offClusters.has(cid)) offClusters.delete(cid); else offClusters.add(cid);
  renderPlot(); renderLegend();
});
const tt = document.getElementById("tt");
function showTip(code, ev) {
  const u = unitByCode.get(code); if (!u) return;
  const cid = unitToCluster.get(code);
  const cn = cid != null ? clusterById(cid) : null;
  tt.innerHTML = `
    <h4 style="color:${cid != null ? clusterColor(cid) : "#fff"}">${u.name || u.code}
      <span style="color:var(--muted);font-weight:400;font-size:11px">${u.code}</span></h4>
    <div><span class="tag">${u.faction}</span><span class="tag">T${u.tier}</span>
         <span class="tag">${u.movementclass}</span>
         <span class="tag">${cn ? "c"+cn.id+": "+cn.name : "unassigned"}</span></div>
    <div class="stats-row">hp ${u.hp} · dps ${u.dps} · rng ${u.range} (tier ${u.rngT}) · aoe ${u.aoe}</div>
    <div class="stats-row">M ${u.metal} · E ${u.energy} · bt ${u.buildtime} · spd ${u.speed}</div>
    <div class="stats-row">aa_frac ${u.aa_frac}${u.has_aa?' · AA':''}${u.has_sub?' · sub':''}${u.has_par?' · paralyzer':''}${u.traj?' · arcing':''}</div>`;
  tt.style.left = (ev.clientX + 12) + "px";
  tt.style.top  = (ev.clientY + 12) + "px";
  tt.style.display = "block";
}
function hideTip() { tt.style.display = "none"; }
document.getElementById("plot").addEventListener("mousemove", (e) => {
  const t = e.target;
  if (t && t.tagName === "circle") {
    showTip(t.dataset.code, e);
    document.querySelectorAll("#plot circle.hi").forEach(c => c.classList.remove("hi"));
    t.classList.add("hi");
  } else hideTip();
});
document.getElementById("plot").addEventListener("mouseleave", hideTip);

// Toggle projection
document.getElementById("btn-tsne").onclick = () => { proj="tsne";
  document.getElementById("btn-tsne").classList.add("active");
  document.getElementById("btn-pca").classList.remove("active"); renderPlot(); };
document.getElementById("btn-pca").onclick = () => { proj="pca";
  document.getElementById("btn-pca").classList.add("active");
  document.getElementById("btn-tsne").classList.remove("active"); renderPlot(); };
window.addEventListener("resize", renderPlot);

// ---- full re-render ----
function renderAll() {
  renderPlot();
  renderLegend();
  renderClusters();
  document.getElementById("stats").textContent =
    `${DATA.units.length} units · ${clusters.length} clusters · ${proj}`;
}
renderAll();
persist();   // make sure localStorage has the initial state too
</script></body></html>
"""
if __name__ == "__main__": main()
