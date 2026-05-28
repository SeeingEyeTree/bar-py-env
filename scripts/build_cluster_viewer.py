"""Build a single-file HTML visualization of unit clusters.

Reads:
  unit_features_land_combat.csv     (the filtered features)
  unit_categories_land_combat.json  (cluster assignments)

Writes:
  cluster_viewer.html  (open in any browser; no internet needed)
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Same feature list as cluster_units.py
LOG_COLS = ["buildtime","metalcost","energycost","health","best_dps","total_dps",
            "max_range","max_aoe","sightdistance","turnrate"]
LINEAR_COLS = ["speed","maxacc","maxdec","maxslope","maxwaterdepth",
               "min_range","n_weapons","techlevel"]
FLAG_COLS = ["canmove","turninplace","has_aa","has_anti_sub","has_paralyzer"]
WT_COLS = ["wt_cannon","wt_beamlaser","wt_lasercannon","wt_missilelauncher",
           "wt_starburstlauncher","wt_torpedolauncher","wt_emgcannon",
           "wt_aircraftbomb","wt_flame","wt_dgun","wt_shield","wt_other"]
FEATURE_COLS = LOG_COLS + LINEAR_COLS + FLAG_COLS + WT_COLS


def load_rows(features_path, cats_path):
    rows = list(csv.DictReader(open(features_path, encoding="utf-8")))
    cats = json.loads(Path(cats_path).read_text())["assignments"]
    for r in rows:
        for k in FEATURE_COLS:
            v = r.get(k)
            try: r[k] = float(v) if v not in (None, "") else 0.0
            except ValueError: r[k] = 0.0
        r["cluster"] = int(cats.get(r["code"], -1))
    return rows


def build_X(rows):
    feats = []
    for c in LOG_COLS: feats.append([math.log1p(r[c]) for r in rows])
    for c in LINEAR_COLS: feats.append([r[c] for r in rows])
    for c in FLAG_COLS: feats.append([r[c] for r in rows])
    for c in WT_COLS: feats.append([min(r[c], 4.0) for r in rows])
    return np.array(feats).T.astype(float)


def main():
    rows = load_rows(
        "/sessions/relaxed-nice-maxwell/mnt/bar-py-env/bar_env/units/unit_features_land_combat.csv",
        "/sessions/relaxed-nice-maxwell/mnt/bar-py-env/bar_env/units/unit_categories_land_combat.json",
    )
    print(f"loaded {len(rows)} rows")
    X = StandardScaler().fit_transform(build_X(rows))
    print(f"feature matrix: {X.shape}")

    # PCA -- linear, axes interpretable
    pca = PCA(n_components=2, random_state=42).fit(X)
    XP = pca.transform(X)
    print(f"pca explained variance: {pca.explained_variance_ratio_}")

    # t-SNE -- non-linear, separates clusters visually
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, init="pca",
                learning_rate="auto").fit_transform(X)
    print(f"t-sne shape: {tsne.shape}")

    # Normalize coords to [0, 1] for easy HTML positioning
    def norm(a):
        lo, hi = a.min(0), a.max(0)
        return (a - lo) / np.where(hi - lo == 0, 1, hi - lo)
    XP_n = norm(XP)
    tsne_n = norm(tsne)

    units = []
    for i, r in enumerate(rows):
        units.append({
            "code": r["code"],
            "name": r.get("name", "") or r["code"],
            "faction": r["faction"],
            "folder": r["folder"],
            "unitgroup": r.get("unitgroup", ""),
            "cluster": r["cluster"],
            "tier": int(float(r["techlevel"])),
            "hp": int(r["health"]),
            "metal": int(r["metalcost"]),
            "energy": int(r["energycost"]),
            "buildtime": int(r["buildtime"]),
            "speed": round(r["speed"], 1),
            "dps": round(r["best_dps"], 1),
            "range": int(r["max_range"]),
            "aoe": int(r["max_aoe"]),
            "n_weapons": int(r["n_weapons"]),
            "has_aa": int(r["has_aa"]),
            "has_sub": int(r["has_anti_sub"]),
            "has_par": int(r["has_paralyzer"]),
            "pca_x": round(float(XP_n[i, 0]), 4),
            "pca_y": round(float(XP_n[i, 1]), 4),
            "tsne_x": round(float(tsne_n[i, 0]), 4),
            "tsne_y": round(float(tsne_n[i, 1]), 4),
        })

    # Compute per-cluster summary
    by_cluster = {}
    for u in units:
        by_cluster.setdefault(u["cluster"], []).append(u)
    clusters = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        clusters.append({
            "id": cid,
            "n": len(members),
            "avg_hp": int(np.mean([m["hp"] for m in members])),
            "avg_dps": int(np.mean([m["dps"] for m in members])),
            "avg_range": int(np.mean([m["range"] for m in members])),
            "avg_metal": int(np.mean([m["metal"] for m in members])),
            "avg_speed": round(float(np.mean([m["speed"] for m in members])), 1),
            "members": sorted(members, key=lambda m: (-m["hp"], m["code"])),
        })

    payload = {"units": units, "clusters": clusters,
               "pca_var": [round(float(v), 3) for v in pca.explained_variance_ratio_]}

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",",":")))
    out = Path("/sessions/relaxed-nice-maxwell/mnt/outputs/cluster_viewer.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BAR Land-Combat Cluster Viewer</title>
<style>
  :root {
    --bg: #14161a; --panel: #1c2128; --border: #2f343c;
    --text: #e6e7eb; --muted: #9aa1ac;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:13px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI',
              Roboto, Ubuntu, sans-serif; }
  header { padding:12px 20px; border-bottom:1px solid var(--border);
           display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  .stats { color:var(--muted); margin-left:auto; font-size:12px; }
  .controls { display:flex; gap:8px; align-items:center; }
  button, select { background:var(--panel); color:var(--text);
                   border:1px solid var(--border); border-radius:6px;
                   padding:6px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:#4dabf7; }
  button.active { background:#234d3a; border-color:#51cf66; }
  .row { display:grid; grid-template-columns:1fr 280px; gap:0;
         min-height:60vh; border-bottom:1px solid var(--border); }
  #plot { position:relative; background:#0e1014; }
  #plot svg { width:100%; height:100%; display:block; }
  #plot circle { transition: r 100ms ease, stroke-width 100ms ease;
                 cursor: pointer; }
  #plot circle.dim { opacity:0.12; }
  #plot circle.hi { stroke:#fff; stroke-width:2; }
  .legend { padding:10px; overflow-y:auto;
            border-left:1px solid var(--border); background:var(--panel); }
  .legend h3 { margin:0 0 8px; font-size:12px; color:var(--muted);
               text-transform:uppercase; letter-spacing:0.5px; }
  .legend-item { display:flex; align-items:center; gap:8px;
                 padding:4px 6px; margin-bottom:2px; cursor:pointer;
                 border-radius:4px; user-select:none; }
  .legend-item:hover { background:#2a2f37; }
  .legend-item.off { opacity:0.35; }
  .swatch { width:14px; height:14px; border-radius:50%;
            border:1px solid rgba(255,255,255,0.15); flex-shrink:0; }
  .legend-item .lbl { flex:1; min-width:0; overflow:hidden;
                      text-overflow:ellipsis; white-space:nowrap; }
  .legend-item .n { color:var(--muted); font-size:11px;
                    font-family:ui-monospace, Menlo, monospace; }
  .tooltip { position:fixed; pointer-events:none; z-index:50;
             background:rgba(20,22,26,0.96); border:1px solid var(--border);
             border-radius:6px; padding:8px 10px; font-size:12px;
             max-width:280px; box-shadow:0 4px 16px rgba(0,0,0,0.4);
             display:none; }
  .tooltip h4 { margin:0 0 4px; font-size:13px; }
  .tooltip .tag { display:inline-block; padding:1px 6px;
                  border-radius:8px; font-size:10px; line-height:14px;
                  background:#2a2f37; margin-right:3px; color:var(--muted); }
  .tooltip .stats-row { color:var(--muted); margin-top:4px;
                        font-family:ui-monospace, Menlo, monospace;
                        font-size:11px; }
  main { padding:16px 20px; }
  main h2 { font-size:14px; color:var(--muted); margin:16px 0 8px;
            text-transform:uppercase; letter-spacing:0.5px; }
  .cluster-block { background:var(--panel); border:1px solid var(--border);
                   border-radius:8px; padding:10px 12px; margin-bottom:10px; }
  .cluster-head { display:flex; gap:10px; align-items:center;
                  margin-bottom:8px; }
  .cluster-head .swatch { width:16px; height:16px; }
  .cluster-head .title { font-weight:600; }
  .cluster-head .summary { color:var(--muted); font-size:12px;
                           font-family:ui-monospace, Menlo, monospace;
                           margin-left:auto; }
  .members { display:grid; grid-template-columns:repeat(auto-fill,
             minmax(220px, 1fr)); gap:6px; }
  .unit { background:#14161a; border:1px solid var(--border);
          border-radius:6px; padding:6px 10px; font-size:12px;
          cursor:pointer; transition: border-color 80ms ease; }
  .unit:hover { border-color:#4dabf7; }
  .unit .nm { font-weight:600; }
  .unit .cd { color:var(--muted); font-family:ui-monospace, monospace;
              font-size:10px; margin-left:6px; }
  .unit .row2 { color:var(--muted); font-size:11px;
                font-family:ui-monospace, monospace; margin-top:2px; }
</style>
</head>
<body>
<header>
  <h1>BAR Land-Combat Cluster Viewer</h1>
  <div class="controls">
    Projection:
    <button id="btn-tsne" class="active">t-SNE</button>
    <button id="btn-pca">PCA</button>
    Color by:
    <button id="btn-cluster" class="active">cluster</button>
    <button id="btn-faction">faction</button>
    <button id="btn-tier">tier</button>
  </div>
  <span class="stats" id="stats"></span>
</header>

<div class="row">
  <div id="plot"></div>
  <div class="legend">
    <h3>clusters</h3>
    <div id="legend"></div>
  </div>
</div>
<div id="tt" class="tooltip"></div>

<main>
  <h2>per-cluster contents</h2>
  <div id="clusters"></div>
</main>

<script>
const DATA = __PAYLOAD__;
const PALETTE = [
  "#ff6b6b","#4dabf7","#51cf66","#ffa94d","#b197fc","#22b8cf",
  "#ffe066","#f783ac","#a9e34b","#9aa1ac","#fcc419","#15aabf",
  "#cc5de8","#74c0fc","#ff8787","#94d82d","#e599f7","#63e6be"
];
function clusterColor(c) { return PALETTE[((c % PALETTE.length) + PALETTE.length) % PALETTE.length]; }
function factionColor(f) { return f==="Armada" ? "#4dabf7" : f==="Cortex" ? "#ff6b6b" : "#9aa1ac"; }
function tierColor(t) { return t===1 ? "#51cf66" : t===2 ? "#ffa94d" : t===3 ? "#ff6b6b" : "#9aa1ac"; }

let proj = "tsne";        // "tsne" | "pca"
let colorBy = "cluster";  // "cluster" | "faction" | "tier"
const offClusters = new Set();

function uColor(u) {
  if (colorBy === "cluster") return clusterColor(u.cluster);
  if (colorBy === "faction") return factionColor(u.faction);
  if (colorBy === "tier")    return tierColor(u.tier);
  return "#9aa1ac";
}

function render() {
  const plot = document.getElementById("plot");
  const rect = plot.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  const pad = 30;
  const xf = proj === "tsne" ? "tsne_x" : "pca_x";
  const yf = proj === "tsne" ? "tsne_y" : "pca_y";

  const svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${
    DATA.units.map((u, i) => {
      const cx = pad + u[xf] * (W - 2*pad);
      const cy = pad + (1 - u[yf]) * (H - 2*pad);  // flip Y so up is up
      const dim = offClusters.has(u.cluster) ? "dim" : "";
      return `<circle data-i="${i}" class="${dim}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5" fill="${uColor(u)}" fill-opacity="0.82" stroke="#0e1014" stroke-width="1"/>`;
    }).join("")
  }</svg>`;
  plot.innerHTML = svg;

  // legend
  const leg = document.getElementById("legend");
  if (colorBy === "cluster") {
    leg.innerHTML = DATA.clusters.map(c => `
      <div class="legend-item${offClusters.has(c.id) ? " off" : ""}" data-cluster="${c.id}">
        <div class="swatch" style="background:${clusterColor(c.id)}"></div>
        <div class="lbl">cluster ${c.id} (${c.members.slice(0,3).map(m=>m.name||m.code).join(", ")})</div>
        <div class="n">${c.n}</div>
      </div>`).join("");
  } else if (colorBy === "faction") {
    leg.innerHTML = `
      <div class="legend-item"><div class="swatch" style="background:#4dabf7"></div><div class="lbl">Armada</div></div>
      <div class="legend-item"><div class="swatch" style="background:#ff6b6b"></div><div class="lbl">Cortex</div></div>`;
  } else {
    leg.innerHTML = `
      <div class="legend-item"><div class="swatch" style="background:#51cf66"></div><div class="lbl">T1</div></div>
      <div class="legend-item"><div class="swatch" style="background:#ffa94d"></div><div class="lbl">T2</div></div>
      <div class="legend-item"><div class="swatch" style="background:#ff6b6b"></div><div class="lbl">T3</div></div>`;
  }

  document.getElementById("stats").textContent =
    `${DATA.units.length} units in ${DATA.clusters.length} clusters · projection: ${proj} ${
      proj==='pca' ? `(PCs explain ${(DATA.pca_var[0]*100).toFixed(0)}% + ${(DATA.pca_var[1]*100).toFixed(0)}%)` : ''}`;
}

function renderClusterBlocks() {
  const root = document.getElementById("clusters");
  root.innerHTML = DATA.clusters.map(c => `
    <div class="cluster-block">
      <div class="cluster-head">
        <div class="swatch" style="background:${clusterColor(c.id)}"></div>
        <div class="title">cluster ${c.id}</div>
        <div class="summary">n=${c.n} · avg hp=${c.avg_hp} · dps=${c.avg_dps} · range=${c.avg_range} · metal=${c.avg_metal} · speed=${c.avg_speed}</div>
      </div>
      <div class="members">${c.members.map(u => `
        <div class="unit" data-code="${u.code}">
          <div><span class="nm">${u.name||u.code}</span><span class="cd">${u.code}</span></div>
          <div class="row2">${u.faction[0]} · T${u.tier} · hp ${u.hp} · dps ${u.dps} · rng ${u.range}${u.has_aa? ' · AA':''}${u.has_par?' · EMP':''}</div>
        </div>`).join("")}</div>
    </div>`).join("");
}

// Tooltip + interactions
const tt = document.getElementById("tt");
function showTip(i, ev) {
  const u = DATA.units[i];
  const color = uColor(u);
  tt.innerHTML = `
    <h4 style="color:${color}">${u.name || u.code} <span style="color:var(--muted); font-weight:400; font-size:11px">${u.code}</span></h4>
    <div>
      <span class="tag">${u.faction}</span>
      <span class="tag">T${u.tier}</span>
      <span class="tag">${u.folder}</span>
      <span class="tag">cluster ${u.cluster}</span>
    </div>
    <div class="stats-row">hp ${u.hp} · dps ${u.dps} · rng ${u.range} · aoe ${u.aoe}</div>
    <div class="stats-row">M ${u.metal} · E ${u.energy} · bt ${u.buildtime} · spd ${u.speed}</div>
    <div class="stats-row">weapons ${u.n_weapons}${u.has_aa?' · AA':''}${u.has_sub?' · sub':''}${u.has_par?' · paralyzer':''}</div>`;
  tt.style.left = (ev.clientX + 12) + "px";
  tt.style.top = (ev.clientY + 12) + "px";
  tt.style.display = "block";
}
function hideTip() { tt.style.display = "none"; }

document.getElementById("plot").addEventListener("mousemove", (e) => {
  const t = e.target;
  if (t && t.tagName === "circle") {
    showTip(parseInt(t.dataset.i, 10), e);
    document.querySelectorAll("#plot circle.hi").forEach(c => c.classList.remove("hi"));
    t.classList.add("hi");
  } else { hideTip(); }
});
document.getElementById("plot").addEventListener("mouseleave", hideTip);

document.getElementById("legend").addEventListener("click", (e) => {
  const item = e.target.closest(".legend-item");
  if (!item || colorBy !== "cluster") return;
  const cid = parseInt(item.dataset.cluster, 10);
  if (offClusters.has(cid)) offClusters.delete(cid); else offClusters.add(cid);
  render();
});

document.getElementById("btn-tsne").onclick = () => {
  proj="tsne";
  document.getElementById("btn-tsne").classList.add("active");
  document.getElementById("btn-pca").classList.remove("active");
  render();
};
document.getElementById("btn-pca").onclick = () => {
  proj="pca";
  document.getElementById("btn-pca").classList.add("active");
  document.getElementById("btn-tsne").classList.remove("active");
  render();
};
for (const k of ["cluster","faction","tier"]) {
  document.getElementById("btn-"+k).onclick = () => {
    colorBy = k;
    ["cluster","faction","tier"].forEach(x =>
      document.getElementById("btn-"+x).classList.toggle("active", x===k));
    render();
  };
}

window.addEventListener("resize", render);
renderClusterBlocks();
render();

// Clicking a unit card in the bottom section highlights it in the plot.
document.getElementById("clusters").addEventListener("click", (e) => {
  const u = e.target.closest(".unit");
  if (!u) return;
  const code = u.dataset.code;
  const idx = DATA.units.findIndex(x => x.code === code);
  if (idx < 0) return;
  document.querySelectorAll("#plot circle.hi").forEach(c => c.classList.remove("hi"));
  const circle = document.querySelector(`#plot circle[data-i="${idx}"]`);
  if (circle) {
    circle.classList.add("hi");
    circle.scrollIntoView({block: "nearest", behavior: "smooth"});
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__": main()
