"""V2 clustering: weighted features, dropped turnrate, added movementclass + role-defining derived features."""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Skewed-positive numerics -> log1p before z-score
LOG_COLS = [
    "buildtime","metalcost","energycost","health",
    "best_dps","total_dps","max_range","max_aoe","sightdistance",
    "weapon_velocity","range_per_speed","dps_per_metal","health_per_metal",
]
# Linear numerics
LINEAR_COLS = [
    "speed","maxacc","maxdec","maxslope","maxwaterdepth",
    "min_range","n_weapons","techlevel",
    "damage_vs_air","range_tier",
]
# Binary flags
FLAG_COLS = [
    "canmove","turninplace","has_aa","has_anti_sub","has_paralyzer",
    "trajectory_high","weapon_velocity_low",
]
# Weapon-type counts -- drop the always-zero ones for land combat
WT_COLS = [
    "wt_cannon","wt_beamlaser","wt_lasercannon","wt_missilelauncher",
    "wt_emgcannon","wt_flame","wt_other",
]
# Movement-class one-hots
MC_COLS = ["mc_BOT","mc_ABOT","mc_TANK","mc_ATANK","mc_HOVER","mc_COMMANDER","mc_OTHER"]

# Per-column weight applied AFTER standardization.
#   role-defining features get 2x so they dominate role decisions
#   movementclass one-hots get 1.5x to keep chassis separation but not dominate
#   everything else 1.0x
WEIGHTS = {
    # role-defining
    "max_range": 2.0, "best_dps": 2.0, "has_aa": 2.0, "has_paralyzer": 2.0,
    "range_per_speed": 2.0, "dps_per_metal": 2.0, "damage_vs_air": 2.0,
    "trajectory_high": 2.0, "range_tier": 3.0, "weapon_velocity_low": 2.0,
    # movement-class chassis hints
    "mc_BOT": 1.5, "mc_ABOT": 1.5, "mc_TANK": 1.5,
    "mc_ATANK": 1.5, "mc_HOVER": 1.5, "mc_COMMANDER": 1.5, "mc_OTHER": 1.5,
}

FEATURE_COLS = LOG_COLS + LINEAR_COLS + FLAG_COLS + WT_COLS + MC_COLS


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in FEATURE_COLS:
            v = r.get(k)
            try: r[k] = float(v) if v not in (None, "") else 0.0
            except ValueError: r[k] = 0.0
    return rows


def build_X(rows):
    cols, feats = [], []
    for c in LOG_COLS:    feats.append([math.log1p(r[c]) for r in rows]); cols.append(c)
    for c in LINEAR_COLS: feats.append([r[c] for r in rows]); cols.append(c)
    for c in FLAG_COLS:   feats.append([r[c] for r in rows]); cols.append(c)
    for c in WT_COLS:     feats.append([min(r[c], 4.0) for r in rows]); cols.append(c)
    for c in MC_COLS:     feats.append([r[c] for r in rows]); cols.append(c)
    X = np.array(feats).T.astype(float)
    return X, cols


def apply_weights(X, cols):
    w = np.array([WEIGHTS.get(c, 1.0) for c in cols])
    return X * w


def print_cluster_summary(rows, labels):
    by_cluster = defaultdict(list)
    for r, lab in zip(rows, labels):
        by_cluster[int(lab)].append(r)
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        n = len(members)
        groups = Counter(m["unitgroup"] for m in members).most_common(3)
        folders = Counter(m["folder"] for m in members).most_common(2)
        mc_fam = Counter(m["movementclass"][:5] for m in members).most_common(3)
        avg = lambda k: np.mean([m[k] for m in members])
        sample = ", ".join(m["name"] or m["code"] for m in members[:8])
        print(f"--- cluster {cid:>2}  n={n:<3}  "
              f"groups={dict(groups)}  folders={dict(folders)}")
        print(f"    avg: hp={avg('health'):6.0f}  dps={avg('best_dps'):5.0f}  "
              f"range={avg('max_range'):5.0f}  metal={avg('metalcost'):5.0f}  "
              f"speed={avg('speed'):4.1f}  aa_frac={avg('damage_vs_air'):.2f}  "
              f"traj={avg('trajectory_high'):.2f}  v_low={avg('weapon_velocity_low'):.2f}  "
              f"rngT={avg('range_tier'):.1f}")
        print(f"    mc: {dict(mc_fam)}")
        print(f"    sample: {sample}")


def fit_models(X):
    out = {}
    for k in range(6, 16):
        try:
            m = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
            labels = m.labels_
            if len(set(labels)) > 1:
                out[("kmeans", k)] = (labels, silhouette_score(X, labels))
        except Exception: pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="/sessions/relaxed-nice-maxwell/mnt/outputs/unit_features_v2_land_combat.csv")
    p.add_argument("--out-json", default="/sessions/relaxed-nice-maxwell/mnt/outputs/unit_categories_v2.json")
    p.add_argument("--out-report", default="/sessions/relaxed-nice-maxwell/mnt/outputs/cluster_report_v2.txt")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    rows = load_rows(args.features)
    print(f"Loaded {len(rows)} units from {args.features}")
    X_raw, cols = build_X(rows)
    X = StandardScaler().fit_transform(X_raw)
    Xw = apply_weights(X, cols)
    print(f"Feature matrix: {Xw.shape}  (cols: {len(cols)})")
    weighted = [(c, WEIGHTS.get(c, 1.0)) for c in cols if WEIGHTS.get(c, 1.0) != 1.0]
    print(f"Weighted cols ({len(weighted)}): {weighted}")

    if args.sweep:
        results = fit_models(Xw)
        print("\n=== sweep ===")
        for (name, k), (_, score) in sorted(results.items()):
            print(f"  {name} k={k:>2}  silhouette={score:+.3f}")
        best = max(results.items(), key=lambda kv: kv[1][1])
        (name, k), (labels, score) = best
        print(f"\nBest: {name} k={k} silhouette={score:+.3f}")
    else:
        m = KMeans(n_clusters=args.k, n_init=20, random_state=42).fit(Xw)
        labels = m.labels_
        score = silhouette_score(Xw, labels)
        print(f"kmeans k={args.k} silhouette={score:+.3f}")
        name = "kmeans"; k = args.k

    print()
    print_cluster_summary(rows, labels)

    # Save outputs
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "version": 2, "model": {"model": name, "k": int(k), "silhouette": float(score)},
        "n_units": len(rows), "features": cols, "weights": WEIGHTS,
        "assignments": {r["code"]: int(lab) for r, lab in zip(rows, labels)},
    }, indent=2))
    print(f"\nWrote {args.out_json}")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"BAR land-combat cluster report (v2)")
        print(f"model: {name}, k={k}, silhouette={score:+.3f}")
        print(f"n_units: {len(rows)}")
        print(f"features ({len(cols)}): {cols}")
        print(f"weights: {WEIGHTS}")
        print()
        print_cluster_summary(rows, labels)
    Path(args.out_report).write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {args.out_report}")

if __name__ == "__main__": main()
