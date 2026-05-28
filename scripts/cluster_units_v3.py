"""V3 clustering: per-tier clustering with rebalanced weights.

Changes from v2:
  - has_aa weight: 2.0 -> 1.0 (let damage_vs_air carry the AA signal at 2.0x)
  - speed weight: 1.0 -> 1.5
  - HARD CONSTRAINT: each tier is clustered independently, no mixed-tier clusters.

Final cluster IDs are stamped tier*100 + sub_cluster (so T1->100..199, T2->200..299, T3->300..399).
"""
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
    # role-defining
    "max_range": 2.0, "best_dps": 2.0, "has_paralyzer": 2.0,
    "range_per_speed": 2.0, "dps_per_metal": 2.0, "damage_vs_air": 2.0,
    "trajectory_high": 2.0, "range_tier": 3.0, "weapon_velocity_low": 2.0,
    # has_aa demoted (binary, noisy; damage_vs_air is the accurate signal)
    "has_aa": 1.0,
    # mobility -- the user cares about response-time
    "speed": 1.5,
    # chassis hints
    "mc_BOT": 1.5, "mc_ABOT": 1.5, "mc_TANK": 1.5,
    "mc_ATANK": 1.5, "mc_HOVER": 1.5, "mc_COMMANDER": 1.5, "mc_OTHER": 1.5,
}
# Drop techlevel from features since each tier is clustered independently.
# (Including it would just be a constant inside each tier-subset.)
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
    for c in LINEAR_COLS:
        if c == "techlevel": continue   # exclude, see comment above
        feats.append([r[c] for r in rows]); cols.append(c)
    for c in FLAG_COLS:   feats.append([r[c] for r in rows]); cols.append(c)
    for c in WT_COLS:     feats.append([min(r[c], 4.0) for r in rows]); cols.append(c)
    for c in MC_COLS:     feats.append([r[c] for r in rows]); cols.append(c)
    X = np.array(feats).T.astype(float)
    return X, cols


def apply_weights(X, cols):
    w = np.array([WEIGHTS.get(c, 1.0) for c in cols])
    return X * w


def pick_best_k(Xw, k_range):
    best = (None, -1.0)
    scores = []
    for k in k_range:
        if k >= len(Xw): continue
        m = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xw)
        labels = m.labels_
        if len(set(labels)) < 2: continue
        s = silhouette_score(Xw, labels)
        scores.append((k, s))
        if s > best[1]: best = (k, s)
    return best, scores


def cluster_tier(rows, tier, k_range):
    sub = [r for r in rows if int(float(r["techlevel"])) == tier]
    if not sub:
        return [], (None, None), []
    X_raw, cols = build_X(sub)
    X = StandardScaler().fit_transform(X_raw)
    Xw = apply_weights(X, cols)
    (best_k, best_s), scores = pick_best_k(Xw, k_range)
    if best_k is None:
        return sub, (None, None), scores
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(Xw)
    return sub, (best_k, best_s, labels), scores


def print_summary(rows, labels):
    by_c = defaultdict(list)
    for r, lab in zip(rows, labels):
        by_c[int(lab)].append(r)
    for cid in sorted(by_c):
        m = by_c[cid]
        n = len(m)
        avg = lambda k: np.mean([float(x[k]) for x in m])
        mc_fam = Counter(x["movementclass"][:5] for x in m).most_common(3)
        groups = Counter(x["unitgroup"] for x in m).most_common(3)
        sample = ", ".join(x["name"] or x["code"] for x in m[:10])
        print(f"--- c{cid:>3}  n={n:<3} tier={m[0]['techlevel']}  "
              f"groups={dict(groups)}")
        print(f"    avg: hp={avg('health'):6.0f}  dps={avg('best_dps'):5.0f}  "
              f"range={avg('max_range'):5.0f}  metal={avg('metalcost'):5.0f}  "
              f"speed={avg('speed'):5.1f}  aa_frac={avg('damage_vs_air'):.2f}  "
              f"rngT={avg('range_tier'):.1f}  traj={avg('trajectory_high'):.2f}")
        print(f"    mc: {dict(mc_fam)}")
        print(f"    sample: {sample}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="/sessions/relaxed-nice-maxwell/mnt/outputs/unit_features_v2_land_combat.csv")
    p.add_argument("--out-json", default="/sessions/relaxed-nice-maxwell/mnt/outputs/unit_categories_v3.json")
    p.add_argument("--out-report", default="/sessions/relaxed-nice-maxwell/mnt/outputs/cluster_report_v3.txt")
    p.add_argument("--k-t1", type=int, default=0, help="0 = auto (silhouette)")
    p.add_argument("--k-t2", type=int, default=0)
    p.add_argument("--k-t3", type=int, default=0)
    args = p.parse_args()

    rows = load_rows(args.features)
    print(f"Loaded {len(rows)} units")
    tier_counts = Counter(int(float(r['techlevel'])) for r in rows)
    print(f"Tiers: {dict(tier_counts)}")

    all_rows = []
    all_labels = []
    summary_by_tier = {}
    for tier, k_default in [(1, args.k_t1), (2, args.k_t2), (3, args.k_t3)]:
        # pick a reasonable k-range per tier (sqrt of n give-or-take)
        n = tier_counts.get(tier, 0)
        if n == 0: continue
        if k_default and k_default > 0:
            k_range = [k_default]
        else:
            lo = max(3, int(n**0.5) - 1)
            hi = max(lo+1, min(n-1, int(n**0.5) + 3))
            k_range = list(range(lo, hi+1))
        sub, (best_k, best_s, labels), scores = cluster_tier(rows, tier, k_range)
        print(f"\n=== T{tier} ({len(sub)} units) ===")
        print(f"  silhouettes: {[f'k={k}:{s:+.3f}' for k,s in scores]}")
        print(f"  picked k={best_k}, silhouette={best_s:+.3f}")
        # offset cluster IDs so they're unique across tiers
        offset = tier * 100
        relabeled = labels + offset
        all_rows.extend(sub)
        all_labels.extend(relabeled.tolist())
        summary_by_tier[tier] = {"k": int(best_k), "silhouette": float(best_s)}

    print()
    print_summary(all_rows, all_labels)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "version": 3,
        "model": "kmeans (per-tier)",
        "per_tier": summary_by_tier,
        "weights": WEIGHTS,
        "n_units": len(all_rows),
        "assignments": {r["code"]: int(lab) for r, lab in zip(all_rows, all_labels)},
    }, indent=2))
    print(f"\nWrote {args.out_json}")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"BAR land-combat clustering v3 (per-tier)")
        print(f"weights: {WEIGHTS}")
        print(f"per-tier results: {summary_by_tier}")
        print(f"n_units: {len(all_rows)}")
        print()
        print_summary(all_rows, all_labels)
    Path(args.out_report).write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {args.out_report}")


if __name__ == "__main__": main()
