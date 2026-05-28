#!/usr/bin/env python3
"""Unsupervised clustering of BAR units.

Reads:
  bar_env/units/unit_features.csv  (from extract_unit_features.py)

For each (model, k) combo:
  - scales features (log1p on skewed positive ones, then z-score)
  - fits the model
  - reports silhouette score and prints a cluster summary

Saves:
  bar_env/units/unit_categories.json  - {model, k, assignments: {code: cluster}}
  bar_env/units/cluster_report.txt    - human-readable cluster contents

Usage:
    python scripts/cluster_units.py
    python scripts/cluster_units.py --k 12 --model kmeans --final
    python scripts/cluster_units.py --sweep   # try k=6..20, all models
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Columns that are skewed positive -> apply log1p before z-scoring.
LOG_COLS = [
    "buildtime", "metalcost", "energycost", "health",
    "best_dps", "total_dps", "max_range", "max_aoe",
    "sightdistance", "turnrate",
]
# Linear-scaled numerics (already on a reasonable scale).
LINEAR_COLS = [
    "speed", "maxacc", "maxdec", "maxslope", "maxwaterdepth",
    "min_range", "n_weapons", "techlevel",
]
# Binary flags.
FLAG_COLS = [
    "canmove", "turninplace",
    "has_aa", "has_anti_sub", "has_paralyzer",
]
# Weapon-type one-hots (counts, capped at 4 in the encoder below).
WT_COLS = [
    "wt_cannon", "wt_beamlaser", "wt_lasercannon", "wt_missilelauncher",
    "wt_starburstlauncher", "wt_torpedolauncher", "wt_emgcannon",
    "wt_aircraftbomb", "wt_flame", "wt_dgun", "wt_shield", "wt_other",
]

FEATURE_COLS = LOG_COLS + LINEAR_COLS + FLAG_COLS + WT_COLS


def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in FEATURE_COLS:
            v = r.get(k)
            try:
                r[k] = float(v) if v not in (None, "") else 0.0
            except ValueError:
                r[k] = 0.0
    return rows


def build_X(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    cols = []
    feats = []
    for c in LOG_COLS:
        feats.append([math.log1p(r[c]) for r in rows]); cols.append(c)
    for c in LINEAR_COLS:
        feats.append([r[c] for r in rows]); cols.append(c)
    for c in FLAG_COLS:
        feats.append([r[c] for r in rows]); cols.append(c)
    for c in WT_COLS:
        # Cap weapon-type counts at 4 so a single oddball with 6 of the same
        # weapon-type doesn't dominate.
        feats.append([min(r[c], 4.0) for r in rows]); cols.append(c)
    X = np.array(feats).T.astype(float)
    return X, cols


def representative_units(rows, labels, k_top=8):
    """Return the most-typical k_top units per cluster (closest to centroid in
    standardized space). Falls back to alphabetical when distances aren't
    available."""
    by_cluster = defaultdict(list)
    for r, lab in zip(rows, labels):
        by_cluster[int(lab)].append(r)
    out = {}
    for cid, members in sorted(by_cluster.items()):
        members.sort(key=lambda r: r["code"])
        out[cid] = members[:k_top]
    return out


def print_cluster_summary(rows, labels):
    by_cluster = defaultdict(list)
    for r, lab in zip(rows, labels):
        by_cluster[int(lab)].append(r)
    print()
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        n = len(members)
        groups = Counter(m["unitgroup"] for m in members).most_common(3)
        folders = Counter(m["folder"] for m in members).most_common(2)
        avg_hp = np.mean([m["health"] for m in members])
        avg_dps = np.mean([m["best_dps"] for m in members])
        avg_range = np.mean([m["max_range"] for m in members])
        avg_metal = np.mean([m["metalcost"] for m in members])
        avg_speed = np.mean([m["speed"] for m in members])
        n_air = sum(1 for m in members if m["folder"] in {"ArmAircraft","CorAircraft"})
        n_aa = sum(1 for m in members if m["has_aa"])
        n_movers = sum(1 for m in members if m["canmove"])
        sample = ", ".join(m["name"] or m["code"] for m in members[:8])
        print(f"--- cluster {cid:>2}  n={n:<3}  "
              f"groups={dict(groups)}  folders={dict(folders)}")
        print(f"    avg: hp={avg_hp:6.0f}  dps={avg_dps:5.0f}  "
              f"range={avg_range:5.0f}  metal={avg_metal:5.0f}  "
              f"speed={avg_speed:4.1f}  movers={n_movers}/{n}  "
              f"AA={n_aa}/{n}  air={n_air}/{n}")
        print(f"    sample: {sample}")


def fit_kmeans(X, k):
    m = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = m.fit_predict(X)
    return labels, m


def fit_gmm(X, k):
    m = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=42, max_iter=200)
    labels = m.fit_predict(X)
    return labels, m


def fit_agglo(X, k):
    m = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = m.fit_predict(X)
    return labels, m


def fit_hdbscan(X):
    try:
        from sklearn.cluster import HDBSCAN
        m = HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = m.fit_predict(X)
        return labels, m
    except ImportError:
        return None, None


MODELS = {"kmeans": fit_kmeans, "gmm": fit_gmm, "agglo": fit_agglo}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features", type=Path,
        default=Path(__file__).resolve().parent.parent / "bar_env" / "units" / "unit_features.csv",
    )
    p.add_argument(
        "--out-json", type=Path,
        default=Path(__file__).resolve().parent.parent / "bar_env" / "units" / "unit_categories.json",
    )
    p.add_argument(
        "--out-report", type=Path,
        default=Path(__file__).resolve().parent.parent / "bar_env" / "units" / "cluster_report.txt",
    )
    p.add_argument("--model", choices=list(MODELS.keys()) + ["hdbscan"],
                   default="kmeans")
    p.add_argument("--k", type=int, default=12,
                   help="number of clusters (ignored by hdbscan)")
    p.add_argument("--sweep", action="store_true",
                   help="grid over k=6..20 for kmeans+gmm+agglo, "
                        "print silhouettes, save the best.")
    p.add_argument("--final", action="store_true",
                   help="Write outputs JSON+report for the chosen "
                        "(--model, --k).")
    args = p.parse_args()

    rows = load_rows(args.features)
    print(f"Loaded {len(rows)} units from {args.features}")
    X_raw, cols = build_X(rows)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"Feature matrix: {X.shape}  (cols: {len(cols)})")

    if args.sweep:
        # Sweep k for each parametric model, plus one HDBSCAN run.
        print("\n=== silhouette sweep ===")
        best = (-1.0, None, None)  # (score, model_name, k)
        for name, fitter in MODELS.items():
            print(f"\n  {name}:")
            for k in range(6, 21):
                try:
                    labels, _ = fitter(X, k)
                except Exception as e:  # noqa: BLE001
                    print(f"    k={k}: failed ({e})"); continue
                if len(set(labels)) < 2:
                    print(f"    k={k}: single cluster, skipping")
                    continue
                s = silhouette_score(X, labels)
                marker = ""
                if s > best[0]:
                    best = (s, name, k); marker = "  <-- best so far"
                print(f"    k={k:>2}  silhouette={s:+.3f}{marker}")
        print("\n  hdbscan:")
        h_labels, _ = fit_hdbscan(X)
        if h_labels is not None:
            n_noise = int((h_labels == -1).sum())
            uniq = set(h_labels) - {-1}
            s = silhouette_score(X[h_labels != -1], h_labels[h_labels != -1]) \
                if len(uniq) > 1 else float("nan")
            print(f"    {len(uniq)} clusters, {n_noise} noise points, "
                  f"silhouette={s:+.3f}")
        print(f"\nBest overall: {best[1]} k={best[2]} silhouette={best[0]:+.3f}")
        # Refit best for the report
        if best[1] is not None:
            labels, _ = MODELS[best[1]](X, best[2])
            print(f"\nCluster summary for best model:")
            print_cluster_summary(rows, labels)
            chosen = {"model": best[1], "k": int(best[2]),
                      "silhouette": float(best[0])}
        else:
            chosen = None
    else:
        if args.model == "hdbscan":
            labels, _ = fit_hdbscan(X)
            if labels is None:
                sys.stderr.write("hdbscan not available in this sklearn.\n")
                return 1
            uniq = set(labels) - {-1}
            print(f"hdbscan: {len(uniq)} clusters, "
                  f"{int((labels==-1).sum())} noise points")
        else:
            labels, _ = MODELS[args.model](X, args.k)
            s = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
            print(f"{args.model} k={args.k} silhouette={s:+.3f}")
        print_cluster_summary(rows, labels)
        chosen = {"model": args.model, "k": int(args.k)}

    if args.final or args.sweep:
        # Write outputs using the labels we already have
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        assignments = {r["code"]: int(lab) for r, lab in zip(rows, labels)}
        args.out_json.write_text(json.dumps({
            "version": 1,
            "model": chosen,
            "n_units": len(rows),
            "features": cols,
            "assignments": assignments,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")

        # Save the human-readable report too
        import io
        buf = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(buf):
            print(f"BAR unit clustering report")
            print(f"model: {chosen}")
            print(f"n_units: {len(rows)}")
            print(f"features ({len(cols)}): {cols}")
            print_cluster_summary(rows, labels)
        args.out_report.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {args.out_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
