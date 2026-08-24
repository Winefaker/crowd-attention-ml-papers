"""
15_control_analysis.py
----------------------
v2 (D5): trending vs never-trending papers.

The HF sample conditions on visibility; the control sample (first-N-by-date arXiv
papers per month x category, minus any that trended) restores the background.

Questions answered:
  1. How different are citation outcomes for trending vs background papers?
  2. Does trending still separate outcomes after adjusting for author h-index and
     time (a matched comparison)? (Descriptive: trending is not randomly assigned.)
  3. Where in the background distribution does the *underrated* trending group sit?
     (If "underrated" trending papers still beat most background papers, the HF crowd
      is informative even at its weakest.)

Output: data/processed/control_results.json
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SNAPSHOT = pd.Timestamp("2026-06-11")
RES = {}


def load_control():
    meta = pd.read_csv(os.path.join(RAW, "arxiv_control.csv"), dtype={"arxiv_id_clean": str})
    s2 = pd.read_csv(os.path.join(RAW, "arxiv_control_s2.csv"), dtype={"arxiv_id_clean": str})
    c = meta.merge(s2, on="arxiv_id_clean", how="inner")
    c = c[c["ss_found"] == 1].copy()
    rel = pd.to_datetime(c["published_v1"], errors="coerce", utc=True).dt.tz_localize(None)
    c["age_months"] = (SNAPSHOT - rel).dt.days / 30.44
    c["release_month"] = rel.dt.to_period("M").astype(str)
    c = c[(c["age_months"] >= 5) & (c["age_months"] <= 40)]
    c["log_citations"] = np.log1p(pd.to_numeric(c["citation_count"], errors="coerce"))
    c["log_max_hindex"] = np.log1p(pd.to_numeric(c["max_hindex"], errors="coerce"))
    c["group"] = "background"
    return c


def load_trending():
    t = pd.read_csv(os.path.join(PROC, "papers_v2.csv"), dtype={"arxiv_id_clean": str})
    t = t[t["citation_count"].notna()]
    t = t[(t["age_months"] >= 5) & (t["age_months"] <= 40)].copy()
    t["group"] = "trending"
    return t


def main():
    c = load_control()
    t = load_trending()
    print(f"control n={len(c)}, trending n={len(t)}")

    qs = [0.25, 0.5, 0.75, 0.9]
    RES["distributions"] = {
        "control": {"n": int(len(c)),
                    "citation_quantiles": {str(q): float(c["citation_count"].quantile(q)) for q in qs},
                    "share_zero": float((c["citation_count"] == 0).mean()),
                    "share_ge100": float((c["citation_count"] >= 100).mean()),
                    "median_max_hindex": float(c["max_hindex"].median())},
        "trending": {"n": int(len(t)),
                     "citation_quantiles": {str(q): float(t["citation_count"].quantile(q)) for q in qs},
                     "share_zero": float((t["citation_count"] == 0).mean()),
                     "share_ge100": float((t["citation_count"] >= 100).mean()),
                     "median_max_hindex": float(t["max_hindex"].median())},
    }
    for g in ["control", "trending"]:
        d = RES["distributions"][g]
        print(f"  {g:9s} median={d['citation_quantiles']['0.5']:.0f} "
              f"p90={d['citation_quantiles']['0.9']:.0f} zero={d['share_zero']:.1%} "
              f">=100 cites={d['share_ge100']:.1%} med h-index={d['median_max_hindex']:.0f}")

    # matched comparison on h-index + release month
    both = pd.concat([
        t[["log_citations", "log_max_hindex", "release_month", "age_months"]].assign(treat=1),
        c[["log_citations", "log_max_hindex", "release_month", "age_months"]].assign(treat=0),
    ], ignore_index=True).dropna()
    both["month_num"] = pd.to_datetime(both["release_month"]).astype("int64") / 1e18
    X = StandardScaler().fit_transform(both[["log_max_hindex", "month_num"]])
    ps = LogisticRegression(max_iter=1000).fit(X, both["treat"]).predict_proba(X)[:, 1]
    both["ps"] = np.clip(ps, 0.02, 0.98)
    tr = both[both.treat == 1].reset_index(drop=True)
    co = both[both.treat == 0].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=1).fit(co[["ps"]].values)
    _, idx = nn.kneighbors(tr[["ps"]].values)
    gap_matched = float((tr["log_citations"].values -
                         co.iloc[idx.flatten()]["log_citations"].values).mean())
    gap_naive = float(tr["log_citations"].mean() - co["log_citations"].mean())
    RES["trending_premium"] = {
        "naive_log_gap": gap_naive, "matched_log_gap": gap_matched,
        "naive_ratio": float(np.expm1(gap_naive) + 1), "matched_ratio": float(np.expm1(gap_matched) + 1),
        "note": "descriptive premium; trending is selected, not randomized",
    }
    print(f"  trending premium: naive {gap_naive:.2f} log-pts, matched(h-index,month) {gap_matched:.2f}"
          f" (~x{np.exp(gap_matched):.1f} citations)")

    # where do underrated trending papers sit vs background?
    sc = pd.read_csv(os.path.join(PROC, "papers_scored_v2.csv"), dtype={"arxiv_id_clean": str})
    und = sc[sc["underrated"] == 1]["citation_count"]
    bg = c["citation_count"]
    pct = float((bg.values[None, :] < und.values[:, None]).mean())
    RES["underrated_vs_background"] = {
        "mean_percentile_of_underrated_in_background": pct,
        "underrated_median": float(und.median()), "background_median": float(bg.median()),
    }
    print(f"  'underrated' trending papers sit at the {pct:.0%} percentile of the background")

    with open(os.path.join(PROC, "control_results.json"), "w") as f:
        json.dump(RES, f, indent=2)
    print("Saved control_results.json")


if __name__ == "__main__":
    main()
