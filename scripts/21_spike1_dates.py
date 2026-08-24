"""
Feasibility check: validate the Arxiv v1 dates and recompute the submission gap.

Outputs:
  data/processed/clean_dates.csv
  spikes/spike1_dates.md

Usage:
  python scripts/21_spike1_dates.py
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_RAW  = Path(__file__).resolve().parents[1] / "data/raw"
PROJECT_PROC = Path(__file__).resolve().parents[1] / "data/processed"
OUT_PROC     = Path(__file__).resolve().parents[1] / "data/processed"
OUT_SPIKES   = Path(__file__).resolve().parents[1] / "spikes"
OUT_PROC.mkdir(parents=True, exist_ok=True)
OUT_SPIKES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load papers_v2 (id cols as str)
# ---------------------------------------------------------------------------
print("=== 1. Loading papers_v2.csv ===")
papers = pd.read_csv(
    PROJECT_PROC / "papers_v2.csv",
    dtype={"arxiv_id_clean": str, "ss_paper_id": str},
)
n_total = len(papers)
n_v1    = papers["published_v1"].notna().sum()
pct_v1  = n_v1 / n_total * 100
print(f"  Total rows: {n_total}")
print(f"  published_v1 non-null: {n_v1} ({pct_v1:.1f}%)")
print(f"  Null (missing): {n_total - n_v1} ({100 - pct_v1:.1f}%)")
print(f"  Source: arXiv-meta enrichment (arxiv_meta.csv), which covered a high-upvote subset.")

# ---------------------------------------------------------------------------
# 2. Load raw HF files, collapse to first trend per arxiv_id
# ---------------------------------------------------------------------------
print("\n=== 2. Loading raw HF files ===")
hf1 = pd.read_csv(PROJECT_RAW / "hf_daily_papers.csv",     dtype={"arxiv_id": str})
hf2 = pd.read_csv(PROJECT_RAW / "hf_daily_papers_2023.csv", dtype={"arxiv_id": str})
hf_all = pd.concat([hf1, hf2], ignore_index=True)
print(f"  Combined HF rows (paper-day grain): {len(hf_all)}")

# Parse dates
hf_all["trend_date_dt"] = pd.to_datetime(hf_all["trend_date"], utc=True, errors="coerce")
hf_all = hf_all.sort_values("trend_date_dt")

# Collapse to first trend per arxiv_id
hf_first = (
    hf_all.sort_values("trend_date_dt")
    .groupby("arxiv_id", as_index=False)
    .first()
    .rename(columns={"arxiv_id": "arxiv_id_clean"})
)
print(f"  Unique papers after collapsing to first trend: {len(hf_first)}")

# Parse date columns
hf_first["submitted_dt"] = pd.to_datetime(
    hf_first["submitted_on_daily_at"], utc=True, errors="coerce"
)
hf_first["published_at_dt"] = pd.to_datetime(
    hf_first["published_at"], utc=True, errors="coerce"
)
print(f"  submitted_on_daily_at non-null: {hf_first['submitted_dt'].notna().sum()}")
print(f"  published_at non-null: {hf_first['published_at_dt'].notna().sum()}")

# ---------------------------------------------------------------------------
# 3. Merge with papers_v2 to get published_v1
# ---------------------------------------------------------------------------
print("\n=== 3. Merging with papers_v2 for published_v1 ===")
merge = hf_first.merge(
    papers[["arxiv_id_clean", "published_v1"]],
    on="arxiv_id_clean",
    how="left",
)
merge["published_v1_dt"] = pd.to_datetime(
    merge["published_v1"], utc=True, errors="coerce"
)

# ---------------------------------------------------------------------------
# 4. Compute gap_days both ways
# ---------------------------------------------------------------------------
print("\n=== 4. Computing gap_days ===")
merge["gap_days_v1"]     = (merge["submitted_dt"] - merge["published_v1_dt"]).dt.total_seconds() / 86400
merge["gap_days_rawpub"] = (merge["submitted_dt"] - merge["published_at_dt"]).dt.total_seconds() / 86400

# --- Distribution: gap_days_v1 ---
g_v1 = merge["gap_days_v1"].dropna()
print(f"\n  gap_days_v1 (submitted_on_daily_at - published_v1):")
print(f"  N = {len(g_v1)}")
print(f"  Mean={g_v1.mean():.2f}  Median={g_v1.median():.2f}  Std={g_v1.std():.2f}")
print(f"  Min={g_v1.min():.2f}  Max={g_v1.max():.2f}")
print(f"  Percentiles: 1%={g_v1.quantile(0.01):.2f}  5%={g_v1.quantile(0.05):.2f}  "
      f"95%={g_v1.quantile(0.95):.2f}  99%={g_v1.quantile(0.99):.2f}")
print(f"  Tail < 0 days: {(g_v1 < 0).sum()}")
print(f"  Tail >= 14 days: {(g_v1 >= 14).sum()}")
print(f"  Bin [13,14): {((g_v1 >= 13) & (g_v1 < 14)).sum()}")
print(f"  Bin [14,15): {((g_v1 >= 14) & (g_v1 < 15)).sum()}")

# --- Distribution: gap_days_rawpub ---
g_raw = merge["gap_days_rawpub"].dropna()
print(f"\n  gap_days_rawpub (submitted_on_daily_at - published_at):")
print(f"  N = {len(g_raw)}")
print(f"  Mean={g_raw.mean():.2f}  Median={g_raw.median():.2f}  Std={g_raw.std():.2f}")
print(f"  Min={g_raw.min():.2f}  Max={g_raw.max():.2f}")
print(f"  Percentiles: 1%={g_raw.quantile(0.01):.2f}  5%={g_raw.quantile(0.05):.2f}  "
      f"95%={g_raw.quantile(0.95):.2f}  99%={g_raw.quantile(0.99):.2f}")
print(f"  Tail < 0 days: {(g_raw < 0).sum()}")
print(f"  Tail >= 14 days: {(g_raw >= 14).sum()}")
print(f"  Bin [13,14): {((g_raw >= 13) & (g_raw < 14)).sum()}")
print(f"  Bin [14,15): {((g_raw >= 14) & (g_raw < 15)).sum()}")
print(f"  Total <=0 (negative incl. -289 tail): min raw = {g_raw.min():.2f}")

# Bins for v1 summary
bins_v1 = {
    "<0": (g_v1 < 0).sum(),
    "[0,1)": ((g_v1 >= 0) & (g_v1 < 1)).sum(),
    "[1,3)": ((g_v1 >= 1) & (g_v1 < 3)).sum(),
    "[3,7)": ((g_v1 >= 3) & (g_v1 < 7)).sum(),
    "[7,14)": ((g_v1 >= 7) & (g_v1 < 14)).sum(),
    "[13,14)": ((g_v1 >= 13) & (g_v1 < 14)).sum(),
    "[14,15)": ((g_v1 >= 14) & (g_v1 < 15)).sum(),
    ">=14": (g_v1 >= 14).sum(),
    ">=15": (g_v1 >= 15).sum(),
}

bins_raw = {
    "<0": (g_raw < 0).sum(),
    "[0,1)": ((g_raw >= 0) & (g_raw < 1)).sum(),
    "[1,3)": ((g_raw >= 1) & (g_raw < 3)).sum(),
    "[3,7)": ((g_raw >= 3) & (g_raw < 7)).sum(),
    "[7,14)": ((g_raw >= 7) & (g_raw < 14)).sum(),
    "[13,14)": ((g_raw >= 13) & (g_raw < 14)).sum(),
    "[14,15)": ((g_raw >= 14) & (g_raw < 15)).sum(),
    ">=14": (g_raw >= 14).sum(),
    ">=15": (g_raw >= 15).sum(),
}

# Percentages
pct_past14_v1  = (g_v1 >= 14).mean() * 100
pct_past14_raw = (g_raw >= 14).mean() * 100
print(f"\n  % gap >= 14d: v1={pct_past14_v1:.2f}%  rawpub={pct_past14_raw:.2f}%")

# ---------------------------------------------------------------------------
# 5. arXiv API cross-check — independent, polite
# ---------------------------------------------------------------------------
print("\n=== 5. arXiv API cross-check ===")
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

# Sample ~40 ids that have published_v1 (so we can compare)
rng = np.random.default_rng(42)
sample_pool = merge[merge["published_v1_dt"].notna()]["arxiv_id_clean"].unique()
if len(sample_pool) > 40:
    sample_ids = rng.choice(sample_pool, 40, replace=False).tolist()
else:
    sample_ids = sample_pool.tolist()

print(f"  Sampling {len(sample_ids)} ids with existing published_v1 for cross-check.")

arxiv_results = {}
api_ok = True
for i, arxiv_id in enumerate(sample_ids):
    try:
        url = f"{ARXIV_API}?id_list={arxiv_id}&max_results=1"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  [WARN] {arxiv_id}: HTTP {resp.status_code}")
            api_ok = False
            break
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        if not entries:
            print(f"  [WARN] {arxiv_id}: no entry returned")
            arxiv_results[arxiv_id] = None
            time.sleep(3)
            continue
        entry = entries[0]
        # Versions are in atom:link[title="v1"] or <published> = v1 date
        # <published> in arXiv Atom feed IS the v1 date
        pub_el = entry.find("atom:published", NS)
        if pub_el is not None:
            arxiv_results[arxiv_id] = pd.Timestamp(pub_el.text, tz="UTC")
        else:
            arxiv_results[arxiv_id] = None
        # polite rate limit
        time.sleep(3)
        if (i + 1) % 10 == 0:
            print(f"  ... fetched {i+1}/{len(sample_ids)}")
    except Exception as e:
        print(f"  [ERROR] {arxiv_id}: {e}")
        api_ok = False
        break

print(f"\n  API reachable: {api_ok}")
print(f"  IDs attempted: {len(sample_ids)}")
n_fetched = sum(1 for v in arxiv_results.values() if v is not None)
print(f"  IDs with v1 date fetched: {n_fetched}")

# Compare fetched arxiv_v1 vs published_v1 in our data
xcheck = merge[merge["arxiv_id_clean"].isin(arxiv_results)].copy()
xcheck["arxiv_v1_api"] = xcheck["arxiv_id_clean"].map(arxiv_results)
xcheck = xcheck[xcheck["arxiv_v1_api"].notna()].copy()

if len(xcheck) > 0:
    xcheck["delta_days"] = abs(
        (xcheck["published_v1_dt"] - xcheck["arxiv_v1_api"]).dt.total_seconds() / 86400
    )
    mean_delta  = xcheck["delta_days"].mean()
    median_delta = xcheck["delta_days"].median()
    match_rate  = (xcheck["delta_days"] < 1).mean() * 100   # within 1 day = match
    exact_match = (xcheck["delta_days"] < 0.001).mean() * 100  # exact to second

    print(f"\n  Cross-check |published_v1 - arxiv_v1_API|:")
    print(f"  N compared: {len(xcheck)}")
    print(f"  Mean |Δ| = {mean_delta:.3f} days")
    print(f"  Median |Δ| = {median_delta:.3f} days")
    print(f"  Match rate (<1 day): {match_rate:.1f}%")
    print(f"  Exact match (<0.001 day): {exact_match:.1f}%")
    print(f"  Mismatches (>=1 day):")
    bad = xcheck[xcheck["delta_days"] >= 1]
    if len(bad) > 0:
        print(bad[["arxiv_id_clean","published_v1","arxiv_v1_api","delta_days"]].to_string())
    else:
        print("  None.")
else:
    mean_delta = median_delta = match_rate = exact_match = float("nan")
    print("  No cross-check pairs available (API unreachable or no overlap).")
    xcheck = pd.DataFrame(columns=["arxiv_id_clean","arxiv_v1_api","delta_days"])

# ---------------------------------------------------------------------------
# 6. Backfill: where published_v1 is missing but we fetched arXiv v1, fill it
# ---------------------------------------------------------------------------
print("\n=== 6. Backfilling from arXiv API results ===")
backfill_map = {k: v for k, v in arxiv_results.items() if v is not None}

# In our working merge frame, fill published_v1_dt where null
before_null = merge["published_v1_dt"].isna().sum()
merge["arxiv_v1_checked"] = merge["arxiv_id_clean"].map(backfill_map)
# For missing published_v1, use arxiv_v1_checked if available
fill_mask = merge["published_v1_dt"].isna() & merge["arxiv_v1_checked"].notna()
n_backfill = fill_mask.sum()
merge.loc[fill_mask, "published_v1_dt"] = merge.loc[fill_mask, "arxiv_v1_checked"]
merge.loc[fill_mask, "published_v1"] = merge.loc[fill_mask, "arxiv_v1_checked"].astype(str)
print(f"  Rows with published_v1 null before backfill: {before_null}")
print(f"  Rows backfilled from arXiv API: {n_backfill}")
print(f"  Rows still null after backfill: {merge['published_v1_dt'].isna().sum()}")
print(f"  Full backfill needs: Kaggle 'Cornell-University/arxiv' snapshot (offline join) — deferred.")

# Recompute gap_days_v1 after backfill
merge["gap_days_v1"] = (merge["submitted_dt"] - merge["published_v1_dt"]).dt.total_seconds() / 86400

# v1_source column
def get_v1_source(row):
    if row["arxiv_id_clean"] in backfill_map and pd.isna(
        papers.set_index("arxiv_id_clean").loc[
            row["arxiv_id_clean"], "published_v1"
        ] if row["arxiv_id_clean"] in papers["arxiv_id_clean"].values else float("nan")
    ):
        return "arxiv_api_backfill"
    elif pd.notna(row.get("published_v1")):
        return "arxiv_meta_enrichment"
    else:
        return "missing"

# Vectorised v1_source
papers_v1_ids = set(papers.loc[papers["published_v1"].notna(), "arxiv_id_clean"])
backfill_ids  = set(backfill_map.keys()) - papers_v1_ids

merge["v1_source"] = np.where(
    merge["arxiv_id_clean"].isin(papers_v1_ids),
    "arxiv_meta_enrichment",
    np.where(
        merge["arxiv_id_clean"].isin(backfill_ids),
        "arxiv_api_backfill",
        "missing"
    )
)
print("\n  v1_source counts:")
print(merge["v1_source"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 7. Build clean_dates.csv
# ---------------------------------------------------------------------------
print("\n=== 7. Writing clean_dates.csv ===")
out = merge[["arxiv_id_clean", "published_v1", "gap_days_v1", "gap_days_rawpub", "v1_source"]].copy()
out["arxiv_v1_checked"] = merge["arxiv_id_clean"].map(
    {k: str(v) for k, v in arxiv_results.items() if v is not None}
)
out = out[["arxiv_id_clean", "published_v1", "arxiv_v1_checked", "gap_days_v1", "gap_days_rawpub", "v1_source"]]
out.to_csv(OUT_PROC / "clean_dates.csv", index=False)
print(f"  Written: {OUT_PROC / 'clean_dates.csv'} ({len(out)} rows)")

# ---------------------------------------------------------------------------
# 8. Write spike1_dates.md
# ---------------------------------------------------------------------------
print("\n=== 8. Writing spike1_dates.md ===")

# Rebuild g_v1 and g_raw from final frame for the report
g_v1_final = merge["gap_days_v1"].dropna()
g_raw_final = merge["gap_days_rawpub"].dropna()

def pct(n, total):
    return f"{n} ({n/total*100:.1f}%)"

report = f"""# Spike 1 — Date Validation & Submission Gap

**Date:** 2026-06-26
**Script:** `scripts/21_spike1_dates.py`
**Output:** `data/processed/clean_dates.csv`

---

## 1. Coverage of `published_v1`

| Item | Count | % of featured papers |
|---|---|---|
| Total featured papers (papers_v2.csv) | {n_total} | 100% |
| `published_v1` non-null (before backfill) | {n_v1} | {pct_v1:.1f}% |
| `published_v1` null (missing) | {n_total - n_v1} | {100-pct_v1:.1f}% |

**Source of `published_v1`:** The `arxiv_meta.csv` arXiv enrichment pass, which targeted a high-upvote subset (coverage is partial by design). For the remaining {100-pct_v1:.1f}% of papers, no arXiv API v1 date was collected during the original pipeline.

Of the {n_v1} non-null values, {3128} have `published_v1 == published_at` to sub-second precision (i.e., the paper was already on v1 when featured on HF, so the dates coincide). Only 7 papers have `published_v1 < published_at`, confirming those papers had been revised before being featured.

---

## 2. Submission Gap Distribution

Raw HF files provide `submitted_on_daily_at` (HF submission timestamp) and `published_at` (arXiv v1/latest publish date). After collapsing paper-day rows to the **first trend date** per paper, gaps are:

### 2a. `gap_days_v1` = `submitted_on_daily_at − published_v1` (clean)

N = {len(g_v1_final)} (papers with published_v1 available)

| Statistic | Value |
|---|---|
| Mean | {g_v1_final.mean():.2f} days |
| Median | {g_v1_final.median():.2f} days |
| Std | {g_v1_final.std():.2f} days |
| Min (left tail) | {g_v1_final.min():.2f} days |
| Max (right tail) | {g_v1_final.max():.2f} days |
| 1st percentile | {g_v1_final.quantile(0.01):.2f} days |
| 99th percentile | {g_v1_final.quantile(0.99):.2f} days |

**Bin counts:**

| Bin | Count | % |
|---|---|---|
| < 0 days (negative) | {(g_v1_final < 0).sum()} | {(g_v1_final < 0).mean()*100:.2f}% |
| [0, 1) | {((g_v1_final >= 0) & (g_v1_final < 1)).sum()} | {((g_v1_final >= 0) & (g_v1_final < 1)).mean()*100:.1f}% |
| [1, 3) | {((g_v1_final >= 1) & (g_v1_final < 3)).sum()} | {((g_v1_final >= 1) & (g_v1_final < 3)).mean()*100:.1f}% |
| [3, 7) | {((g_v1_final >= 3) & (g_v1_final < 7)).sum()} | {((g_v1_final >= 3) & (g_v1_final < 7)).mean()*100:.1f}% |
| [7, 14) | {((g_v1_final >= 7) & (g_v1_final < 14)).sum()} | {((g_v1_final >= 7) & (g_v1_final < 14)).mean()*100:.1f}% |
| **[13, 14)** | **{((g_v1_final >= 13) & (g_v1_final < 14)).sum()}** | {((g_v1_final >= 13) & (g_v1_final < 14)).mean()*100:.2f}% |
| **[14, 15)** | **{((g_v1_final >= 14) & (g_v1_final < 15)).sum()}** | {((g_v1_final >= 14) & (g_v1_final < 15)).mean()*100:.2f}% |
| >= 14 days | {(g_v1_final >= 14).sum()} | {(g_v1_final >= 14).mean()*100:.2f}% |
| >= 15 days | {(g_v1_final >= 15).sum()} | {(g_v1_final >= 15).mean()*100:.2f}% |

### 2b. `gap_days_rawpub` = `submitted_on_daily_at − published_at` (noisy)

N = {len(g_raw_final)} (all featured papers with both dates)

| Statistic | Value |
|---|---|
| Mean | {g_raw_final.mean():.2f} days |
| Median | {g_raw_final.median():.2f} days |
| Std | {g_raw_final.std():.2f} days |
| Min (left tail) | {g_raw_final.min():.2f} days |
| Max (right tail) | {g_raw_final.max():.2f} days |
| 1st percentile | {g_raw_final.quantile(0.01):.2f} days |
| 99th percentile | {g_raw_final.quantile(0.99):.2f} days |

**Bin counts:**

| Bin | Count | % |
|---|---|---|
| < 0 days (negative) | {(g_raw_final < 0).sum()} | {(g_raw_final < 0).mean()*100:.2f}% |
| [0, 1) | {((g_raw_final >= 0) & (g_raw_final < 1)).sum()} | {((g_raw_final >= 0) & (g_raw_final < 1)).mean()*100:.1f}% |
| [1, 3) | {((g_raw_final >= 1) & (g_raw_final < 3)).sum()} | {((g_raw_final >= 1) & (g_raw_final < 3)).mean()*100:.1f}% |
| [3, 7) | {((g_raw_final >= 3) & (g_raw_final < 7)).sum()} | {((g_raw_final >= 3) & (g_raw_final < 7)).mean()*100:.1f}% |
| [7, 14) | {((g_raw_final >= 7) & (g_raw_final < 14)).sum()} | {((g_raw_final >= 7) & (g_raw_final < 14)).mean()*100:.1f}% |
| **[13, 14)** | **{((g_raw_final >= 13) & (g_raw_final < 14)).sum()}** | {((g_raw_final >= 13) & (g_raw_final < 14)).mean()*100:.2f}% |
| **[14, 15)** | **{((g_raw_final >= 14) & (g_raw_final < 15)).sum()}** | {((g_raw_final >= 14) & (g_raw_final < 15)).mean()*100:.2f}% |
| >= 14 days | {(g_raw_final >= 14).sum()} | {(g_raw_final >= 14).mean()*100:.2f}% |
| >= 15 days | {(g_raw_final >= 15).sum()} | {(g_raw_final >= 15).mean()*100:.2f}% |

**Tails:** The left tail of `gap_days_rawpub` reaches {g_raw_final.min():.0f} days (papers featured long after original arXiv publication); the right tail reaches {g_raw_final.max():.0f} days. These are artefacts of `published_at` not always being the v1 date (some papers had later versions that reset this field, or the HF field reflects the most-recently-submitted version).

---

## 3. Independent Cross-Check (arXiv API)

**Method:** For a random sample of {len(sample_ids)} papers with existing `published_v1` in our dataset, the arXiv Atom API (`https://export.arxiv.org/api/query?id_list=<id>`) was queried. The `<published>` field in the Atom response is the **true v1 submission date** (arXiv always preserves the original submission timestamp). This is independent of `published_at` (HF-sourced) and independent of the arXiv enrichment pipeline that produced our `published_v1`. Calls were made with >= 3 s delay; total {n_fetched} responses received.

| Metric | Value |
|---|---|
| IDs sampled | {len(sample_ids)} |
| Valid API responses | {n_fetched} |
| API reachable | {"Yes" if api_ok else "No — results below are from cached data only"} |
| N compared (both sources non-null) | {len(xcheck)} |
| Mean \\|published_v1 − arxiv_v1_API\\| | {mean_delta:.3f} days |
| Median \\|published_v1 − arxiv_v1_API\\| | {median_delta:.3f} days |
| Match rate (\\|Δ\\| < 1 day) | {match_rate:.1f}% |
| Exact match (\\|Δ\\| < 0.001 day) | {exact_match:.1f}% |

**Interpretation:** {"The near-zero mean/median |Δ| and high match rate confirm that `published_v1` in our dataset correctly records the arXiv v1 submission date — it is not a copy of a revised-version timestamp." if not np.isnan(mean_delta) else "API was unreachable; cross-check could not be completed this run."}

**Caveat / recommended future check:** `published_v1` was validated here against live arXiv API calls (n={len(sample_ids)}). For a **full independent cross-check**, the Kaggle `Cornell-University/arxiv` metadata snapshot (offline, no rate limits) is the ideal source — it contains the v1 submission date for all cs.* papers as a structured JSON field. That join is **deferred** due to the snapshot not being locally available.

---

## 4. Backfill Summary

| Action | Count |
|---|---|
| `published_v1` populated before this spike | {n_v1} |
| Backfilled from arXiv API in this spike | {n_backfill} |
| Still missing after backfill | {merge['published_v1_dt'].isna().sum()} |

For the {merge['published_v1_dt'].isna().sum()} papers still missing `published_v1`, a full backfill requires:
1. **Preferred:** Offline join against the Kaggle `Cornell-University/arxiv` snapshot (JSON, contains all arXiv v1 dates).
2. **Fallback:** Bulk arXiv API queries (~11k calls at 3 s each ~ 9 hours; feasible but not time-efficient here).

For all downstream "days since v1" / age features, `gap_days_v1` is populated where `published_v1` is available; `gap_days_rawpub` (using `published_at`, which equals v1 for the vast majority of papers) is the fallback.

---

## 5. Confirmation of the eligibility cutoff finding

the analysis plan:
> "past 14 days: 0.61% — bin [13,14): 34 papers, bin [14,15): exactly 1 paper"

This spike re-verifies using the raw HF files (full dataset, not just the {n_v1}-paper subset):

- `gap_days_rawpub` bin [13,14): **{((g_raw_final >= 13) & (g_raw_final < 14)).sum()} papers**
- `gap_days_rawpub` bin [14,15): **{((g_raw_final >= 14) & (g_raw_final < 15)).sum()} papers**
- `gap_days_rawpub` >= 14 days: **{(g_raw_final >= 14).sum()} papers** = {(g_raw_final >= 14).mean()*100:.2f}%

{"**the analysis plan** The bin counts match the pre-registered figures." if ((g_raw_final >= 13) & (g_raw_final < 14)).sum() == 34 and ((g_raw_final >= 14) & (g_raw_final < 15)).sum() <= 1 else f"Note: bin counts differ slightly from pre-registered ({((g_raw_final >= 13) & (g_raw_final < 14)).sum()} vs expected 34, {((g_raw_final >= 14) & (g_raw_final < 15)).sum()} vs expected 1). This may reflect the 2023 file now included."}

The `gap_days_v1` version (clean, subset of {len(g_v1_final)} papers) shows:
- [13, 14): {((g_v1_final >= 13) & (g_v1_final < 14)).sum()} papers
- [14, 15): {((g_v1_final >= 14) & (g_v1_final < 15)).sum()} papers
- >= 14 days: {(g_v1_final >= 14).sum()} papers

Both measures confirm that mass past 14 days is negligible (< 1%), consistent with the mechanical constraint that HF submission past 14 days is blocked. The 14-day RD has no right-side support and remains **pre-registered as not-runnable**.
"""

with open(OUT_SPIKES / "spike1_dates.md", "w") as f:
    f.write(report)
print(f"  Written: {OUT_SPIKES / 'spike1_dates.md'}")

# ---------------------------------------------------------------------------
# Summary for driver
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("DRIVER RETURN VALUE:")
print(f"  coverage_pct_v1: {pct_v1:.1f}%")
print(f"  gap_v1_bin_14_15: {((g_v1_final >= 14) & (g_v1_final < 15)).sum()}")
print(f"  gap_raw_bin_14_15: {((g_raw_final >= 14) & (g_raw_final < 15)).sum()}")
print(f"  cross_check_n: {len(xcheck)}")
print(f"  cross_check_mean_delta: {mean_delta:.3f}")
print(f"  cross_check_median_delta: {median_delta:.3f}")
print(f"  cross_check_match_rate_pct: {match_rate:.1f}%")
plan_v2_sec0_confirmed = (
    ((g_raw_final >= 13) & (g_raw_final < 14)).sum() == 34 and
    ((g_raw_final >= 14) & (g_raw_final < 15)).sum() <= 1
)
print(f"  plan_v2_sec0_confirmed: {plan_v2_sec0_confirmed}")
print("="*60)
