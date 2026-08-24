"""
23_crowding_cohort.py
Step C2 — Build same-day CROWDING instruments from the HF (paper, day) panel.

Instruments are leave-one-out; cohort day = first_trend_date (per the analysis plan).

Outputs
-------
data/processed/crowding.csv
    arxiv_id_clean, cohort_day, cohort_size, Z1_logcompet, Z2_count,
    Z3_blockbuster, Z1p_othersub, ego_upvotes

WARNING: arxiv_id_clean MUST be read with dtype=str (e.g. pd.read_csv(...,
dtype={'arxiv_id_clean': str})). Reading as float64 silently corrupts IDs
such as 2412.01230 → 2412.0123.

Sanity report is printed to stdout.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_2024 = str(Path(__file__).resolve().parents[2] / "data/raw/hf_daily_papers.csv")
RAW_2023 = str(Path(__file__).resolve().parents[2] / "data/raw/hf_daily_papers_2023.csv")
PAPERS_V2 = str(Path(__file__).resolve().parents[2] / "data/processed/papers_v2.csv")
OUT_CSV = str(Path(__file__).resolve().parents[2] / "data/processed/crowding.csv")


def strip_version(arxiv_id: str) -> str:
    """Remove arXiv version suffix (e.g. 2301.12345v2 -> 2301.12345)."""
    return re.sub(r"v\d+$", "", str(arxiv_id).strip())


# ---------------------------------------------------------------------------
# 1. Load and union raw panel files
# ---------------------------------------------------------------------------
print("Loading raw panel files...")

df_2024 = pd.read_csv(
    RAW_2024,
    usecols=["arxiv_id", "trend_date", "upvotes"],
    dtype={"arxiv_id": str},
)
df_2023 = pd.read_csv(
    RAW_2023,
    usecols=["arxiv_id", "trend_date", "upvotes"],
    dtype={"arxiv_id": str},
)

panel = pd.concat([df_2024, df_2023], ignore_index=True)
print(f"  Raw panel (unioned): {len(panel):,} rows, {panel['arxiv_id'].nunique():,} unique arxiv_ids")

# Strip version suffixes
panel["arxiv_id_clean"] = panel["arxiv_id"].apply(strip_version)

# Parse dates
panel["trend_date"] = pd.to_datetime(panel["trend_date"]).dt.date

# ---------------------------------------------------------------------------
# 2. Load papers_v2 for subfield and first_trend_date
# ---------------------------------------------------------------------------
print("Loading papers_v2 for subfield and first_trend_date...")

pv = pd.read_csv(
    PAPERS_V2,
    usecols=["arxiv_id_clean", "subfield", "first_trend_date"],
    dtype={"arxiv_id_clean": str},
)
pv["first_trend_date"] = pd.to_datetime(pv["first_trend_date"]).dt.date
pv = pv.rename(columns={"first_trend_date": "cohort_day"})

print(f"  papers_v2: {len(pv):,} rows, {pv['arxiv_id_clean'].nunique():,} unique IDs")

# ---------------------------------------------------------------------------
# 3. Dedupe panel to one (paper, cohort-day) row
#    cohort day = first_trend_date from papers_v2 (not re-derived from raw)
#    Use that day's upvotes as the same-day upvote value for both ego and
#    competitors (consistently — not a max-over-days).
# ---------------------------------------------------------------------------
# Join cohort_day and subfield onto panel
panel = panel.merge(
    pv[["arxiv_id_clean", "cohort_day", "subfield"]],
    on="arxiv_id_clean",
    how="inner",
)

# Keep only the row whose trend_date equals the cohort_day (first_trend_date)
# For the one paper with 2 trend days this drops the non-first day.
cohort = panel[panel["trend_date"] == panel["cohort_day"]].copy()

# Safety check: if a paper has multiple rows matching cohort_day (shouldn't
# happen with clean data), keep the first occurrence.
cohort = cohort.drop_duplicates(subset="arxiv_id_clean", keep="first")

cohort = cohort.rename(columns={"upvotes": "ego_upvotes"})[
    ["arxiv_id_clean", "cohort_day", "subfield", "ego_upvotes"]
]

print(f"  Cohort table after dedupe: {len(cohort):,} papers")

# ---------------------------------------------------------------------------
# 5. Build instruments via cohort-day merge (vectorised LOO)
# ---------------------------------------------------------------------------
# For each day d, compute:
#   day_total_upvotes_d  = sum of all upvotes on day d
#   day_count_d          = number of papers on day d
#   day_max_upvotes_d    = max upvotes on day d
# Then ego-subtract to get LOO values.
#
# For Z1' (other-subfield) we need per-(day, subfield) sums.

# --- Day-level aggregates ---
day_agg = (
    cohort.groupby("cohort_day")["ego_upvotes"]
    .agg(day_total=("sum"), day_count=("count"), day_max=("max"))
    .reset_index()
)

# the analysis plan: tau = 90th pct of {day_max_upvotes[d] over all cohort-days d}
# NOT the per-paper upvote distribution (which would set tau far too low).
tau = float(np.percentile(day_agg["day_max"].values, 90))
print(f"\n  tau (90th pct of per-day-max upvotes) = {tau:.1f}")

# --- (Day, subfield)-level upvote sum ---
day_sub_agg = (
    cohort.groupby(["cohort_day", "subfield"])["ego_upvotes"]
    .sum()
    .reset_index()
    .rename(columns={"ego_upvotes": "same_sub_sum"})
)

# --- Merge back onto cohort ---
cohort = cohort.merge(day_agg, on="cohort_day", how="left")
cohort = cohort.merge(day_sub_agg, on=["cohort_day", "subfield"], how="left")

# --- LOO instrument construction ---

# Z1: log(1 + sum_{j≠i} upvotes_j)  =  log(1 + day_total - ego_upvotes)
cohort["Z1_logcompet"] = np.log1p(cohort["day_total"] - cohort["ego_upvotes"])

# Z2: count_{j≠i}  =  day_count - 1
cohort["Z2_count"] = cohort["day_count"] - 1

# Z3: 1[ max_{j≠i} upvotes_j >= tau ]
# max_{j≠i} = day_max  UNLESS ego IS the day_max paper AND the day_max is
# not achieved by anyone else (i.e., ego is the unique max).
# Handle correctly: max_{j≠i} = max of all upvotes except ego.
# Approximate: if ego_upvotes == day_max, we need the second max.
# We can compute this efficiently with a second groupby pass.
# But for most papers ego_upvotes < day_max → max_{j≠i} = day_max.
# For the small fraction where ego IS the day max, compute second max.

# Identify days where some paper is the unique max
# Build (day, ego) second-max lookup only for papers that are the day max
ego_is_max = cohort["ego_upvotes"] == cohort["day_max"]

if ego_is_max.any():
    # For each day, get all upvote values sorted descending
    # We need second-max per day; use a helper
    def day_sorted_upvotes(sub):
        vals = sorted(sub["ego_upvotes"].tolist(), reverse=True)
        return vals

    # Build: (arxiv_id_clean, day) -> max_{j≠i}
    max_excl = {}
    for day, grp in cohort.groupby("cohort_day"):
        vals_sorted = sorted(grp["ego_upvotes"].tolist(), reverse=True)
        day_max_val = vals_sorted[0]
        second_max_val = vals_sorted[1] if len(vals_sorted) > 1 else 0.0
        for _, row in grp.iterrows():
            if row["ego_upvotes"] == day_max_val:
                # exclude ego: max of others = second_max (if ego is unique day_max)
                # if multiple papers tie at day_max, max_{j≠i} = day_max
                count_at_max = vals_sorted.count(day_max_val)
                excl = day_max_val if count_at_max > 1 else second_max_val
            else:
                excl = day_max_val
            max_excl[(row["arxiv_id_clean"], day)] = excl

    cohort["max_excl"] = cohort.apply(
        lambda r: max_excl.get((r["arxiv_id_clean"], r["cohort_day"]), r["day_max"]),
        axis=1,
    )
else:
    cohort["max_excl"] = cohort["day_max"]

cohort["Z3_blockbuster"] = (cohort["max_excl"] >= tau).astype(int)

# Z1' (other-subfield): log(1 + sum_{j≠i, subfield_j ≠ subfield_i} upvotes_j)
#   = log(1 + (day_total - same_sub_sum))
#   [day_total already excludes nothing; same_sub_sum includes ego's own upvotes
#    within its subfield, so total_other_sub = day_total - same_sub_sum;
#    this is already leave-one-out for the subfield bucket because we removed
#    same-subfield entirely — ego's own upvotes are in same_sub_sum, so they
#    are excluded from Z1']
cohort["Z1p_othersub"] = np.log1p(cohort["day_total"] - cohort["same_sub_sum"])

# cohort_size = day_count (including ego, so n_cofeatured_i = day_count - 1 = Z2)
cohort = cohort.rename(columns={"day_count": "cohort_size"})

# ---------------------------------------------------------------------------
# 6. Handle singletons (no competitors): already correct by construction
#    Z1 = log1p(0) = 0, Z2 = 0, Z3 = 0, Z1' = log1p(0) = 0
# ---------------------------------------------------------------------------
singletons_mask = cohort["cohort_size"] == 1
n_singletons = singletons_mask.sum()
# Confirm Z-values are zero for singletons
assert (cohort.loc[singletons_mask, "Z1_logcompet"] == 0).all(), "Singleton Z1 != 0"
assert (cohort.loc[singletons_mask, "Z2_count"] == 0).all(), "Singleton Z2 != 0"
assert (cohort.loc[singletons_mask, "Z3_blockbuster"] == 0).all(), "Singleton Z3 != 0"
assert (cohort.loc[singletons_mask, "Z1p_othersub"] == 0).all(), "Singleton Z1' != 0"

# ---------------------------------------------------------------------------
# 7. Assemble final output
# ---------------------------------------------------------------------------
out = cohort[
    [
        "arxiv_id_clean",
        "cohort_day",
        "cohort_size",
        "Z1_logcompet",
        "Z2_count",
        "Z3_blockbuster",
        "Z1p_othersub",
        "ego_upvotes",
    ]
].copy()

out = out.sort_values(["cohort_day", "arxiv_id_clean"]).reset_index(drop=True)

# IMPORTANT: keep arxiv_id_clean as string. Downstream MUST read with
# dtype={'arxiv_id_clean': str}; float64 corrupts IDs like 2412.01230 → 2412.0123.
out["arxiv_id_clean"] = out["arxiv_id_clean"].astype(str)
out.to_csv(OUT_CSV, index=False)
print(f"\n  Written: {OUT_CSV} ({len(out):,} rows)")

# ---------------------------------------------------------------------------
# 8. Sanity report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SANITY REPORT")
print("=" * 60)

print(f"\ntau (90th pct of per-day-max upvotes) = {tau:.1f}")

print("\nCohort size (Z2_count + 1) distribution:")
cs = out["cohort_size"]
print(f"  min={cs.min()}, p25={cs.quantile(0.25):.0f}, "
      f"median={cs.median():.0f}, p75={cs.quantile(0.75):.0f}, "
      f"max={cs.max()}, mean={cs.mean():.2f}")

# Value counts for small cohorts
vc = cs.value_counts().sort_index()
print("\n  cohort_size  count")
for k, v in vc.head(10).items():
    print(f"  {k:11}  {v:6,}")
if len(vc) > 10:
    print(f"  ... ({len(vc)} unique sizes total)")

print(f"\nSingleton days (cohort_size=1, Z2=0): {n_singletons:,} papers "
      f"({100*n_singletons/len(out):.1f}%)")

share_z3 = out["Z3_blockbuster"].mean()
print(f"\nShare with Z3=1 (blockbuster day): {share_z3:.4f} ({100*share_z3:.1f}%)")

mean_z1 = out["Z1_logcompet"].mean()
print(f"\nMean Z1 (log competing attention): {mean_z1:.4f}")

# Correlation of Z1 with ego upvotes
corr_z1_ego = out["Z1_logcompet"].corr(out["ego_upvotes"])
print(f"\nCorr(Z1_logcompet, ego_upvotes): {corr_z1_ego:.4f}")
print("  [First-stage theory: competition → less ego attention → expect NEGATIVE]")
if corr_z1_ego > 0:
    print("  NOTE: Raw corr is positive (bigger days have more competition AND more "
          "attention on average). The negative effect emerges conditional on day FE.")

print("\nZ2_count (# competitors) distribution:")
z2 = out["Z2_count"]
print(f"  min={z2.min()}, p25={z2.quantile(0.25):.0f}, "
      f"median={z2.median():.0f}, p75={z2.quantile(0.75):.0f}, max={z2.max()}")

print("\nZ1_logcompet distribution:")
z1 = out["Z1_logcompet"]
print(f"  min={z1.min():.3f}, p25={z1.quantile(0.25):.3f}, "
      f"median={z1.median():.3f}, p75={z1.quantile(0.75):.3f}, max={z1.max():.3f}")

print("\nZ1p_othersub distribution:")
z1p = out["Z1p_othersub"]
print(f"  min={z1p.min():.3f}, p25={z1p.quantile(0.25):.3f}, "
      f"median={z1p.median():.3f}, p75={z1p.quantile(0.75):.3f}, max={z1p.max():.3f}")

print("\nego_upvotes distribution:")
eu = out["ego_upvotes"]
print(f"  min={eu.min()}, p25={eu.quantile(0.25):.0f}, "
      f"median={eu.median():.0f}, p75={eu.quantile(0.75):.0f}, max={eu.max()}")

print("\nOutput columns:", list(out.columns))
print("=" * 60)
print("Done.")
