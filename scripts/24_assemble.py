"""
C4 — Assemble analysis_final.csv
=================================
Joins papers_v2 base with project_final feature tables:
  clean_dates, prepub_prestige, crowding
Applies the v2 analysis filter (citation_count present, 5 <= age_months <= 40).

NOTE (D3-final): prepub_prestige_tierB.csv (leakage-free Tier B prestige) is still
collecting in the background; it does not exist yet.  When ready, D3-final will
left-join it onto analysis_final on arxiv_id_clean.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
PROC_V2    = BASE_DIR / "data/processed/papers_v2.csv"
PROC_FINAL = BASE_DIR / "data/processed"

OUT_CSV  = PROC_FINAL / "analysis_final.csv"
OUT_DICT = PROC_FINAL / "analysis_final_DICT.md"

STR_ID = {"arxiv_id_clean": str}   # CRITICAL: never let pandas cast to float64

# ── columns to carry forward from papers_v2 ───────────────────────────────────
KW_COLS = [
    "kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning", "kw_benchmark",
    "kw_survey", "kw_efficient", "kw_multimodal", "kw_rl", "kw_scaling",
]
BASE_KEEP = [
    "arxiv_id_clean",
    "upvotes", "log_upvotes",
    "citation_count", "log_citations",
    "influential_citations", "reference_count",
    "subfield", "release_month", "release_year", "age_months",
    "max_hindex", "last_author_hindex",
    "author_max_appear", "has_github",
    "n_authors",
    "n_trend_days", "first_trend_date",
    "title_n_words", "title_has_colon", "abstract_n_chars",
    "num_comments", "github_stars", "primary_category",   # Aug-2026: extra HF signals + arXiv cat (validation only)
] + KW_COLS

# ── 1. load base ──────────────────────────────────────────────────────────────
print("Loading papers_v2 …")
base = pd.read_csv(PROC_V2, dtype=STR_ID, low_memory=False)
# keep only the columns we need (intersect with what's actually present)
keep = [c for c in BASE_KEEP if c in base.columns]
base = base[keep].copy()
n_base = len(base)
print(f"  base rows: {n_base:,}")

# Verify arxiv_id_clean is NOT float (object or StringDtype are both fine)
assert not pd.api.types.is_float_dtype(base["arxiv_id_clean"]), \
    "arxiv_id_clean is float64 in base — id truncation hazard!"

# ── 2. load feature tables ────────────────────────────────────────────────────
print("Loading clean_dates …")
dates = pd.read_csv(PROC_FINAL / "clean_dates.csv", dtype=STR_ID)
# keep only new columns (drop published_v1 if base already has it)
dates_keep = ["arxiv_id_clean", "gap_days_v1", "gap_days_rawpub", "v1_source"]
if "published_v1" not in base.columns:
    dates_keep.insert(1, "published_v1")
dates = dates[[c for c in dates_keep if c in dates.columns]]
n_dates = len(dates)
print(f"  clean_dates rows: {n_dates:,}")

print("Loading prepub_prestige …")
prestige_cols_wanted = [
    "arxiv_id_clean",
    "first_author_papercount_cur2026", "last_author_papercount_cur2026",
    "max_papercount_cur2026",
    "first_author_papercount_cur2026_w99", "last_author_papercount_cur2026_w99",
    "max_papercount_cur2026_w99",
    "prestige_resolved", "anchor_date_used",
]
prestige_raw = pd.read_csv(PROC_FINAL / "prepub_prestige.csv", dtype=STR_ID)
prestige = prestige_raw[[c for c in prestige_cols_wanted if c in prestige_raw.columns]]
n_prestige = len(prestige)
print(f"  prepub_prestige rows: {n_prestige:,}")

print("Loading crowding …")
crowding = pd.read_csv(PROC_FINAL / "crowding.csv", dtype=STR_ID)
n_crowding = len(crowding)
print(f"  crowding rows: {n_crowding:,}")

# ── 3. left-join merges ───────────────────────────────────────────────────────
print("\nMerging …")

df = base.merge(dates,    on="arxiv_id_clean", how="left", suffixes=("", "_dates"))
# ID join: check via gap_days_rawpub which has 100% coverage in source
n_match_dates_id  = df["gap_days_rawpub"].notna().sum()   # should equal n_base
n_match_dates_v1  = df["gap_days_v1"].notna().sum()       # subset: v1 date resolved
n_match_dates = n_match_dates_id   # canonical join match count
print(f"  clean_dates   ID join:    {n_match_dates_id:,} / {n_base:,}  "
      f"({100*n_match_dates_id/n_base:.1f}%)  [all IDs found]")
print(f"  clean_dates   gap_days_v1 coverage: {n_match_dates_v1:,} / {n_base:,}  "
      f"({100*n_match_dates_v1/n_base:.1f}%)  [v1 date resolved subset]")

df = df.merge(prestige,  on="arxiv_id_clean", how="left", suffixes=("", "_prestige"))
n_match_prestige = df["prestige_resolved"].notna().sum()
print(f"  prepub_prestige match: {n_match_prestige:,} / {n_base:,}  "
      f"({100*n_match_prestige/n_base:.1f}%)")

df = df.merge(crowding,  on="arxiv_id_clean", how="left", suffixes=("", "_crowd"))
n_match_crowding = df["Z1_logcompet"].notna().sum()
print(f"  crowding      match: {n_match_crowding:,} / {n_base:,}  "
      f"({100*n_match_crowding/n_base:.1f}%)")

# ── Aug-2026 additions ───────────────────────────────────────────────────────
# (a) uniform subfield taxonomy (one keyword rule set for every paper; see
#     scripts/30_subfield_uniform.py). The v2 `subfield` mixes arXiv-category
#     labels (fetched high-upvote-first) with keyword labels, so label source proxies
#     attention. `subfield` is kept for backward comparison; models should use subfield_kw.
SUBFIELD_KW = BASE_DIR / "data/subfield_kw.csv"
skw = pd.read_csv(SUBFIELD_KW, dtype=STR_ID)
df = df.merge(skw, on="arxiv_id_clean", how="left")
print(f"  subfield_kw   match: {df['subfield_kw'].notna().sum():,} / {n_base:,}")
df["subfield_kw"] = df["subfield_kw"].fillna("Other")

# (b) Tier B leakage-free prestige (strictly-prior paper counts + years active as of the
#     paper's own submission date). Collected by scripts/22_prepub_prestige_fixed.py.
TIERB = PROC_FINAL / "prepub_prestige_tierB.csv"
if TIERB.exists():
    tb = pd.read_csv(TIERB, dtype=STR_ID)
    tb = tb[["arxiv_id_clean", "first_author_prior_papers_true", "last_author_prior_papers_true",
             "max_prior_papers_true", "max_years_active", "tierB_resolved"]].copy()
    # clean obvious S2 author-disambiguation garbage (years active > 60, first year < 1950)
    tb["max_years_active"] = tb["max_years_active"].clip(lower=0, upper=60)
    for c in ["first_author_prior_papers_true", "last_author_prior_papers_true", "max_prior_papers_true"]:
        q99 = tb[c].quantile(0.99)
        tb[c + "_w99"] = tb[c].clip(upper=q99)
    tb["log1p_max_prior_papers_true"] = np.log1p(tb["max_prior_papers_true"])
    df = df.merge(tb, on="arxiv_id_clean", how="left")
    print(f"  tierB prestige match: {df['tierB_resolved'].notna().sum():,} / {n_base:,}  "
          f"(resolved {int(df['tierB_resolved'].fillna(False).sum()):,})")
else:
    print("  tierB prestige: file not found (skipped)")

assert len(df) == n_base, f"Row count changed after merges: {len(df)} != {n_base}"

# ── 4. derive dow ─────────────────────────────────────────────────────────────
print("\nDeriving day-of-week (dow) from first_trend_date …")
df["first_trend_date"] = pd.to_datetime(df["first_trend_date"], errors="coerce")
df["dow"] = df["first_trend_date"].dt.day_name()  # Mon..Sun strings

# ── 5. null rates for key FE columns ─────────────────────────────────────────
print("\nNull rates for key columns:")
for col in ["release_month", "subfield", "dow"]:
    n_null = df[col].isna().sum()
    print(f"  {col}: {n_null:,} nulls ({100*n_null/len(df):.2f}%)")

# ── 6. apply v2 analysis filter ───────────────────────────────────────────────
print("\nApplying v2 analysis filter …")
n_before = len(df)
mask = (
    df["citation_count"].notna() &
    df["age_months"].between(5, 40, inclusive="both")
)
df_filt = df[mask].copy()
n_after = len(df_filt)
print(f"  Before filter: {n_before:,}")
print(f"  After  filter: {n_after:,}  (v2 reference was 11,347)")

# ── 7. FE sanity ─────────────────────────────────────────────────────────────
print("\nFixed-effect levels (post-filter):")
n_rm   = df_filt["release_month"].nunique()
n_sf   = df_filt["subfield"].nunique()
n_dow  = df_filt["dow"].nunique()
print(f"  release_month levels: {n_rm}")
print(f"  subfield      levels: {n_sf}")
print(f"  subfield_kw   levels: {df_filt['subfield_kw'].nunique()}")
print(f"  dow           levels: {n_dow}")

# sparse FE cells: (release_month × subfield) cells with < 5 obs
cell_counts = (
    df_filt.groupby(["release_month", "subfield"], observed=True)
    .size()
)
n_sparse = (cell_counts < 5).sum()
print(f"  (release_month x subfield) cells < 5 obs: {n_sparse} "
      f"out of {len(cell_counts)} total cells")

# ── 8. sanity correlations ────────────────────────────────────────────────────
print("\nSanity correlations (post-filter):")
corr_z1p_upvotes  = df_filt[["Z1p_othersub", "log_upvotes"]].corr().iloc[0, 1]
corr_upv_cit      = df_filt[["log_upvotes",  "log_citations"]].corr().iloc[0, 1]
print(f"  corr(Z1p_othersub, log_upvotes) = {corr_z1p_upvotes:.4f}")
print(f"  corr(log_upvotes, log_citations) = {corr_upv_cit:.4f}")

# ── 9. write output CSV ───────────────────────────────────────────────────────
print(f"\nWriting {OUT_CSV} …")
df_filt.to_csv(OUT_CSV, index=False)
print(f"  Wrote {len(df_filt):,} rows x {len(df_filt.columns)} columns")

# verify id dtype survived round-trip
df_check = pd.read_csv(OUT_CSV, dtype=STR_ID, nrows=5)
assert not pd.api.types.is_float_dtype(df_check["arxiv_id_clean"]), \
    "arxiv_id_clean is float64 in output — id truncation hazard!"
print("  arxiv_id_clean dtype check: OK (string)")

# ── 10. write data dictionary ─────────────────────────────────────────────────
dict_text = f"""# analysis_final — Data Dictionary
Generated by `scripts/24_assemble.py`

## Assembly summary

| Item | Value |
|------|-------|
| Base (papers_v2) rows | {n_base:,} |
| clean_dates ID join | {n_match_dates_id:,} / {n_base:,} ({100*n_match_dates_id/n_base:.1f}%) — all IDs matched |
| clean_dates gap_days_v1 coverage | {n_match_dates_v1:,} / {n_base:,} ({100*n_match_dates_v1/n_base:.1f}%) — v1 date resolved subset |
| prepub_prestige match | {n_match_prestige:,} / {n_base:,} ({100*n_match_prestige/n_base:.1f}%) |
| crowding match | {n_match_crowding:,} / {n_base:,} ({100*n_match_crowding/n_base:.1f}%) |
| Rows before v2 filter | {n_before:,} |
| Rows after v2 filter (citation present, 5 ≤ age_months ≤ 40) | {n_after:,} |
| v2 reference count | 11,347 |

## Fixed-effect levels (post-filter)

| Dimension | Levels |
|-----------|--------|
| release_month | {n_rm} |
| subfield | {n_sf} |
| dow | {n_dow} |
| (release_month × subfield) cells < 5 obs | {n_sparse} |

## Sanity correlations (post-filter)

| Pair | r |
|------|---|
| Z1p_othersub vs log_upvotes | {corr_z1p_upvotes:.4f} |
| log_upvotes vs log_citations | {corr_upv_cit:.4f} |

## Column provenance

### From papers_v2 (base)
| Column | Description |
|--------|-------------|
| arxiv_id_clean | Paper identifier (kept as string — float64 silently truncates trailing zeros) |
| upvotes | HuggingFace paper upvote count |
| log_upvotes | log1p(upvotes) — primary engagement outcome |
| citation_count | Semantic Scholar citation count (filter criterion: must be non-null) |
| log_citations | log1p(citation_count) — secondary outcome |
| influential_citations | SS influential citation count |
| reference_count | SS reference count |
| subfield | AI subfield label (FE dimension) |
| release_month | YYYY-MM of HF trending appearance (FE dimension) |
| release_year | Year of trending appearance |
| age_months | Months since arXiv v1 to HF trend date (filter: 5–40) |
| max_hindex | Max h-index across all authors at paper time |
| last_author_hindex | Last author h-index |
| author_max_appear | Max author appearance count (proxy for community prominence) |
| has_github | Binary: GitHub repo linked on HF |
| n_authors | Number of authors |
| n_trend_days | Days on HF trending list |
| first_trend_date | Date first appeared on HF trending (source of dow) |
| title_n_words | Word count of paper title |
| title_has_colon | Binary: title contains colon |
| abstract_n_chars | Character length of abstract |
| kw_llm, kw_agent, kw_diffusion, kw_reasoning, kw_benchmark, kw_survey, kw_efficient, kw_multimodal, kw_rl, kw_scaling | Keyword flags from title/abstract |

### Derived
| Column | Description |
|--------|-------------|
| dow | Day of week (Mon..Sun) of first_trend_date (FE dimension) |

### From clean_dates (data/processed/clean_dates.csv)
| Column | Description |
|--------|-------------|
| published_v1 | arXiv v1 submission date |
| gap_days_v1 | Days from arXiv v1 to first HF trend date (NaN for ~72% of rows where v1 not resolved) |
| gap_days_rawpub | Days from raw `published_at` field to first HF trend date (100% coverage) |
| v1_source | Source used to resolve v1 date |

### From prepub_prestige (data/processed/prepub_prestige.csv)
| Column | Description |
|--------|-------------|
| first_author_papercount_cur2026 | First author paper count as of early 2026 |
| last_author_papercount_cur2026 | Last author paper count as of early 2026 |
| max_papercount_cur2026 | Max paper count across first/last author as of early 2026 |
| first_author_papercount_cur2026_w99 | Same, winsorised at 99th percentile |
| last_author_papercount_cur2026_w99 | Same, winsorised |
| max_papercount_cur2026_w99 | Same, winsorised |
| prestige_resolved | Resolved prestige score (method-blended) |
| anchor_date_used | Anchor date used for pre-pub prestige lookup |

> **Note (D3-final):** `prepub_prestige_tierB.csv` (leakage-free Tier B prestige,
> using only pre-publication author histories) is still collecting in the background
> and does not yet exist. When ready, D3-final will left-join it onto
> `analysis_final.csv` on `arxiv_id_clean`.

### From crowding (data/processed/crowding.csv)
| Column | Description |
|--------|-------------|
| cohort_day | Date defining the co-submission cohort |
| cohort_size | Number of HF papers in the same daily cohort |
| Z1_logcompet | log competition index (all subfields) |
| Z2_count | Raw competitor count |
| Z3_blockbuster | Blockbuster crowding indicator |
| Z1p_othersub | log competition index (other subfields only) |
| ego_upvotes | Ego paper upvotes as recorded in crowding table |

## Key modelling notes

- `arxiv_id_clean` is **always** read with `dtype={{'arxiv_id_clean': str}}`.
  Float64 silently truncates IDs ending in `0` (e.g. 2412.01230 → 2412.0123).
- v2 analysis filter: `citation_count` non-null AND `5 ≤ age_months ≤ 40`.
  This matches the filter used in the v2 baseline models (reference n = 11,347).
- Tier B prestige join is pending (see note above).
"""

print(f"Writing {OUT_DICT} …")
OUT_DICT.write_text(dict_text)
print("  Done.")

print("\n=== C4 complete ===")
print(f"  analysis_final.csv:  {n_after:,} rows x {len(df_filt.columns)} columns")
print(f"  clean_dates   ID join: {100*n_match_dates_id/n_base:.1f}%  "
      f"| gap_days_v1 coverage: {100*n_match_dates_v1/n_base:.1f}%")
print(f"  prestige      match rate: {100*n_match_prestige/n_base:.1f}%")
print(f"  crowding      match rate: {100*n_match_crowding/n_base:.1f}%")
print(f"  FE levels — release_month:{n_rm}, subfield:{n_sf}, dow:{n_dow}")
print(f"  Sparse (release_month x subfield) cells: {n_sparse}")
print(f"  corr(Z1p_othersub, log_upvotes) = {corr_z1p_upvotes:.4f}")
print(f"  corr(log_upvotes, log_citations) = {corr_upv_cit:.4f}")
