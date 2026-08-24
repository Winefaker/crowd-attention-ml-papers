"""
23_crowding_cohort_v3.py
Same-day crowding instruments recomputed on the UNIFORM keyword taxonomy (`subfield_kw`).

Why a v3: the v1 instrument Z1p_othersub = log1p(day_total - own_(day x subfield) sum) was built on the
legacy `subfield` column, which mixes two taxonomies keyed on an upvote-ordered arXiv fetch
(diagnostics/02_stats_methods_critique.md §A.6, §B, §D.5).  This script rebuilds the same objects on
`subfield_kw` (data/subfield_kw.csv, one rule set for every paper) and adds the two
diagnostic instruments the audit used (§B.1):

    Z1p_kw        = log1p(day_total - own_kw_cell_sum)          # other-subfield upvote sum (v1 definition, new taxonomy)
    Zc_kw         = log1p(day_count - own_kw_cell_count)        # other-subfield paper COUNT (no upvote content)
    Zown_loo_kw   = log1p(own_kw_cell_sum - ego_upvotes)        # own-subfield leave-one-out peer sum (null-first-stage check)
    kw_cell_size  = number of papers in the (cohort_day x subfield_kw) cell
    kw_singleton  = 1[kw_cell_size == 1]

Cohort day = first_trend_date; same-day upvote value = the raw panel row on that day (as in v1).
Output: data/processed/crowding_v3.csv  (arxiv_id_clean, cohort_day, day_total, day_count, own_kw_cell_sum,
        Z1p_kw, Zc_kw, Zown_loo_kw, kw_cell_size, kw_singleton).  Nothing else is touched;
        25_crowding_iv_v3.py merges this file onto analysis_final.csv at run time.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW_2024 = str(Path(__file__).resolve().parents[1] / "data/raw/hf_daily_papers.csv")
RAW_2023 = str(Path(__file__).resolve().parents[1] / "data/raw/hf_daily_papers_2023.csv")
PAPERS_V2 = str(Path(__file__).resolve().parents[1] / "data/processed/papers_v2.csv")
SUBFIELD_KW = str(Path(__file__).resolve().parents[1] / "data/subfield_kw.csv")
V1_CROWDING = str(Path(__file__).resolve().parents[1] / "data/processed/crowding.csv")
OUT_CSV = str(Path(__file__).resolve().parents[1] / "data/processed/crowding_v3.csv")


def strip_version(a):
    return re.sub(r"v\d+$", "", str(a).strip())


panel = pd.concat([pd.read_csv(p, usecols=["arxiv_id", "trend_date", "upvotes"], dtype={"arxiv_id": str})
                   for p in (RAW_2024, RAW_2023)], ignore_index=True)
panel["arxiv_id_clean"] = panel["arxiv_id"].map(strip_version)
panel["trend_date"] = pd.to_datetime(panel["trend_date"]).dt.date

pv = pd.read_csv(PAPERS_V2, usecols=["arxiv_id_clean", "first_trend_date"], dtype={"arxiv_id_clean": str})
pv["cohort_day"] = pd.to_datetime(pv["first_trend_date"]).dt.date
kw = pd.read_csv(SUBFIELD_KW, dtype={"arxiv_id_clean": str})
assert kw["arxiv_id_clean"].is_unique
pv = pv.merge(kw[["arxiv_id_clean", "subfield_kw"]], on="arxiv_id_clean", how="inner")

cohort = panel.merge(pv[["arxiv_id_clean", "cohort_day", "subfield_kw"]], on="arxiv_id_clean", how="inner")
cohort = cohort[cohort["trend_date"] == cohort["cohort_day"]].drop_duplicates("arxiv_id_clean", keep="first")
cohort = cohort.rename(columns={"upvotes": "ego_upvotes"})[["arxiv_id_clean", "cohort_day", "subfield_kw", "ego_upvotes"]]
print(f"cohort table: {len(cohort):,} papers")

g_day = cohort.groupby("cohort_day")["ego_upvotes"]
cohort["day_total"] = g_day.transform("sum")
cohort["day_count"] = g_day.transform("size")
g_cell = cohort.groupby(["cohort_day", "subfield_kw"])["ego_upvotes"]
cohort["own_kw_cell_sum"] = g_cell.transform("sum")
cohort["kw_cell_size"] = g_cell.transform("size")
cohort["kw_singleton"] = (cohort["kw_cell_size"] == 1).astype(int)

cohort["Z1p_kw"] = np.log1p(cohort["day_total"] - cohort["own_kw_cell_sum"])
cohort["Zc_kw"] = np.log1p(cohort["day_count"] - cohort["kw_cell_size"])
cohort["Zown_loo_kw"] = np.log1p(cohort["own_kw_cell_sum"] - cohort["ego_upvotes"])

# sanity: identical universe / ego upvotes / day totals to the v1 file
v1 = pd.read_csv(V1_CROWDING, dtype={"arxiv_id_clean": str})
chk = cohort.merge(v1[["arxiv_id_clean", "ego_upvotes", "cohort_size", "Z1_logcompet"]],
                   on="arxiv_id_clean", suffixes=("", "_v1"))
assert len(chk) == len(cohort) == len(v1), (len(chk), len(cohort), len(v1))
assert (chk["ego_upvotes"] == chk["ego_upvotes_v1"]).all()
assert (chk["day_count"] == chk["cohort_size"]).all()
assert np.allclose(np.log1p(chk["day_total"] - chk["ego_upvotes"]), chk["Z1_logcompet"])
assert (cohort["Zown_loo_kw"] >= 0).all() and (cohort["Z1p_kw"] >= 0).all()

out = cohort[["arxiv_id_clean", "cohort_day", "day_total", "day_count", "own_kw_cell_sum",
              "Z1p_kw", "Zc_kw", "Zown_loo_kw", "kw_cell_size", "kw_singleton"]].copy()
out["arxiv_id_clean"] = out["arxiv_id_clean"].astype(str)
out = out.sort_values(["cohort_day", "arxiv_id_clean"]).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
cells = cohort.groupby(["cohort_day", "subfield_kw"]).size()
print(f"written {OUT_CSV}: {len(out):,} rows; (day x subfield_kw) cells={len(cells):,}, singleton cell share="
      f"{(cells == 1).mean():.3f}, papers in singleton cells={cohort['kw_singleton'].mean():.3f}")
print("Done.")
