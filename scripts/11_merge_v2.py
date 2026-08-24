"""
11_merge_v2.py
--------------
Build the v2 analysis dataset:
  - HF panel = 2023 backfill + 2024-25 main collection, collapsed to one row/paper
  - Semantic Scholar v2 (uniform 2026-06-11 snapshot, with author h-index prestige)
  - targeted outcome repairs (09)
  - arXiv categories where available (03's partial output, for validation)
  - all v1 features + new prestige features

Output: data/processed/papers_v2.csv
"""
import pandas as pd
import numpy as np
import os
import re
import sys

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SNAPSHOT = pd.Timestamp("2026-06-11")

sys.path.insert(0, os.path.dirname(__file__))
# reuse the keyword->subfield rules from session 1 (kept in one place)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "m4", os.path.join(os.path.dirname(__file__), "04_merge_and_features.py"))
m4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4)
coarse_subfield = m4.coarse_subfield


def clean_id(x):
    x = str(x).strip()
    return x.split("v")[0] if "v" in x.split(".")[-1] else x


def main():
    # ---- HF panel (both windows) ---------------------------------------
    parts = []
    for f in ["hf_daily_papers_2023.csv", "hf_daily_papers.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            d = pd.read_csv(p, dtype={"arxiv_id": str})
            parts.append(d)
    hf = pd.concat(parts, ignore_index=True)
    hf["arxiv_id_clean"] = hf["arxiv_id"].map(clean_id)
    print(f"HF panel: {len(hf)} rows, {hf['arxiv_id_clean'].nunique()} unique papers")

    hf_sorted = hf.sort_values("trend_date")
    df = hf_sorted.groupby("arxiv_id_clean").agg(
        upvotes=("upvotes", "max"),
        num_comments=("num_comments", "max"),
        github_stars=("github_stars", "max"),
        n_trend_days=("trend_date", "nunique"),
        first_trend_date=("trend_date", "first"),
        hf_title=("hf_title", "first"),
        hf_summary=("hf_summary", "first"),
        published_at=("published_at", "first"),
        n_authors_hf=("n_authors_hf", "first"),
        submitted_by=("submitted_by", "first"),
        ai_keywords=("ai_keywords", "first"),
        github_repo=("github_repo", "first"),
        author_names=("author_names", "first"),
    ).reset_index()

    # ---- S2 v2 (outcome + h-index prestige) ----------------------------
    ss = pd.read_csv(os.path.join(RAW, "semantic_scholar_v2.csv"), dtype={"arxiv_id_clean": str})
    ss = ss.drop_duplicates("arxiv_id_clean")
    df = df.merge(ss, on="arxiv_id_clean", how="left")

    # ---- outcome repairs ------------------------------------------------
    rep_path = os.path.join(RAW, "s2_repairs.csv")
    df["citation_repaired"] = 0
    if os.path.exists(rep_path) and os.path.getsize(rep_path) > 10:
        rep = pd.read_csv(rep_path, dtype={"arxiv_id_clean": str}).drop_duplicates("arxiv_id_clean")
        df = df.merge(rep[["arxiv_id_clean", "repaired_citations", "repaired_influential",
                           "repaired_references"]], on="arxiv_id_clean", how="left")
        fix = df["repaired_citations"].notna()
        df.loc[fix, "citation_count"] = df.loc[fix, "repaired_citations"]
        df.loc[fix & df["influential_citations"].isna(), "influential_citations"] = \
            df.loc[fix, "repaired_influential"]
        df.loc[fix & df["reference_count"].isna(), "reference_count"] = \
            df.loc[fix, "repaired_references"]
        df.loc[fix, "ss_found"] = 1
        df.loc[fix, "citation_repaired"] = 1
        print(f"applied {int(fix.sum())} outcome repairs")

    # ---- arXiv categories (partial, validation) ------------------------
    ax_path = os.path.join(RAW, "arxiv_meta.csv")
    if os.path.exists(ax_path) and os.path.getsize(ax_path) > 10:
        ax = pd.read_csv(ax_path, dtype={"arxiv_id_clean": str}).drop_duplicates("arxiv_id_clean")
        keep = [c for c in ["arxiv_id_clean", "primary_category", "all_categories",
                            "published_v1", "latest_version", "n_authors_arxiv"] if c in ax.columns]
        df = df.merge(ax[keep], on="arxiv_id_clean", how="left")
        print(f"arXiv categories available for {df['primary_category'].notna().sum()} papers")
    for c in ["primary_category", "all_categories", "published_v1", "latest_version", "n_authors_arxiv"]:
        if c not in df.columns:
            df[c] = np.nan

    # ---- dates & age (snapshot 2026-06-11) ------------------------------
    rel = pd.to_datetime(df["published_v1"].fillna(df["published_at"]),
                         errors="coerce", utc=True).dt.tz_localize(None)
    rel = rel.fillna(pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None))
    df["release_dt"] = rel
    df["release_month"] = rel.dt.to_period("M").astype(str)
    df["release_year"] = rel.dt.year
    df["age_months"] = (SNAPSHOT - rel).dt.days / 30.44

    # ---- outcome & signal ------------------------------------------------
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")
    df["log_citations"] = np.log1p(df["citation_count"])
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce")
    df["log_upvotes"] = np.log1p(df["upvotes"])
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0)
    df["github_stars"] = pd.to_numeric(df["github_stars"], errors="coerce").fillna(0)
    df["has_github"] = df["github_repo"].notna().astype(int)
    df["reference_count"] = pd.to_numeric(df["reference_count"], errors="coerce")

    # ---- prestige: REAL h-index + v1 recurrence proxy -------------------
    for c in ["max_hindex", "mean_hindex", "first_author_hindex", "last_author_hindex"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log_max_hindex"] = np.log1p(df["max_hindex"])
    df["log_last_hindex"] = np.log1p(df["last_author_hindex"])

    # v1 recurrence proxy (rebuilt on the full panel, for comparison)
    from collections import defaultdict
    author_papers = defaultdict(set)
    hf_a = hf.dropna(subset=["author_names"])
    for _, r in hf_a.iterrows():
        for nm in str(r["author_names"]).split(";"):
            nm = nm.strip()
            if nm:
                author_papers[nm].add(r["arxiv_id_clean"])
    acount = {nm: len(s) for nm, s in author_papers.items()}
    pa = hf_a.sort_values("trend_date").groupby("arxiv_id_clean")["author_names"].first().to_dict()

    def prest(aid):
        names = [n.strip() for n in str(pa.get(aid, "")).split(";") if n.strip()]
        if not names:
            return 0
        return max(max(acount.get(nm, 1) - 1, 0) for nm in names)
    df["author_max_appear"] = df["arxiv_id_clean"].map(prest)
    df["log_author_max_appear"] = np.log1p(df["author_max_appear"])

    df["n_authors"] = df["n_authors_arxiv"].fillna(df["n_authors_s2"]).fillna(df["n_authors_hf"])
    df["log_n_authors"] = np.log1p(pd.to_numeric(df["n_authors"], errors="coerce"))

    # ---- subfield --------------------------------------------------------
    df["subfield"] = df.apply(lambda r: coarse_subfield(
        r.get("primary_category"), None, r.get("ai_keywords")), axis=1)

    # ---- text features ----------------------------------------------------
    df["title"] = df["hf_title"].fillna("")
    df["title_n_words"] = df["title"].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    df["title_has_colon"] = df["title"].str.contains(":").astype(int)
    df["abstract_n_chars"] = df["hf_summary"].fillna("").str.len()
    tl = df["title"].str.lower()
    for kw in ["llm", "agent", "diffusion", "reasoning", "benchmark",
               "survey", "efficient", "multimodal", "rl", "scaling"]:
        df[f"kw_{kw}"] = tl.str.contains(rf"\b{kw}", regex=True).astype(int)

    os.makedirs(PROC, exist_ok=True)
    out = os.path.join(PROC, "papers_v2.csv")
    df.to_csv(out, index=False)
    print(f"papers_v2: {df.shape[0]} x {df.shape[1]}")
    print(f"  with citations: {df['citation_count'].notna().sum()}")
    print(f"  with h-index:   {df['max_hindex'].notna().sum()}")
    print(f"  repaired:       {int(df['citation_repaired'].sum())}")
    print(f"  release years:  {df['release_year'].value_counts().sort_index().to_dict()}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
