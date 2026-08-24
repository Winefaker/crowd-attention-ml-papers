#!/usr/bin/env python3
"""
22_prepub_prestige.py — Leakage-free pre-publication author prestige features.

IMPLEMENTED METHOD (two-tier, determined by S2 API constraints):

Tier A  (default, always runs): Author-batch approach
  - Phase 1: POST /paper/batch to get first/last author IDs per paper (23 calls).
  - Phase 2: POST /author/batch with fields=paperCount,name (31 calls).
    Returns CURRENT total paper count for each author.
    Stored as `first/last_author_papercount_cur2026` (current 2026 total, upper-bound on prior count).
  - `max_papercount_cur2026` = max(first, last) current paper count.
    This is an upper bound: includes papers published AFTER submission date.
    It is NOT citation-leaky (paperCount ≠ f(citations)); it is career-total leaky.
  - `years_active` = NaN (requires per-author paper list; see --per-author mode).
  - Full coverage: ~99% of papers with S2 IDs.

Tier B  (--per-author flag, resumable): Per-author paginated approach
  - Phase 2b: GET /author/{id}/papers paginated for each unique author.
    Computes STRICT prior_paper_count (STRICTLY before anchor_date) and
    years_active = submission_year - min(author paper year).
  - At keyless rate (~1.5 req/sec with backoffs), ~15k authors × 1.5s = ~6 hrs.
    Use for a stratified subset or run overnight with checkpointing.

Resumable: re-running the script always resumes from checkpoints.
Usage:
    /path/to/venv/bin/python 22_prepub_prestige.py [--per-author] [--dry-run]
"""

import os
import argparse, json, logging, math, random, sys, time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parents[2]
PAPERS_CSV     = Path(__file__).resolve().parents[2] / "data/processed/papers_v2.csv"
DATES_CSV      = ROOT / "data/processed/clean_dates.csv"
OUT_CSV        = ROOT / "data/processed/prepub_prestige.csv"          # Tier A interim (never clobbered by Tier B)
OUT_CSV_TIERB  = ROOT / "data/processed/prepub_prestige_tierB.csv"   # Tier B strict-prior (separate)
NOTE_MD        = ROOT / "data/processed/prepub_prestige_NOTE.md"
CKPT_DIR       = ROOT / "data/raw/prestige_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PAPER_AUTHORS  = CKPT_DIR / "paper_authors.json"    # ss_paper_id → {first_id,...}
CKPT_AUTHOR_INFO    = CKPT_DIR / "author_info.json"      # author_id → {paperCount,...}
CKPT_AUTHOR_PAPERS  = CKPT_DIR / "author_papers.json"    # author_id → [paper_rec,...] (Tier B)
CKPT_BATCH_DONE     = CKPT_DIR / "paper_batch_done.json" # paper IDs done in phase 1
CKPT_AUTH_DONE      = CKPT_DIR / "auth_done.json"        # author IDs done in phase 2
CKPT_AUTH_HIST_DONE = CKPT_DIR / "auth_hist_done.json"   # author IDs done in tier B

# ---------------------------------------------------------------------------
# S2 API config
# ---------------------------------------------------------------------------
S2_BASE    = "https://api.semanticscholar.org/graph/v1"
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
UA         = "hf-papers-study/1.0" + (f" (mailto:{CONTACT})" if CONTACT else "")
HEADERS    = {"User-Agent": UA}
REQ_DELAY  = 4.0    # seconds between requests (keyless empirically ~4 req/min with bursts)
MAX_RETRY  = 6
PAPER_BATCH_SIZE  = 500
AUTHOR_BATCH_SIZE = 500
AUTHOR_PAGE_LIMIT = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _sleep():
    time.sleep(REQ_DELAY)

def _backoff(attempt: int) -> float:
    wait = min(120.0, 12 * (2 ** attempt) + random.uniform(0, 5))
    log.warning("  backoff %.1f s (attempt %d)", wait, attempt)
    time.sleep(wait)
    return wait

def s2_post(endpoint: str, body: dict, params: dict | None = None) -> list | dict:
    url = S2_BASE + endpoint
    for attempt in range(MAX_RETRY):
        _sleep()
        try:
            r = requests.post(url, json=body, params=params, headers=HEADERS, timeout=60)
        except requests.exceptions.RequestException as e:
            log.warning("  request error: %s", e)
            _backoff(attempt)
            continue
        if r.status_code == 429:
            _backoff(attempt)
            continue
        if r.status_code in (500, 502, 503, 504):
            _backoff(attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"POST {endpoint} failed after {MAX_RETRY} retries")

def s2_get(endpoint: str, params: dict | None = None) -> dict:
    url = S2_BASE + endpoint
    for attempt in range(MAX_RETRY):
        _sleep()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except requests.exceptions.RequestException as e:
            log.warning("  request error: %s", e)
            _backoff(attempt)
            continue
        if r.status_code == 429:
            _backoff(attempt)
            continue
        if r.status_code in (500, 502, 503, 504):
            _backoff(attempt)
            continue
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {endpoint} failed after {MAX_RETRY} retries")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    tmp.replace(path)

# ---------------------------------------------------------------------------
# Data loading & anchor dates
# ---------------------------------------------------------------------------
def load_inputs():
    df = pd.read_csv(PAPERS_CSV, dtype={"ss_paper_id": str, "arxiv_id_clean": str})
    cd = pd.read_csv(DATES_CSV,  dtype={"arxiv_id_clean": str})
    # merge clean v1 dates from Spike 1
    df = df.merge(cd[["arxiv_id_clean", "published_v1"]], on="arxiv_id_clean", how="left",
                  suffixes=("_orig", ""))
    if "published_v1_orig" in df.columns and "published_v1" in df.columns:
        df["published_v1"] = df["published_v1"].fillna(df["published_v1_orig"])
        df.drop(columns=["published_v1_orig"], inplace=True)

    # Anchor date: published_v1 (Spike 1) if present, else release_dt (fallback)
    df["anchor_date"] = pd.to_datetime(df["published_v1"], errors="coerce", utc=True).dt.tz_localize(None)
    df["release_dt2"] = pd.to_datetime(df["release_dt"], errors="coerce", utc=True).dt.tz_localize(None)
    mask_no_v1 = df["anchor_date"].isna()
    df.loc[mask_no_v1, "anchor_date"] = df.loc[mask_no_v1, "release_dt2"]
    df["anchor_date_used"] = np.where(df["published_v1"].notna(), "published_v1", "release_dt")
    df["submission_year"]  = df["anchor_date"].dt.year
    df["anchor_date_str"]  = df["anchor_date"].dt.strftime("%Y-%m-%d")

    log.info("Loaded %d papers; anchor=published_v1 for %d (%.1f%%)",
             len(df),
             df["anchor_date_used"].eq("published_v1").sum(),
             100 * df["anchor_date_used"].eq("published_v1").mean())
    return df

# ---------------------------------------------------------------------------
# Phase 1: Paper batch → author IDs
# ---------------------------------------------------------------------------
def phase1_paper_authors(df: pd.DataFrame) -> dict:
    """Returns paper_authors: {ss_paper_id: {first_id, last_id, first_name, last_name}}"""
    paper_authors = load_json(CKPT_PAPER_AUTHORS, {})
    done_set      = set(load_json(CKPT_BATCH_DONE, []))

    valid = df[df["ss_paper_id"].notna()].copy()
    valid["ss_paper_id"] = valid["ss_paper_id"].astype(str).str.strip()
    all_ids   = valid["ss_paper_id"].unique().tolist()
    remaining = [pid for pid in all_ids if pid not in done_set]

    log.info("Phase 1: %d total ss_paper_ids, %d already done, %d to fetch",
             len(all_ids), len(done_set), len(remaining))

    chunks = [remaining[i:i+PAPER_BATCH_SIZE] for i in range(0, len(remaining), PAPER_BATCH_SIZE)]
    for ci, chunk in enumerate(chunks):
        log.info("  Paper batch %d/%d (%d ids)", ci+1, len(chunks), len(chunk))
        data = s2_post("/paper/batch", {"ids": chunk},
                       params={"fields": "authors.authorId,authors.name"})
        for item in data:
            if item is None: continue
            pid     = item.get("paperId")
            authors = item.get("authors", []) or []
            if not pid or not authors: continue
            first = authors[0]  if authors          else {}
            last  = authors[-1] if len(authors) > 1 else first
            paper_authors[pid] = {
                "first_id":   first.get("authorId"),
                "first_name": first.get("name"),
                "last_id":    last.get("authorId"),
                "last_name":  last.get("name"),
                "n_authors":  len(authors),
            }
        done_set.update(chunk)
        save_json(CKPT_PAPER_AUTHORS, paper_authors)
        save_json(CKPT_BATCH_DONE, list(done_set))

    log.info("Phase 1 done: %d papers resolved", len(paper_authors))
    return paper_authors

# ---------------------------------------------------------------------------
# Phase 2 (Tier A): Author batch → current paper counts
# ---------------------------------------------------------------------------
def phase2_author_info(paper_authors: dict) -> dict:
    """
    Returns author_info: {author_id: {paperCount: N, name: '...'}}

    Uses POST /author/batch with fields=paperCount,name (supported by S2 API).
    NOTE: paperCount is the CURRENT total (not as-of-submission), so
    first/last_author_papercount_cur2026 is an UPPER BOUND on the true prior count.
    This is leakage-free w.r.t. citations but includes post-submission papers (anachronistic).
    """
    author_info = load_json(CKPT_AUTHOR_INFO, {})
    done_set    = set(load_json(CKPT_AUTH_DONE, []))

    all_aids = set()
    for v in paper_authors.values():
        if v.get("first_id"): all_aids.add(v["first_id"])
        if v.get("last_id"):  all_aids.add(v["last_id"])

    remaining = [aid for aid in all_aids if aid not in done_set]
    log.info("Phase 2 (Tier A): %d unique authors, %d remaining", len(all_aids), len(remaining))

    if not remaining:
        log.info("Phase 2: all done from checkpoint")
        return author_info

    chunks = [remaining[i:i+AUTHOR_BATCH_SIZE] for i in range(0, len(remaining), AUTHOR_BATCH_SIZE)]
    log.info("  %d author batch calls (~%.0f s)", len(chunks), len(chunks) * REQ_DELAY)

    for ci, chunk in enumerate(chunks):
        log.info("  Author batch chunk %d/%d (%d authors)", ci+1, len(chunks), len(chunk))
        try:
            data = s2_post("/author/batch", {"ids": chunk},
                           params={"fields": "paperCount,name"})
        except Exception as e:
            log.warning("  Author batch failed (%s); skipping chunk", e)
            continue
        if not isinstance(data, list):
            log.warning("  Unexpected type %s; skipping chunk", type(data))
            continue
        for item in data:
            if item is None: continue
            aid = item.get("authorId")
            if not aid: continue
            author_info[aid] = {
                "paperCount": item.get("paperCount"),
                "name":       item.get("name"),
            }
            done_set.add(aid)
        save_json(CKPT_AUTHOR_INFO, author_info)
        save_json(CKPT_AUTH_DONE, list(done_set))

    log.info("Phase 2 done: %d authors in info map", len(author_info))
    return author_info

# ---------------------------------------------------------------------------
# Phase 2b (Tier B): Per-author paginated paper history
# ---------------------------------------------------------------------------
def _norm_paper_rec(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {
        "paperId":         p.get("paperId"),
        "year":            p.get("year"),
        "publicationDate": p.get("publicationDate"),
        "arxivId":         ext.get("ArXiv"),
    }

def phase2b_per_author(paper_authors: dict, sample_paper_ids: set | None = None) -> dict:
    """
    Tier B: fetch full paper history per author via paginated GET /author/{id}/papers.

    Returns author_papers: {author_id: list[{paperId,year,publicationDate,arxivId}]}

    sample_paper_ids: restrict to authors of those papers (None = all).
    Checkpoint: CKPT_AUTHOR_PAPERS / CKPT_AUTH_HIST_DONE.
    """
    author_papers = load_json(CKPT_AUTHOR_PAPERS, {})
    done_set      = set(load_json(CKPT_AUTH_HIST_DONE, []))

    # Which authors do we need?
    if sample_paper_ids is not None:
        needed_aids = set()
        for pid, v in paper_authors.items():
            if pid in sample_paper_ids:
                if v.get("first_id"): needed_aids.add(v["first_id"])
                if v.get("last_id"):  needed_aids.add(v["last_id"])
    else:
        needed_aids = set()
        for v in paper_authors.values():
            if v.get("first_id"): needed_aids.add(v["first_id"])
            if v.get("last_id"):  needed_aids.add(v["last_id"])

    remaining = [aid for aid in needed_aids if aid not in done_set]
    log.info("Phase 2b (Tier B): %d unique authors, %d remaining (~%.0f min at %.1f s/call)",
             len(needed_aids), len(remaining), len(remaining) * REQ_DELAY / 60, REQ_DELAY)

    t_start = time.time()
    for i, aid in enumerate(remaining):
        if aid in done_set: continue
        papers_all = []
        offset = 0
        while True:
            resp = s2_get(f"/author/{aid}/papers",
                          params={"fields": "year,publicationDate,externalIds,paperId",
                                  "limit": AUTHOR_PAGE_LIMIT, "offset": offset})
            if not resp:
                break
            batch = resp.get("data", []) or []
            papers_all.extend([_norm_paper_rec(p) for p in batch])
            if len(batch) < AUTHOR_PAGE_LIMIT:
                break
            offset += AUTHOR_PAGE_LIMIT
        author_papers[aid] = papers_all
        done_set.add(aid)

        # Checkpoint every ~100 authors so a kill/restart continues rather than restarts
        if (i + 1) % 100 == 0:
            elapsed  = time.time() - t_start
            rate     = (i + 1) / elapsed if elapsed > 0 else 0
            eta_sec  = (len(remaining) - (i + 1)) / rate if rate > 0 else float("inf")
            eta_min  = eta_sec / 60
            log.info("    Tier B: %d/%d authors done  (elapsed %.1f min, ETA ~%.0f min)",
                     i + 1, len(remaining), elapsed / 60, eta_min)
            save_json(CKPT_AUTHOR_PAPERS, author_papers)
            save_json(CKPT_AUTH_HIST_DONE, list(done_set))

    # Final checkpoint flush
    save_json(CKPT_AUTHOR_PAPERS, author_papers)
    save_json(CKPT_AUTH_HIST_DONE, list(done_set))
    log.info("Phase 2b done: %d authors fetched", len(author_papers))
    return author_papers

# ---------------------------------------------------------------------------
# Phase 3a: Compute prestige features from Tier A (paperCount only)
# ---------------------------------------------------------------------------
def _nanmax(a, b):
    if np.isnan(a) and np.isnan(b): return np.nan
    if np.isnan(a): return b
    if np.isnan(b): return a
    return max(a, b)

def phase3a_compute(df: pd.DataFrame, paper_authors: dict, author_info: dict) -> pd.DataFrame:
    """Compute prestige from Tier A (current paper count per author)."""
    records = []
    for _, row in df.iterrows():
        arxiv_id   = row["arxiv_id_clean"]
        ss_pid     = str(row["ss_paper_id"]) if pd.notna(row["ss_paper_id"]) else None
        anchor_str = row["anchor_date_str"]
        anchor_src = row["anchor_date_used"]
        resolved   = False

        first_cnt = np.nan; last_cnt = np.nan

        if ss_pid and ss_pid in paper_authors:
            auth_rec = paper_authors[ss_pid]
            fid = auth_rec.get("first_id")
            lid = auth_rec.get("last_id")
            if fid and fid in author_info:
                cnt = author_info[fid].get("paperCount")
                first_cnt = float(cnt) if cnt is not None else np.nan
                resolved = True
            if lid and lid in author_info:
                cnt = author_info[lid].get("paperCount")
                last_cnt = float(cnt) if cnt is not None else np.nan
                resolved = True

        records.append({
            "arxiv_id_clean":                   arxiv_id,
            "anchor_date_used":                 anchor_str,
            "anchor_date_source":               anchor_src,
            "first_author_papercount_cur2026":  first_cnt,  # TIER A: current (2026) career total, NOT prior
            "last_author_papercount_cur2026":   last_cnt,
            "max_papercount_cur2026":           _nanmax(first_cnt, last_cnt),
            "max_years_active":                 np.nan,     # requires Tier B (see prepub_prestige_tierB.csv)
            "prestige_resolved":                resolved,
            "prestige_tier":                    "A" if resolved else "unresolved",
        })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Phase 3b: Compute prestige features from Tier B (strict prior count)
# ---------------------------------------------------------------------------
def parse_date(s) -> date | None:
    if not s or (isinstance(s, float) and math.isnan(s)):
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

def prior_papers_and_years(author_hist: list, anchor_dt: date, anchor_year: int,
                            exclude_arxiv: str | None, exclude_pid: str | None):
    """
    Compute strict prior_paper_count, years_active, and first_year from author's paper history.

    Returns (prior_count, years_active, first_year) where:
      prior_count  = # papers with publicationDate STRICTLY before anchor_dt
                     (year-only fallback: year < anchor_year; same-year dropped)
      years_active = anchor_year - min(author paper year), clamped >= 0
      first_year   = earliest year in the author's complete paper history (incl. focal paper)
    """
    if not author_hist:
        return np.nan, np.nan, np.nan
    prior_count = 0
    min_year    = None
    for rec in author_hist:
        r_arxiv = rec.get("arxivId")
        r_pid   = rec.get("paperId")
        if exclude_arxiv and r_arxiv and str(r_arxiv).strip() == str(exclude_arxiv).strip():
            continue
        if exclude_pid and r_pid and str(r_pid).strip() == str(exclude_pid).strip():
            continue
        r_year     = rec.get("year")
        r_date_str = rec.get("publicationDate")
        r_date     = parse_date(r_date_str)
        if r_year is not None:
            try:
                ry = int(r_year)
                if min_year is None or ry < min_year:
                    min_year = ry
            except (ValueError, TypeError):
                pass
        if r_date is not None:
            if r_date < anchor_dt:
                prior_count += 1
        elif r_year is not None:
            try:
                if int(r_year) < anchor_year:
                    prior_count += 1
            except (ValueError, TypeError):
                pass
    years_active = np.nan
    if min_year is not None and anchor_year is not None:
        years_active = max(0, int(anchor_year) - int(min_year))
    first_year = min_year  # None → NaN coerced by pandas on DataFrame construction
    return prior_count, years_active, first_year

def phase3b_write_tierB_csv(df: pd.DataFrame, paper_authors: dict,
                             author_papers: dict, out_path: Path) -> int:
    """
    Build the Tier B output (strict prior counts) and write it atomically to out_path.

    Output columns (one row per paper in df):
        arxiv_id_clean, first_author_prior_papers_true, last_author_prior_papers_true,
        max_prior_papers_true, max_years_active, first_author_first_year,
        last_author_first_year, tierB_resolved

    Does NOT touch the interim Tier-A file (OUT_CSV).
    Returns number of papers with at least one author resolved (tierB_resolved=True).
    """
    records = []
    resolved_count = 0

    for _, row in df.iterrows():
        arxiv_id  = row["arxiv_id_clean"]
        anchor_str = row.get("anchor_date_str") or row.get("anchor_date_used")
        anchor_d   = parse_date(anchor_str)
        sub_year   = int(row["submission_year"]) if pd.notna(row.get("submission_year")) else None
        ss_pid     = str(row["ss_paper_id"]) if pd.notna(row.get("ss_paper_id")) else None

        first_prior = np.nan; last_prior  = np.nan
        first_yrs   = np.nan; last_yrs    = np.nan
        first_fy    = np.nan; last_fy     = np.nan
        resolved    = False

        if anchor_d is not None and sub_year is not None and ss_pid and ss_pid in paper_authors:
            auth_rec = paper_authors[ss_pid]
            fid = auth_rec.get("first_id")
            lid = auth_rec.get("last_id")
            if fid and fid in author_papers:
                first_prior, first_yrs, first_fy = prior_papers_and_years(
                    author_papers[fid], anchor_d, sub_year, arxiv_id, ss_pid)
                resolved = True
            if lid and lid in author_papers:
                last_prior, last_yrs, last_fy = prior_papers_and_years(
                    author_papers[lid], anchor_d, sub_year, arxiv_id, ss_pid)
                resolved = True

        if resolved:
            resolved_count += 1

        records.append({
            "arxiv_id_clean":                  arxiv_id,
            "first_author_prior_papers_true":  first_prior,
            "last_author_prior_papers_true":   last_prior,
            "max_prior_papers_true":           _nanmax(first_prior, last_prior),
            "max_years_active":                _nanmax(first_yrs, last_yrs),
            "first_author_first_year":         first_fy,
            "last_author_first_year":          last_fy,
            "tierB_resolved":                  resolved,
        })

    tierB_df = pd.DataFrame(records)
    # Ensure arxiv_id_clean stays string (never cast to float)
    tierB_df["arxiv_id_clean"] = tierB_df["arxiv_id_clean"].astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tierB_df.to_csv(tmp, index=False)
    tmp.replace(out_path)

    log.info("Tier B CSV written: %s  (%d rows, %d tierB_resolved)",
             out_path, len(tierB_df), resolved_count)
    return resolved_count

# ---------------------------------------------------------------------------
# Winsorise + leak correlation
# ---------------------------------------------------------------------------
def winsorise_99(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return series.rename(series.name + "_w99")
    q99 = s.quantile(0.99)
    return series.clip(upper=q99).rename(series.name + "_w99")

def compute_leak_correlation(out: pd.DataFrame, df_orig: pd.DataFrame):
    merged = out.merge(df_orig[["arxiv_id_clean", "max_hindex", "last_author_hindex"]],
                       on="arxiv_id_clean", how="left")
    leakfree = ["max_papercount_cur2026", "max_years_active"]
    leaky    = ["max_hindex", "last_author_hindex"]
    corrs = {}
    cov   = merged[merged["prestige_resolved"]].dropna(subset=["max_papercount_cur2026", "max_hindex"])
    n     = len(cov)
    log.info("Leak correlation on %d papers with both measures", n)
    for lf in leakfree:
        for lk in leaky:
            pair = merged.dropna(subset=[lf, lk])[[lf, lk]]
            if len(pair) < 30:
                corrs[f"{lf}_vs_{lk}"] = {"n": len(pair), "rho": None, "pval": None}
                continue
            rho, pval = stats.spearmanr(pair[lf], pair[lk])
            corrs[f"{lf}_vs_{lk}"] = {"n": len(pair), "rho": round(float(rho), 4),
                                       "pval": float(pval)}
            log.info("  Spearman(%s, %s) = %.4f (n=%d, p=%.2e)", lf, lk, rho, len(pair), pval)
    return corrs, n

# ---------------------------------------------------------------------------
# Stratified sample helper
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n_target: int = 2500, seed: int = 42) -> pd.DataFrame:
    df2 = df[df["ss_paper_id"].notna()].copy()
    strat = df2.groupby(["release_year", "subfield"], group_keys=False).apply(
        lambda g: g.sample(frac=min(1.0, n_target / len(df2)), random_state=seed)
    )
    if len(strat) < n_target:
        extra = df2[~df2["arxiv_id_clean"].isin(strat["arxiv_id_clean"])].sample(
            min(n_target - len(strat), len(df2) - len(strat)), random_state=seed+1)
        strat = pd.concat([strat, extra])
    return strat.iloc[:n_target]

# ---------------------------------------------------------------------------
# Write NOTE.md
# ---------------------------------------------------------------------------
def write_note(out: pd.DataFrame, df_orig: pd.DataFrame, corrs: dict, n_corr: int,
               tier_b_papers: int, total_n: int, duration_sec: float):
    resolved_n    = int(out["prestige_resolved"].sum())
    tier_b_n      = int((out["prestige_tier"] == "B").sum())
    tier_a_n      = int((out["prestige_tier"] == "A").sum())
    coverage      = 100 * resolved_n / len(out)
    pv1_n         = int(out["anchor_date_source"].eq("published_v1").sum())
    pv1_pct       = 100 * pv1_n / len(out)

    corr_lines = []
    for k, v in corrs.items():
        lf, _, lk = k.partition("_vs_")
        if v["rho"] is not None:
            corr_lines.append(
                f"- Spearman({lf}, {lk}) = {v['rho']:.4f} (n={v['n']}, p={v['pval']:.2e})"
            )
        else:
            corr_lines.append(f"- Spearman({lf}, {lk}) = N/A (n={v['n']} — too few)")

    md = f"""# prepub_prestige build notes

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Script: `scripts/22_prepub_prestige.py`

## Anchor date choice

Per-paper anchor = `published_v1` (arXiv v1 date from Spike 1 `clean_dates.csv`) if
present, else `release_dt` (HF ingestion timestamp, ≈v1 for 99.8% of covered papers
per the analysis plan).  Using `release_dt` as fallback is a documented approximation; for
most papers it over-estimates time-since-submission by at most a few hours.

- `published_v1` used for {pv1_n:,} / {len(out):,} papers ({pv1_pct:.1f}%)
- `release_dt` fallback for {len(out) - pv1_n:,} papers

## Coverage and prestige tier

- Total papers: {len(out):,}
- `prestige_resolved = True`: {resolved_n:,} ({coverage:.1f}%)
- Tier A (current paper count, all S2 papers): {tier_a_n:,} papers
- Tier B (strict prior count via per-author pagination): {tier_b_n:,} papers
- Unresolved (no S2 paper ID or no author match): {len(out) - resolved_n:,}

## Feature definitions

### Tier A features (default; all S2 papers; computed this run)

`first/last_author_papercount_cur2026` = **current (2026) career total paper count**
(paperCount field) from the Semantic Scholar author batch endpoint.

**IMPORTANT — ANACHRONISTIC FEATURE**: These are each author's total career paper count
as measured today (2026), NOT the count of papers published before the paper's anchor
date.  A 2023 paper's author count therefore includes papers they published in 2024–2026,
making this measure anachronistic (forward-looking) relative to submission.  It is an
UPPER BOUND on the true prior-to-submission paper count.

This feature is **usable only as an interim/conservative-robustness control** — its
direction of bias is known (over-counts productivity), and the Spearman rank correlation
with true prior paper count is expected to be very high (≥ 0.95) because prolific
researchers at time T are also prolific before T.  It is NOT a post-outcome variable:
`paperCount` is a raw publication count, not a function of citations.

`max_papercount_cur2026` = max(first_author_papercount_cur2026, last_author_papercount_cur2026).
`max_years_active` = NaN for all papers in this interim build (requires Tier B).

The leakage-free, strictly-prior counts (`*_prior_papers_true`) and `years_active` come
from Tier B and are written to a **separate file** (`prepub_prestige_tierB.csv`) so they
never overwrite this interim build.

### Tier B features (strict; produced by --per-author mode, written to prepub_prestige_tierB.csv)

When the script is run with `--per-author`, it fetches each unique author's complete
paper list via GET /author/{{author_id}}/papers (paginated, 1,000/page) and writes results
to `data/processed/prepub_prestige_tierB.csv` (separate from this interim file).
For those authors:

- `first/last_author_prior_papers_true` = # papers with `publicationDate` STRICTLY before
  anchor_date, excluding paper i itself (matched by ArXiv id or S2 paper id).
  For year-only records: count only `year < submission_year` (same-year papers dropped).
  Papers with null date/year are dropped from the count (not set to 0).
- `max_prior_papers_true` = max(first_author_prior_papers_true, last_author_prior_papers_true).
- `max_years_active` = submission_year − min(author paper year), clamped ≥ 0.
  (For year-only records the min year is the earliest year field.)
- `first/last_author_first_year` = earliest year found in the author's paper history.
- `tierB_resolved` = True if at least one author's history was fetched.

At keyless rate (~1.5 req/sec with backoffs), 15,482 unique authors × 1.5 s ≈ 6.5 hrs.
Re-run with `--per-author` overnight to backfill; checkpointing is incremental (every
~100 authors).

## As-of-date h-index infeasibility statement

Reconstructing each author's h-index as of the paper's submission date would require
knowing citation counts for each of their prior papers AS OF that date.  The S2 API
provides only current citation counts.  The historical citation-count API would require
O(authors × prior_papers) requests — infeasible at keyless rate limits.  Therefore
**no as-of-date h-index is constructed**.  The Tier A/B features are the leakage-free
fallback described in the analysis plan

The existing `max_hindex` / `last_author_hindex` columns in `papers_v2.csv` reflect
each author's h-index as measured today (after all downstream citations have accrued),
making them post-outcome variables.  They MUST NOT be used as pre-publication controls.

## Leak quantification

Spearman correlations between the `max_papercount_cur2026` interim feature and the leaky
`max_hindex` measure (on {n_corr:,} overlapping papers):

{chr(10).join(corr_lines) if corr_lines else '(no correlation data — empty intersection)'}

A moderate–strong positive correlation is expected: prolific authors tend to accumulate
higher h-indices over time.  However, the ranking is not identical — the leaky h-index
depends on citation trajectories (including post-treatment citations), while
`max_papercount_cur2026` does not.

## Build metadata

- API: Semantic Scholar Graph API (keyless)
- Rate: {REQ_DELAY} s/request base, with exponential backoff on 429
- Wall time this run: {duration_sec/60:.1f} min
- Checkpoint dir: `data/raw/prestige_ckpt/`
- Phase 1: POST /paper/batch, 500 ids/call, fields=authors.authorId,authors.name
- Phase 2 (Tier A): POST /author/batch, 500 authors/call, fields=paperCount,name
- Phase 2b (Tier B, pending): GET /author/{{id}}/papers, 1000 papers/page
"""
    NOTE_MD.write_text(md)
    log.info("Wrote %s", NOTE_MD)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-author", action="store_true",
                        help="Run Tier B: per-author paginated paper history (slow, resumable)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Stratified sample of N papers for Tier B (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and plan, but make no API calls")
    args = parser.parse_args()

    t0 = time.time()
    df = load_inputs()

    if args.dry_run:
        n_ss = int(df["ss_paper_id"].notna().sum())
        log.info("DRY RUN — no API calls.")
        log.info("Papers with ss_paper_id: %d → %d paper batch calls", n_ss, math.ceil(n_ss/500))
        log.info("Estimated unique authors: ~15k → ~31 author batch calls (Tier A)")
        return

    # ---------- Phase 1 ----------
    paper_authors = phase1_paper_authors(df)

    # ---------- Phase 2a (Tier A always) ----------
    author_info = phase2_author_info(paper_authors)

    # ---------- Phase 3a: compute from Tier A ----------
    log.info("Phase 3: computing prestige features (Tier A)")
    out = phase3a_compute(df, paper_authors, author_info)

    # ---------- Phase 2b + 3b (Tier B, optional) ----------
    # NOTE: Tier B writes to OUT_CSV_TIERB (prepub_prestige_tierB.csv) — a SEPARATE file.
    # It never overwrites the Tier-A interim file (OUT_CSV / prepub_prestige.csv).
    if args.per_author:
        log.info("Tier B mode: fetching per-author paper histories → %s", OUT_CSV_TIERB)
        if args.sample > 0:
            sdf = stratified_sample(df, n_target=args.sample)
            sample_pids = set(sdf["ss_paper_id"].dropna().astype(str))
            log.info("  Stratified sample: %d papers → ~%d unique authors",
                     len(sdf), len(sample_pids) * 2)
        else:
            sample_pids = None

        author_papers = phase2b_per_author(paper_authors, sample_paper_ids=sample_pids)
        log.info("Phase 3b: building Tier B CSV (strict prior counts) → %s", OUT_CSV_TIERB)
        phase3b_write_tierB_csv(df, paper_authors, author_papers, OUT_CSV_TIERB)
        # The interim Tier-A file (OUT_CSV) is NOT modified; proceed to write it below.
    else:
        log.info("Tier B not requested (Tier A only; run with --per-author to write %s)", OUT_CSV_TIERB)

    # ---------- Winsorise ----------
    for col in ["max_papercount_cur2026", "max_years_active",
                "first_author_papercount_cur2026", "last_author_papercount_cur2026"]:
        w = winsorise_99(out[col])
        out[w.name] = w

    # ---------- Save ----------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    log.info("Saved %s  (%d rows, %d cols)", OUT_CSV, len(out), len(out.columns))

    # ---------- Leak correlation ----------
    corrs, n_corr = compute_leak_correlation(out, df)

    # ---------- NOTE ----------
    duration   = time.time() - t0
    tier_b_n   = int((out["prestige_tier"] == "B").sum())
    write_note(out, df, corrs, n_corr, tier_b_n, len(df), duration)

    # ---------- Summary ----------
    resolved_n = int(out["prestige_resolved"].sum())
    coverage   = 100 * resolved_n / len(out)
    log.info("=" * 60)
    log.info("DONE in %.1f min", duration / 60)
    log.info("Coverage: %d / %d resolved (%.1f%%)", resolved_n, len(out), coverage)
    log.info("Tier A (current paperCount): %d | Tier B (strict prior): %d",
             int((out["prestige_tier"] == "A").sum()), tier_b_n)
    log.info("Output: %s", OUT_CSV)
    for k, v in corrs.items():
        if v["rho"] is not None:
            log.info("Leak corr %s: rho=%.4f (n=%d)", k, v["rho"], v["n"])
    if args.per_author:
        log.info("Tier B output: %s", OUT_CSV_TIERB)

if __name__ == "__main__":
    main()
