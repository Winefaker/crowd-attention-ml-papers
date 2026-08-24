"""
08_collect_s2_v2.py
-------------------
v2 (D2 + D3): one uniform Semantic Scholar pass over ALL HF papers (2023 backfill +
2024-25 main sample) fetching, per paper:

  - citation outcome (citationCount, influentialCitationCount) — snapshot 2026-06-11
  - referenceCount (placebo outcome), venue, publicationDate, isOpenAccess
  - the author list with hIndex / paperCount / citationCount  -> REAL prestige

Output: data/raw/semantic_scholar_v2.csv (one row per unique arXiv id)
The session-1 file (semantic_scholar.csv, snapshot 2026-06-05) is left untouched.
"""
import requests
import pandas as pd
import numpy as np
import time
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(RAW, "semantic_scholar_v2.csv")
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = ("title,citationCount,influentialCitationCount,referenceCount,year,"
          "publicationDate,isOpenAccess,publicationVenue,externalIds,"
          "authors.authorId,authors.name,authors.hIndex,authors.paperCount,authors.citationCount")
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}


def clean_id(x):
    x = str(x).strip()
    return x.split("v")[0] if "v" in x.split(".")[-1] else x


def load_ids():
    ids = set()
    for f in ["hf_daily_papers.csv", "hf_daily_papers_2023.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={"arxiv_id": str})
            ids |= set(df["arxiv_id"].dropna().map(clean_id))
    return sorted(ids)


def author_stats(authors):
    """Prestige features from the author list (h-index based)."""
    hs = [a.get("hIndex") for a in authors if a and a.get("hIndex") is not None]
    pcs = [a.get("paperCount") for a in authors if a and a.get("paperCount") is not None]
    out = {
        "n_authors_s2": len(authors),
        "max_hindex": max(hs) if hs else None,
        "mean_hindex": float(np.mean(hs)) if hs else None,
        "first_author_hindex": None,
        "last_author_hindex": None,
        "max_author_papers": max(pcs) if pcs else None,
    }
    if authors:
        fa, la = authors[0], authors[-1]
        out["first_author_hindex"] = fa.get("hIndex") if fa else None
        out["last_author_hindex"] = la.get("hIndex") if la else None
    return out


def main():
    ids = load_ids()
    print(f"Uniform v2 fetch for {len(ids)} unique arXiv ids (snapshot 2026-06-11)")
    rows = []
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        res = None
        for attempt in range(5):
            try:
                r = requests.post(BATCH_URL, params={"fields": FIELDS},
                                  json={"ids": [f"arXiv:{x}" for x in batch]},
                                  headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    res = r.json()
                    break
                time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(3 * (attempt + 1))
        if res is None:
            res = [None] * len(batch)
        for qid, p in zip(batch, res):
            if not p:
                rows.append({"arxiv_id_clean": qid, "ss_found": 0})
                continue
            venue = p.get("publicationVenue") or {}
            row = {
                "arxiv_id_clean": qid,
                "ss_found": 1,
                "ss_paper_id": p.get("paperId"),
                "citation_count": p.get("citationCount"),
                "influential_citations": p.get("influentialCitationCount"),
                "reference_count": p.get("referenceCount"),
                "ss_year": p.get("year"),
                "ss_pub_date": p.get("publicationDate"),
                "venue_name": venue.get("name"),
                "is_open_access": p.get("isOpenAccess"),
                "ss_title": p.get("title"),
            }
            row.update(author_stats(p.get("authors") or []))
            rows.append(row)
        if (i // 100 + 1) % 10 == 0:
            print(f"  {i + len(batch)}/{len(ids)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(1.1)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    found = int(out["ss_found"].sum())
    has_h = out["max_hindex"].notna().sum()
    print(f"DONE. {len(out)} ids | found {found} | with h-index {has_h} -> {OUT}")


if __name__ == "__main__":
    main()
