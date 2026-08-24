"""
09_repair_outcomes.py
---------------------
v2 (D4 revised): targeted repair of implausible Semantic Scholar records using the
S2 title-match endpoint (which can land on the *published* version of a paper when
the arXiv-id record is stale or split).

Two target groups:
  (a) ss_found == 0            -> try to recover the paper by HF title match
  (b) citation_count == 0 and paper is >= 10 months old
                               -> a 0 for an old HF-trending paper is suspicious
                                  (session-1 bug: e.g. "Differential Transformer")

Accept a match only when the normalized titles agree (exact or near-exact), and only
ever *raise* the citation count (never lower it).

Output: data/raw/s2_repairs.csv  (arxiv_id_clean, repaired fields, match evidence)
"""
import requests
import pandas as pd
import numpy as np
import time
import os
import re

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(RAW, "s2_repairs.csv")
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "title,citationCount,influentialCitationCount,referenceCount,externalIds,publicationDate"
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}
SNAPSHOT = pd.Timestamp("2026-06-11")


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def title_sim(a, b):
    """Token Jaccard on normalized titles."""
    sa, sb = set(norm(a).split()), set(norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def clean_id(x):
    x = str(x).strip()
    return x.split("v")[0] if "v" in x.split(".")[-1] else x


def load_targets():
    ss = pd.read_csv(os.path.join(RAW, "semantic_scholar_v2.csv"), dtype={"arxiv_id_clean": str})
    # HF titles + release dates for age
    hf_parts = []
    for f in ["hf_daily_papers.csv", "hf_daily_papers_2023.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            h = pd.read_csv(p, dtype={"arxiv_id": str})
            h["arxiv_id_clean"] = h["arxiv_id"].map(clean_id)
            hf_parts.append(h[["arxiv_id_clean", "hf_title", "published_at"]])
    hf = pd.concat(hf_parts).drop_duplicates("arxiv_id_clean")
    df = ss.merge(hf, on="arxiv_id_clean", how="left")
    rel = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["age_months"] = (SNAPSHOT - rel).dt.days / 30.44

    grp_a = df[(df["ss_found"] == 0) & df["hf_title"].notna()]
    grp_b = df[(df["ss_found"] == 1) & (df["citation_count"] == 0) & (df["age_months"] >= 10)]
    print(f"targets: not-found={len(grp_a)}, suspicious-zero={len(grp_b)}")
    return pd.concat([grp_a.assign(reason="not_found"),
                      grp_b.assign(reason="zero_old")])


def match_one(title, session):
    try:
        r = session.get(MATCH_URL, params={"query": title[:300], "fields": FIELDS}, timeout=30)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return data[0] if data else None
        if r.status_code == 429:
            time.sleep(10)
        return None
    except Exception:
        time.sleep(5)
        return None


def main():
    targets = load_targets()
    session = requests.Session()
    session.headers.update(HEADERS)
    rows = []
    n = len(targets)
    for i, (_, t) in enumerate(targets.iterrows()):
        title = t["hf_title"] if pd.notna(t["hf_title"]) else t.get("ss_title")
        if not isinstance(title, str) or len(title) < 8:
            continue
        m = match_one(title, session)
        if m:
            sim = title_sim(title, m.get("title", ""))
            new_cit = m.get("citationCount")
            if sim >= 0.85 and new_cit is not None:
                old = t.get("citation_count")
                if pd.isna(old) or new_cit > old:
                    rows.append({
                        "arxiv_id_clean": t["arxiv_id_clean"],
                        "reason": t["reason"],
                        "old_citations": old,
                        "repaired_citations": new_cit,
                        "repaired_influential": m.get("influentialCitationCount"),
                        "repaired_references": m.get("referenceCount"),
                        "matched_title": m.get("title"),
                        "title_sim": round(sim, 3),
                    })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] repairs so far: {len(rows)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(1.4)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"DONE. {len(rows)} repairs out of {n} targets -> {OUT}")


if __name__ == "__main__":
    main()
