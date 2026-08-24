"""
02_enrich_semantic_scholar.py
-----------------------------
Outcome variable: scholarly impact (citation count) from the Semantic Scholar
Graph API, plus a few extra fields used as controls / features.

Uses the batch endpoint (up to 500 ids per POST):
    POST https://api.semanticscholar.org/graph/v1/paper/batch
We query by "arXiv:<id>" so the join key is the arXiv id.

Output: data/raw/semantic_scholar.csv  (one row per unique arxiv id)
"""
import requests
import pandas as pd
import time
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HF = os.path.join(RAW, "hf_daily_papers.csv")
OUT = os.path.join(RAW, "semantic_scholar.csv")

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = ("title,citationCount,influentialCitationCount,year,referenceCount,"
          "fieldsOfStudy,publicationVenue,externalIds,publicationDate,isOpenAccess")
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}


def clean_arxiv_id(x):
    """Strip a trailing version suffix (2501.08313v2 -> 2501.08313)."""
    x = str(x).strip()
    if "v" in x.split(".")[-1]:
        # only strip if it looks like NNNN.NNNNNvN
        base = x.split("v")[0]
        return base
    return x


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_batch(ids, session):
    body = {"ids": [f"arXiv:{i}" for i in ids]}
    for attempt in range(5):
        try:
            r = session.post(BATCH_URL, params={"fields": FIELDS}, json=body, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 504, 503):
                time.sleep(3 * (attempt + 1))
                continue
            print(f"   batch status {r.status_code}: {r.text[:120]}")
            time.sleep(2)
        except Exception as e:
            print(f"   exception {e}")
            time.sleep(3 * (attempt + 1))
    return [None] * len(ids)


def main():
    hf = pd.read_csv(HF)
    hf["arxiv_id_clean"] = hf["arxiv_id"].map(clean_arxiv_id)
    ids = sorted(hf["arxiv_id_clean"].dropna().unique().tolist())
    print(f"Fetching citations for {len(ids)} unique arXiv ids")

    session = requests.Session()
    session.headers.update(HEADERS)
    rows = []
    BATCH = 100  # conservative to stay under public rate limits
    for bi, batch in enumerate(chunks(ids, BATCH)):
        res = fetch_batch(batch, session)
        for qid, p in zip(batch, res):
            if not p:
                rows.append({"arxiv_id_clean": qid, "ss_found": 0})
                continue
            venue = p.get("publicationVenue") or {}
            fos = p.get("fieldsOfStudy") or []
            rows.append({
                "arxiv_id_clean": qid,
                "ss_found": 1,
                "ss_paper_id": p.get("paperId"),
                "citation_count": p.get("citationCount"),
                "influential_citations": p.get("influentialCitationCount"),
                "reference_count": p.get("referenceCount"),
                "ss_year": p.get("year"),
                "ss_pub_date": p.get("publicationDate"),
                "fields_of_study": "; ".join(fos),
                "venue_name": venue.get("name"),
                "venue_type": venue.get("type"),
                "is_open_access": p.get("isOpenAccess"),
                "ss_title": p.get("title"),
            })
        if (bi + 1) % 5 == 0:
            print(f"  batch {bi+1}: {len(rows)} rows so far", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(1.1)  # ~1 req/sec, polite for the public (keyless) endpoint

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    found = int(out["ss_found"].sum())
    print(f"DONE. {len(out)} ids, {found} found on Semantic Scholar -> {OUT}")


if __name__ == "__main__":
    main()
