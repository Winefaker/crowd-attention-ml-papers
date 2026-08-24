"""
07_backfill_hf_2023.py
----------------------
v2 (D1): extend the HF Daily Papers sample backward to the feature's launch era,
2023-05-01 -> 2023-12-31. Same flattening logic as 01_collect_hf_daily.py.

Output: data/raw/hf_daily_papers_2023.csv
"""
import requests
import pandas as pd
import time
import os
from datetime import date, timedelta

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hf_daily_papers_2023.csv")
BASE = "https://huggingface.co/api/daily_papers"
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}

START = date(2023, 5, 1)
END = date(2023, 12, 31)


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(d, session):
    for attempt in range(3):
        try:
            r = session.get(BASE, params={"date": d.isoformat()}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return []
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def flatten(rec, trend_date):
    paper = rec.get("paper", {}) or {}
    authors = paper.get("authors", []) or []
    author_names = []
    for a in authors:
        if isinstance(a, dict):
            nm = a.get("name") or a.get("fullname")
            if nm:
                author_names.append(nm.strip())
    return {
        "arxiv_id": paper.get("id"),
        "trend_date": trend_date.isoformat(),
        "hf_title": (paper.get("title") or "").strip().replace("\n", " "),
        "upvotes": paper.get("upvotes"),
        "author_names": "; ".join(author_names),
        "num_comments": rec.get("numComments"),
        "github_stars": paper.get("githubStars"),
        "github_repo": paper.get("githubRepo"),
        "published_at": paper.get("publishedAt"),
        "submitted_on_daily_at": paper.get("submittedOnDailyAt"),
        "n_authors_hf": len(authors),
        "submitted_by": (rec.get("submittedBy") or {}).get("user")
        if isinstance(rec.get("submittedBy"), dict) else rec.get("submittedBy"),
        "is_author_participating": rec.get("isAuthorParticipating"),
        "ai_keywords": "; ".join(paper.get("ai_keywords", []) or []),
        "hf_summary": (paper.get("summary") or "").strip().replace("\n", " "),
    }


def main():
    session = requests.Session()
    session.headers.update(HEADERS)
    rows = []
    days = list(daterange(START, END))
    for i, d in enumerate(days):
        for rec in fetch_day(d, session):
            row = flatten(rec, d)
            if row["arxiv_id"]:
                rows.append(row)
        if (i + 1) % 25 == 0 or i == len(days) - 1:
            print(f"  [{i+1}/{len(days)}] {d.isoformat()}  rows={len(rows)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"DONE. {len(df)} rows, {df['arxiv_id'].nunique()} unique ids -> {OUT}")


if __name__ == "__main__":
    main()
