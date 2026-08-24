"""
01_collect_hf_daily.py
-----------------------
Collect the community-attention signal from the Hugging Face Daily Papers API.

For each calendar date in the requested range, the endpoint
    https://huggingface.co/api/daily_papers?date=YYYY-MM-DD
returns the list of papers that "trended" (were featured) that day, each with an
upvote count, number of comments, the submitter, and the linked arXiv id.

Output: data/raw/hf_daily_papers.csv  (one row per (paper, trending-day))

This is the day-of-release community attention signal described in the proposal.
"""
import requests
import pandas as pd
import time
import sys
from datetime import date, timedelta
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hf_daily_papers.csv")
BASE = "https://huggingface.co/api/daily_papers"
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}

# Date range: ~2 years of submissions. Today is 2026-06-05, so even the most
# recent papers here are >=5 months old, giving them time to accumulate citations.
START = date(2024, 1, 1)
END = date(2025, 12, 31)


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(d, session):
    """Return list of paper-records for a single date, or [] on failure."""
    for attempt in range(3):
        try:
            r = session.get(BASE, params={"date": d.isoformat()}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return []
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return []


def flatten(rec, trend_date):
    """Pull the fields we care about out of one HF daily-paper record."""
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
    n = len(days)
    for i, d in enumerate(days):
        recs = fetch_day(d, session)
        for rec in recs:
            row = flatten(rec, d)
            if row["arxiv_id"]:
                rows.append(row)
        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {d.isoformat()}  cumulative rows={len(rows)}", flush=True)
            # checkpoint periodically so a crash doesn't lose progress
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(0.25)  # be polite

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"DONE. {len(df)} (paper,day) rows -> {OUT}")
    print(f"Unique arxiv ids: {df['arxiv_id'].nunique()}")


if __name__ == "__main__":
    main()
