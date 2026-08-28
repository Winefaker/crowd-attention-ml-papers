# Can the crowd spot important ML papers?

Community attention, citation impact, and who the crowd misjudges. A study of 11344 machine learning
papers featured on Hugging Face Daily Papers, linking the upvotes a paper collected to the citations
it earned later.

**[Open the interactive dashboard](https://winefaker.github.io/crowd-attention-ml-papers/)**


Doubling upvotes goes with 1.74 times the later citations [1.67, 1.81] after prestige, subfield and
time controls. Forward in time, attention lifts top decile detection AUC from 0.723 to 0.816, a gain
of +0.093 [+0.072, +0.109] for the pre-specified logistic model and +0.078 [+0.050, +0.101] for
gradient boosting, over a baseline that passed a leakage audit. The crowd over rewards work from
newer teams and under rewards work from established ones. The link is predictive, not causal.

## What is here

| Path | What |
|---|---|
| `index.html` | The dashboard. One self contained file, no server and no external libraries. Open it from disk or from the link above. |
| `scripts/` | Every collection, modelling and figure script, numbered in the order they run. |
| `data/` | The assembled analysis file and the inputs for the analysis steps that run without re-collecting anything. |
| `results/` | Model output as JSON, the numbers ledger, and the per paper prediction scores. Files without a `_v3` in the name are the first pass, kept for history and marked as superseded. |
| `requirements.txt` | The packages the analysis needs. |
| `publish_dashboard.sh` | Helper that created this repository, and turns on Github Pages when it is public. Not needed to run the analysis. |

## The dashboard

Every featured paper is a point, placed by its attention percentile and its impact percentile, both
adjusted for age, subfield and release month. Violet is underrated (bottom third of attention, top
third of impact), green is overrated (the mirror case). Filter by subfield, release year, age,
prestige band or title, hover a point for the paper's details, click to open it on Arxiv. Below the
explorer are leaderboards of the most misjudged papers, a sortable table of every paper in the
current filter, and eight evidence panels.

## The pipeline

Run from the repository root with Python 3.12 (`pip install -r requirements.txt`). The headline
numbers come from these scripts in this order.

**Collection.** Needs network access to three public APIs that take no key. Set `CONTACT_EMAIL` first
if you want a contact address in the User-Agent, which is polite but optional.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_collect_hf_daily.py`, `07_backfill_hf_2023.py` | Query the HF Daily Papers feed once per calendar day over the feed's whole life. |
| 2 | `02_enrich_semantic_scholar.py`, `08_collect_s2_v2.py` | Citations, references and per author h-indexes from Semantic Scholar, in one uniform pass. |
| 3 | `03_enrich_arxiv.py` | Arxiv categories. |
| 4 | `09_repair_outcomes.py` | Repair the 113 outcomes that had migrated to a published version, by title match. |
| 5 | `10_collect_arxiv_control.py` | Draw the 3280 paper never trending control sample. |
| 6 | `11_merge_v2.py` | Merge the sources on the version stripped Arxiv identifier into `papers_v2.csv`. |
| 7 | `21_spike1_dates.py`, `22_prepub_prestige_fixed.py` | Clean the Arxiv v1 dates, then build author histories counting only papers published strictly before each focal paper. The second one is slow and resumable. |

**Analysis.** No network access needed. Steps 11 to 16 run straight from the files in `data/`. The
three steps marked † also read `papers_v2.csv` and the raw feed dumps, which are too large to ship,
so they need the collection steps above first.

| Step | Script | What it does |
|---|---|---|
| 8 † | `30_subfield_uniform.py` | One keyword rule set over every paper, giving the 13 subfields. |
| 9 † | `24_assemble.py` | Build `data/processed/analysis_final.csv`, 11344 papers by 65 columns. |
| 10 † | `23_crowding_cohort_v3.py` | Same day crowding variables for the instrument. |
| 11 | `25_crowding_iv_v3.py` | The crowding instrument, two stage least squares with Anderson Rubin intervals. |
| 12 | `26_prediction_v3.py` | The forward in time prediction test and the leakage audit. |
| 13 | `27_association_v3.py` | The association ladder, E-value, placebo, mixed model, matching, over and under rated groups. |
| 14 | `28_results_gate_v3.py` | Collect every number into `results/D5_results_gate_v3.md`. |
| 15 | `make_figures_v3.py` | Draw the analysis charts into `figures/`. |
| 16 | `build_dashboard.py` | Rebuild `index.html`. |

`20_spike0_pool_probe.py` and `21_spike1_dates.py` are the two feasibility checks that killed the
natural experiments: the API exposes no pool of submitted but not featured papers, and the 14 day
eligibility cutoff has essentially no papers on its far side. `scripts/superseded/` holds the earlier
versions of five scripts, kept so the changes behind the revised numbers can be traced. Scripts 04 to
06 and 12 to 17 are the first pass analysis.

One numbering wrinkle: the subfield taxonomy script keeps the number 30 because it was written last,
but it runs before `24_assemble.py`, as the order above shows.

## Reproducing the analysis

These steps run from a fresh clone, with no network access and nothing to collect.

```bash
pip install -r requirements.txt
python scripts/25_crowding_iv_v3.py    # seconds
python scripts/26_prediction_v3.py     # about 6 minutes
python scripts/27_association_v3.py    # about 7 minutes
python scripts/28_results_gate_v3.py
python scripts/make_figures_v3.py
python scripts/build_dashboard.py
```

Random seeds are fixed, so these reproduce the published numbers. The estimates, intervals and chart
data all come from the results files those scripts write. `results/D5_results_gate_v3.md` is the
numbers ledger: every headline number traces to a row in it, with the script and the JSON key that
produced it.

## Data

One row per paper, joined on the version stripped Arxiv identifier across three public APIs. Hugging
Face Daily Papers gives the attention signal and the sample frame. Semantic Scholar gives citations,
references and author records. Arxiv gives categories and the control sample.

`data/processed/analysis_final.csv` is the assembled analysis file, and `analysis_final_DICT.md`
describes every column. `data/processed/` also carries the leakage free prestige measure (the
`prepub_prestige_tierB.csv` file, where prior papers and years active are counted strictly before
each focal paper), the crowding variables, the over and under rated labels, the cleaned Arxiv dates,
and a four column text file with the titles and keywords the association script needs. Two inputs are
shipped ready made rather than rebuilt by a script here: `author_appear_prior.csv`, which counts how
often an author appeared in the feed before each paper, and `crowding.csv` from the first pass. The
raw API dumps are not in the repository because of their size, so the collection steps rebuild them.

Every field is public bibliographic metadata and no author name appears in any shipped file. The
author level fields are counts, the h-index, prior papers and years active, taken over a paper's
author list and stored against the paper. For the papers with a single author those counts are that
one author's, and the Arxiv identifier identifies the paper, so read them as the public Semantic
Scholar figures they are.

The over and under rated labels are descriptive residual groups, not judgements of a paper's quality.

## Licence

Code under the MIT licence, see `LICENSE`. The data files are derived from public APIs and are
included so the analysis can be checked.
