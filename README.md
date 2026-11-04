# County Flood & Hurricane Risk

**[Open the dashboard →](https://ganeshpinnu.github.io/county-flood-hurricane-risk/)**

Pick any county in the lower 48 and see the live 4-day weather forecast, any
National Weather Service warning in force right now, and — across the Gulf
Coast, where the models are trained and validated — modelled flood and
hurricane risk for a given day alongside the real federal disasters that have
hit that county before.

Presenting this? [PROJECT_EXPLAINER.docx](PROJECT_EXPLAINER.docx) is a
walkthrough of the method, the results and the limits, written for a reader
with no statistics background. Regenerate it with `python make_explainer.py`.

The dashboard is a static page. No server, no build step, no mapping library,
no API keys.

## The question

Given a county and a day, will a **federal flood or hurricane disaster
declaration begin there within the next 3 days?**

This target is deliberate. An "is a disaster active today" label is trivially
predictable from yesterday's label and would score well while being worthless.
Onset is the hard version of the question and the only one worth asking.

## What this is, and what it is not

It is **not a weather forecaster.** It does not predict rain. It consumes
rainfall — observed and forecast — and maps it to declaration risk.

It is **not a warning system.** Onset is rare: 0.073% of county-days for
floods, 0.224% for storms. Raw probabilities are correspondingly tiny, so the
dashboard shows a band ("Elevated", "High") derived from quantiles of 150,000
sampled county-days. "High" means this county-day is in the top 1% of all
county-days, not that a disaster is likely.

The label is a **bureaucratic event, not a physical one.** FEMA declarations
depend on damage estimates and state requests. A severe flood with no
declaration counts here as a negative.

## Results

Panel of **1,986,871 county-days** across 356 Gulf Coast counties.
Splits are contiguous blocks, never random — adjacent days share weather state,
so a shuffled split leaks across the boundary.

| | Train | Validate | Test |
|---|---|---|---|
| Years | 2010–2019 | 2020–2022 | 2023–2026 |

| Hazard | Model kept | Test AP | Base rate | Lift | vs best baseline |
|---|---|---|---|---|---|
| Flood | 1-feature logistic | 0.0565 | 0.073% | **77.1×** | 1.00× |
| Hurricane | XGBoost, monotone | 0.0397 | 0.224% | **17.7×** | **1.71×** |

Average precision and lift over prevalence are reported rather than accuracy or
ROC-AUC. At a 0.07% base rate a model that always answers "no" scores 99.93%
accuracy while being completely useless.

**The flood result is the interesting one.** Gradient boosting *lost* to a
single rainfall ratio (`ratio_fwd3` — forecast 3-day rain over that county's
own climatological normal), so the simpler model is the one kept. With ~1,500
training positives spread over 356 counties, the trees fit thresholds specific
to individual counties instead of learning a process. That is a result, not an
oversight.

### The differentiator: does it work where it has never been?

The usual way to report a score is to test on counties the model already
trained on. That measures memory, not skill. Here each state is removed from
training **entirely**, then scored — and compared against the pooled model on
identical rows, so prevalence cancels and the ratio is a clean retention
figure.

| Held-out state | Storm onsets | Skill retained |
|---|---|---|
| Mississippi | 99 | 92.8% |
| Texas | 132 | 87.7% |
| Alabama | 159 | 87.6% |
| Louisiana | 101 | 70.6% |
| Florida | 427 | 66.8% |
| **Weighted mean** | | **76.6%** |

**Hurricane skill retains 77% on unfamiliar ground — a 23% loss. Flood retains
100%.**

The flood number is not luck. A one-feature monotone model's *ranking* cannot
change when it is refitted, so it carries no local thresholds to lose. The
contrast is the whole point: gradient boosting buys 1.71× over the best
baseline at home and pays 23% of it back in territory it has never seen. The
simpler model buys nothing extra and loses nothing.

### Skill by test year

The same trained model, scored separately on each year it never saw. No
refitting. Flat is the result you want — it means the model learned a process
rather than a period. Years with fewer than 20 onsets are suppressed as noise.

| Year | Flood AP | Flood onsets | Hurricane AP | Hurricane onsets |
|---|---|---|---|---|
| 2023 | — | 3 | 0.0228 | 183 |
| 2024 | 0.0074 | 162 | 0.0629 | 684 |
| 2025 | 0.2106 | 78 | — | 21 |
| 2026 | 0.0690 | 57 | 0.0757 | 30 |

## Coverage

| | Counties |
|---|---|
| Drawn on the map | 3,109 (all of the lower 48) |
| Risk modelled | 356 |
| Live weather + NWS warnings | 3,109 |

Risk modelling is restricted to the five Gulf states, and within them to
counties with a rainfall station in the source data. Every other county is
drawn unshaded and says so when clicked. Weather and official warnings are
queried per-county at the county's area-weighted centroid and work nationwide.

---

## Repository layout

### The current pipeline

Run in this order. Each script writes what the next one reads.

| File | What it is | Why it exists |
|---|---|---|
| `config.py` | Shared paths, year splits, hazard type lists, climatology window. | One place to change a split or a data directory. The climatology baseline is frozen to 1990–2010 here so anomaly features for later years never see their own distribution — widening it leaks test information into training features. |
| `build_daily.py` | Builds the county-**day** panel: rainfall features at daily resolution, FEMA flood/storm onset labels, population density. Writes `artifacts/daily_panel.parquet`. | The earlier panel was county-*month*, which cannot answer "is there a warning today". This is the dataset everything downstream trains and scores on. |
| `proto_models.py` | Defines both models and, critically, the single place that knows how to *score* them. | Keeping scoring in one module means the training script, the transfer harness, the live fetcher and the dashboard export cannot drift apart. Fitting is split from scoring so the transfer test can refit without overwriting production artifacts. |
| `train_proto.py` | Fits both models, runs the Tier-0 baselines, writes `artifacts/proto_metrics.json`. | Baselines are reported *next to* the model on purpose. This is the file that proves gradient boosting lost to one feature on floods. |
| `transfer_proto.py` | Leave-one-state-out and per-year evaluation. Writes `artifacts/proto_transfer.json`. | The differentiator. Produces the only numbers in this project measured in territory the model has never seen. |
| `export_proto.py` | Builds the three JSON payloads the dashboard loads: `meta.json`, `events.json`, `history.json`. | Scoring every county-day would be a ~2M-entry payload. This ships only the windows a user actually opens — around real FEMA incidents, plus the most recent 60 observed days — which keeps the site small enough to serve statically. |
| `stage4_geo.py` | Downloads public county/state GeoJSON and projects it to SVG path strings. Writes `docs/data/geo.json`. | Projecting in Python instead of the browser means the dashboard needs no mapping library and no dependencies. Uses Albers equal-area conic so county areas stay comparable by eye. Guarantees every county keeps a clickable shape even when it falls under the sliver threshold at national zoom. |
| `fetch_live.py` | Fetches Open-Meteo weather per county and scores today plus the next 3 days. Writes `docs/data/live.json`. | Open-Meteo returns up to 92 past days alongside the forecast, so one request per county supplies both the antecedent windows and the forward-looking rain the models need. No API key. |
| `serve.py` | Local dev server on port 8787. | Identical to `http.server` except it tells the browser not to cache, so an edit to `app.js` shows up on reload instead of hours later. |

### The dashboard

| File | What it is | Why it exists |
|---|---|---|
| `docs/index.html` | Page structure: controls, map, county detail panel, model card. | GitHub Pages serves from `/docs`, so this directory *is* the site. |
| `docs/app.js` | All behaviour: map drawing, county selection, risk bands, live weather and NWS alert fetches, event history, model card rendering. | Plain JavaScript, no framework. Everything the page shows is either a shipped JSON file or a live keyless API call. |
| `docs/styles.css` | Styling and both colour themes. | Every colour the map uses is a CSS variable, so switching theme repaints the map without rebuilding any geometry. |
| `docs/data/geo.json` | 3,109 pre-projected county paths, 49 state outlines, county names and centroids. | The map. Centroids are area-weighted on each county's largest ring, so the weather query point lands *inside* the county — a bounding-box centre would not, for coastal shapes. |
| `docs/data/meta.json` | Model metrics, transfer results, risk bands, and the full county list flagged modelled / not modelled. | Lets the dashboard tell "outside the modelled region" apart from "modelled, but no data for this date". |
| `docs/data/history.json` | Model risk by date for the shipped windows. | What the map shades in Past-events mode. |
| `docs/data/events.json` | Real FEMA declarations per county, up to 12 each. | The clickable disaster history. Selecting one jumps the map to that date to show what the model would have said. |
| `docs/data/live.json` | Today + 3 days, scored from the live forecast. | What the map shades in Live-forecast mode. Regenerated daily by GitHub Actions. |
| `.github/workflows/refresh-live.yml` | Daily job that re-runs `fetch_live.py` and commits the result. | Without it the live view silently freezes on the day it was last run. |

### Supporting data and artifacts

| File | What it is | Why it exists |
|---|---|---|
| `refdata/station_county_mapping.csv` | Weather station → county lookup with coordinates. | Small and derived, so it is committed; the raw rainfall panel it came from is 25 MB and is not. |
| `artifacts/proto_flood.json` | The flood model: scaler, one coefficient, intercept. | Pure arithmetic — small enough to read by eye and verify. |
| `artifacts/proto_storm.json` | The trained XGBoost hurricane model. | Loaded by every script that scores storms. |
| `artifacts/proto_metrics.json` | Test metrics and all Tier-0 baseline scores. | The dashboard's model card reads this file rather than hard-coded numbers, so the page cannot disagree with the training run. |
| `artifacts/proto_transfer.json` | Per-state and per-year transfer results. | Same reason: the headline retention figure is read, not retyped. |
| `requirements.txt` | Python dependencies. | |
| `.gitignore` | Excludes the large parquet panels and raw upstream GeoJSON. | Those files are regenerable and not ours to republish. Everything the dashboard and the daily refresh need in a clean checkout is committed. |

### Documentation

| File | What it is | Why it exists |
|---|---|---|
| `PROJECT_EXPLAINER.docx` | Plain-language walkthrough of the method, results and limits. | For presenting the project without having built it. Every number is pulled from the artifacts, so the document cannot drift from the results. |
| `make_explainer.py` | Generates the `.docx` above. | Keeping it generated rather than hand-written is what makes the "cannot drift" guarantee true. |

### Earlier approach — kept for provenance

An earlier version asked a different question: *given that a county has just
exceeded its own 1-in-100-day rainfall threshold, will it be declared within 14
days?* That design was abandoned because county-month resolution cannot support
a daily warning dashboard. These files are retained so the path from there to
here is auditable, but **they are not part of the current pipeline** and are not
needed to reproduce the results above.

| File | What it was |
|---|---|
| `build_panel.py` | Built the county-**month** panel the earlier model used. |
| `stage2_feasibility.py` | Sample-size count run *before* any modelling. It is why the original design was abandoned: ~126 flood disasters over ~115 distinct dates is not enough to train on. |
| `stage2.py` | The conditional vulnerability model — given a hazard day, which counties were declared. |
| `stage3.py` | Platt calibration, operating points, leave-one-state-out for that model. |
| `stage4_export.py` | Froze that model to a standalone arithmetic artifact, verified against sklearn to 2e-9. |
| `stage4_live.py` | Live path for that model using NCEI GHCN-Daily and the NWS gridpoint API. |
| `baselines.py` | Tier-0 baselines, including month-of-year climatology. Climatology is not a strawman — `P(event │ county, month)` already encodes "Louisiana in September is dangerous". |
| `daily_model.py`, `train_xgb.py` | Early daily rainfall experiments that led to the current panel design. |
| `tft_colab.py`, `gen_notebook.py` | Temporal Fusion Transformer experiment and the generator for its Colab notebook. Predicts the rainfall *distribution* via quantile loss rather than a binary label. |
| `stage.ipynb`, `stage1-complete.ipynb` | Exploratory notebooks from the hazard stage. |
| `artifacts/stage2/`, `artifacts/stage3/`, `artifacts/stage4/` | Outputs of the above. |
| `docs/data/model.json`, `docs/data/replay.json` | Payloads the earlier dashboard loaded. |

---

## Reproducing

```bash
pip install -r requirements.txt

python build_daily.py       # county-day panel with onset labels
python train_proto.py       # fit both models, run Tier-0 baselines
python transfer_proto.py    # leave-one-state-out + per-year evaluation
python stage4_geo.py        # project county geometry to SVG paths
python export_proto.py      # write the dashboard payloads
python fetch_live.py        # score today + next 3 days from live weather

python serve.py             # http://localhost:8787
```

The raw rainfall panel and the FEMA/SVI source tables are not committed — they
are large and not ours to republish. `build_daily.py` expects them in the
directory named by `CLIMATE_DATA_DIR`; everything after that step runs from
committed files.

## Known limits

- **The label is a bureaucratic event.** FEMA declarations depend on damage
  estimates and state requests, not only on physical severity.
- **Training rainfall is observed, not forecast.** Features use station
  rainfall over the prediction window, which a real forecaster would not have.
  The honest fix is retraining on archived forecasts (GEFSv12 reforecast);
  until then the lead-time claim is not operationally sound. This is the
  single most important caveat on this page.
- **356 of 3,109 counties are modelled.** The held-out-state result above is
  the evidence for what extending coverage would cost.
- **No soil moisture, no river gauges.** Antecedent wetness is proxied by 30-
  and 90-day rainfall, a weak substitute for catchment state.
- **No hurricane wind field.** Storm risk is driven by rainfall and season;
  track, intensity and storm surge are absent entirely.
- **Small positive counts.** 300 flood and 918 storm onsets in the test period.
  Confidence intervals on every metric here are correspondingly wide, and three
  of the five states cannot support a state-level flood score at all.

## Data sources

NOAA NCEI GHCN-Daily · FEMA Disaster Declarations Summaries ·
CDC/ATSDR Social Vulnerability Index · Open-Meteo (forecast and ERA5 archive) ·
NWS api.weather.gov (active alerts)
