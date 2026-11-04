# County Flood & Hurricane Risk

**[Open the dashboard →](https://ganeshpinnu5.github.io/county-flood-hurricane-risk/)**

Pick any county in the lower 48 and see the live 4-day weather forecast, any
National Weather Service warning in force right now, and — across the Gulf
Coast, where the models are trained and validated — modelled flood and
hurricane risk for a given day alongside the flood episodes and cyclone passes
actually on record for it.

The full training write-up is
[V2_TRAINING_RESULTS.docx](V2_TRAINING_RESULTS.docx): target, features, splits,
baselines, both architectures, the lead sweep and the deviation scoring, every
number read out of the sealed run directory. Regenerate it with
`python make_report.py`.

The dashboard is a static page. No server, no build step, no mapping library,
no API keys.

## The question

Given a county and a day, will a **NOAA-recorded flood episode or a HURDAT2
cyclone pass begin there within the next 3 days?**

Two things about that target are deliberate.

**Onset, not occurrence.** An "is a flood active today" label is trivially
predictable from yesterday's label and would score well while being worthless.

**Observations, not declarations.** An earlier version of this project
predicted FEMA disaster declarations. A declaration is an administrative act:
it arrives weeks after the water does, it requires a governor to request it,
and plenty of real flooding never receives one. That model was partly
predicting paperwork. Flood days now come from NOAA StormEvents and cyclone
passes from the HURDAT2 best track, both of which record the hazard itself. The
change multiplied the flood training signal by a factor of fourteen — 1,517
positives became 20,766.

## What this is, and what it is not

It is **not a weather forecaster.** It does not predict rain. It consumes
rainfall — observed and forecast — and maps it to hazard-onset risk.

It is **not a warning system.** Onset is rare: 1.05% of county-days for floods,
0.38% for hurricanes. Raw probabilities are correspondingly small, so the
dashboard shows a band ("Elevated", "High") derived from quantiles of the
scored panel. "High" means this county-day ranks in the top 1% of county-days,
not that a hazard is likely.

## Results

Panel of **3,252,826 county-days** across 356 Gulf Coast counties, 2000–2026.
Splits are contiguous blocks, never random — adjacent days share weather state,
so a shuffled split leaks across the boundary.

| | Train | Validate | Test |
|---|---|---|---|
| Years | 2000–2014 | 2015–2019 | 2020–2026 |
| Rows | 1,867,821 | 619,985 | 765,020 |
| Flood onsets | 20,766 | 7,132 | 8,052 |
| Hurricane onsets | 4,306 | 1,600 | 2,244 |

Average precision and lift over prevalence are reported rather than accuracy or
ROC-AUC. At a 1% base rate a model that always answers "no" scores 99% accuracy
while being useless.

| Hazard | Model | Test AP | Base rate | Lift | vs best baseline |
|---|---|---|---|---|---|
| Flood | XGBoost, monotone | **0.2599** | 1.05% | 24.7× | **2.28×** |
| Flood | TFT, calibrated | 0.2350 | 1.05% | 22.2× | 2.06× |
| Hurricane | XGBoost, monotone | **0.1753** | 0.38% | 45.7× | **4.66×** |
| Hurricane | TFT, calibrated | 0.0541 | 0.38% | 14.1× | 1.44× |

The baselines are deliberately crude — one raw feature, or the historical rate
for the calendar month. On a rare-event target a crude predictor is much harder
to beat than it sounds: the best flood baseline is a single rainfall ratio
scoring 0.1141, which is 10.8× the base rate on its own.

The dashboard serves the XGBoost models, because they win both hazards. The
transformer's numbers are read from the run's results JSON rather than
re-scored in the browser, so the page and the document cannot diverge.

### Gradient boosting against the transformer

Both were trained on the same panel, the same splits and the same label. The
transformer sees a 30-day encoder window per county, is trained with a
positive-class weight of 50, and is then calibrated with isotonic regression
fitted on validation so its outputs can be read as probabilities. It is the
heavier model by a wide margin — 8.4 hours on an A100 against minutes of
boosting — and it loses on both hazards, badly on hurricanes.

### How much warning does it actually buy?

Reporting one number for a "1 to 3 day forecast" hides the thing a user needs
most. "Something starts in the next three days" is a much easier question than
"something starts three days from now", so a single pooled score quietly
flatters the longer leads. Every lead below therefore has its own model,
trained and scored on its own target.

| Hazard | Lead | Base rate | XGBoost AP | TFT AP |
|---|---|---|---|---|
| Flood | +1 day | 0.373% | **0.1718** | 0.1071 |
| Flood | +2 days | 0.371% | **0.1267** | 0.1259 |
| Flood | +3 days | 0.371% | **0.0749** | 0.0636 |
| Hurricane | +1 day | 0.128% | **0.0988** | 0.0541 |
| Hurricane | +2 days | 0.128% | **0.0721** | 0.0511 |
| Hurricane | +3 days | 0.128% | **0.0372** | 0.0343 |

Boosting decays monotonically with lead on both hazards, which is the expected
shape. The transformer is monotone on hurricanes but **not** on floods (0.1071,
0.1259, 0.0636). That inversion is reported rather than smoothed; it is a
reason to treat the transformer's per-lead numbers as noisier than the boosted
ones, not a finding about lead time.

The honest reading of this table: at +3 days the flood model scores 0.0749,
which is 20× its base rate but far below the pooled 0.2599. The pooled number
is the one a less careful write-up would have published as a three-day
forecast.

### Deviation from observation

Average precision ranks; it does not say what a user would experience at a
threshold. Scored at each hazard's 99th-percentile probability over the test
period:

| Hazard | Hits | Misses | False alarms | Detection rate | Precision | CSI | Brier skill |
|---|---|---|---|---|---|---|---|
| Flood | 2,517 | 5,483 | 5,056 | 31.5% | 33.2% | 0.193 | 0.133 |
| Hurricane | 867 | 1,353 | 4,926 | 39.1% | 15.0% | 0.121 | 0.094 |

Roughly one flood alarm in three is real and roughly one hurricane alarm in
seven. Both beat the base rate substantially and neither is a siren.

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

## Provenance

Every published number comes from one sealed training run,
`20261104T015759Z-run`, finished after 8.44 hours on an NVIDIA A100-SXM4-80GB
(xgboost 3.4.1, torch 2.11.0+cu130). The run is self-contained: no file outside
its own directory contributed to a reported number. It was trained on a panel
with sha256 `33a1cabae86575a1…`, and each of the eight boosters was reloaded and
confirmed to reproduce its recorded test average precision to five decimals
before export.

Runs are sealed for a reason. Earlier work wrote into a shared `artifacts/`
directory, and three times a result could not be traced back to the panel that
produced it. A run directory is hash-verified against its panel; a loose model
directory is whatever was written last.

Scoring a saved booster has one trap worth naming. Early stopping keeps
training for `early` more rounds after the best one, and `save_model` writes all
of them, so the file holds `best_iteration + early` trees — here the flood file
has 387 trees for a best iteration of 336. Predicting with the whole file
silently scores an overfit model. `v2_models.py` therefore passes
`iteration_range` explicitly, with `best_iteration` read from the run's results
JSON rather than from the file.

## Repository layout

### The pipeline

| File | What it is | Why it exists |
|---|---|---|
| `config.py` | Shared paths, year splits, hazard type lists, climatology window. | One place to change a split or a data directory. The climatology baseline is frozen to 1990–2010 so anomaly features for later years never see their own distribution — widening it leaks test information into training features. |
| `labels.py` | Builds flood onsets from NOAA StormEvents and cyclone passes from the HURDAT2 best track. | The observational label set. Keeping it separate from the panel build means the label definition can be audited on its own. |
| `build_daily.py` | Builds the county-**day** panel: rainfall features, hazard onset labels, population density, social vulnerability. Writes `artifacts/daily_panel.parquet`. | The dataset everything downstream trains and scores on. |
| `pipeline.py` | The whole training run: baselines, XGBoost, the TFT, the lead sweep, the deviation scoring, plots. Writes one sealed run directory. | A run is stamped with the panel hash, library versions and GPU, so a result can always be traced to its inputs. |
| `gen_pipeline_nb.py` | Generates `pipeline.ipynb` from `pipeline.py`. | The notebook is the Colab entry point. Generating it means the two cannot drift. |
| `archive_colab_run.py` | Pulls a finished Colab run directory back down intact. | A run that only exists on a disposable VM is not provenance. |
| `v2_models.py` | The single place that knows how to *load and score* the trained models. | Both consumers import from here, so the dashboard can never show risk from a model other than the one the paper reports. |
| `export_v2.py` | Builds the JSON payloads the dashboard loads: `meta.json`, `events.json`, `history.json`. | Scoring every county-day would be a 3M-entry payload. This ships only the windows a user actually opens. |
| `fetch_live.py` | Fetches Open-Meteo weather per county and scores today plus the next 3 days. Writes `docs/data/live.json`. | Open-Meteo returns up to 92 past days alongside the forecast, so one request per county supplies both the antecedent windows and the forward-looking rain. No API key. |
| `stage4_geo.py` | Downloads public county/state GeoJSON and projects it to SVG path strings. | Projecting in Python means the dashboard needs no mapping library. Albers equal-area conic, so county areas stay comparable by eye. |
| `make_report.py` | Generates `V2_TRAINING_RESULTS.docx` from the sealed run. | Keeping it generated is what makes "the document cannot drift from the results" true. |
| `hpc/` | Slurm batch scripts for the same pipeline on a cluster. | The GPU path that does not depend on a Colab session staying alive. |
| `serve.py` | Local dev server on port 8787. | Identical to `http.server` except it tells the browser not to cache. |

### The dashboard

| File | What it is |
|---|---|
| `docs/index.html` | Page structure: controls, map, county detail panel, model card. GitHub Pages serves from `/docs`, so this directory *is* the site. |
| `docs/app.js` | All behaviour: map drawing, county selection, risk bands, live weather and NWS alert fetches, event history, model card. Plain JavaScript, no framework. |
| `docs/styles.css` | Styling and both colour themes. Every map colour is a CSS variable, so switching theme repaints without rebuilding geometry. |
| `docs/data/geo.json` | 3,109 pre-projected county paths, 49 state outlines, names and centroids. Centroids are area-weighted on each county's largest ring, so the weather query point lands *inside* the county. |
| `docs/data/meta.json` | Model metrics, risk bands, coverage, and the full county list flagged modelled / not modelled. |
| `docs/data/history.json` | Model risk by date for the shipped windows. What the map shades in Past-events mode. |
| `docs/data/events.json` | Recorded flood episodes and cyclone passes per county, each paired with what the models said on the three days before onset. |
| `docs/data/live.json` | Today + 3 days, scored from the live forecast. Regenerated daily by GitHub Actions. |
| `.github/workflows/refresh-live.yml` | Daily job that re-runs `fetch_live.py` and commits the result. Without it the live view silently freezes on the day it was last run. |

### The sealed run

`artifacts/runs/20261104T015759Z-run/` holds the manifest, per-model results,
plots, console log, per-epoch TFT metrics and the eight XGBoost boosters. The
boosters ship because the daily refresh job scores with them from a clean
checkout. The transformer checkpoints do not — nothing the site serves reads
them, and the dashboard takes the TFT's numbers from the results JSON.

## Reproducing

```bash
pip install -r requirements.txt

python build_daily.py                   # county-day panel with onset labels
python pipeline.py                      # the full training run -> artifacts/runs/<id>/
python stage4_geo.py                    # project county geometry to SVG paths
python export_v2.py                     # write the dashboard payloads
python fetch_live.py                    # score today + next 3 days from live weather
python make_report.py                   # regenerate the results document

python serve.py                         # http://localhost:8787
```

`python v2_models.py` is a smoke check: it reloads every booster and confirms
it reproduces the test average precision the run recorded, which catches a
truncated download or a drifted feature list before either reaches the
dashboard.

The raw rainfall panel and the FEMA/SVI source tables are not committed — they
are large and not ours to republish. `build_daily.py` expects them in the
directory named by `CLIMATE_DATA_DIR`; everything after that step runs from
committed files.

After a panel rebuild, regenerate the live feature cache with
`python fetch_live.py --rebuild-static`. `refdata/county_static.csv` holds
per-county rainfall climatology that forms the denominators of the ratio
features, and the top flood feature is one of them — a cache from an older
panel would quietly score every county against the wrong normal.

## Known limits

- **The label is an observational record, not hazard physics.** A StormEvents
  flood episode exists because a local Weather Service office wrote one up, so
  a real flood nobody reported counts as a negative. Hurricane positives are
  purely geometric — the county intersected the best track — so a county grazed
  by a weak system and a county destroyed by a major landfall are the same
  positive.
- **Training rainfall is observed, not forecast.** Features use station
  rainfall over the prediction window, which a real forecaster would not have
  on the morning the model runs. This is the single largest overstatement in
  the project. The fix is retraining on archived forecasts (GEFSv12
  reforecast); until then treat every lead-time number as an upper bound.
- **No held-out-state validation in this run.** Earlier work measured what the
  models give back when a whole state is removed from training. This run did
  not repeat it, so nothing here supports a claim about counties outside the
  Gulf Coast. This is the clearest gap.
- **356 of 3,109 counties are modelled.** Training is restricted to the five
  Gulf states and, within them, to counties with a rainfall station in the
  source data.
- **The rainfall record has holes.** The panel spans 9,645 calendar days and
  the median county has 9,595, but the worst has 3,702 and 28 counties fall
  below 80%. 296 of the 3,023 listed events have no day of their three-day
  lead-up in the record, so no model reading exists for them.
- **Hurricane labels stop at 2024.** HURDAT2 is finalised a season behind, so
  hurricane metrics are scored on 2020–2024 while flood metrics use the full
  test window. The two hazards' test sets are not the same size.
- **No soil moisture, no river gauges.** Antecedent wetness is proxied by 30-,
  60- and 90-day rainfall, a weak substitute for catchment state.
- **No hurricane wind field.** Hurricane risk is driven by rainfall and season;
  track forecast, intensity and storm surge are absent entirely.
- **8,052 flood and 2,244 hurricane onsets in the test period.** Confidence
  intervals on every metric here are correspondingly wide, and none are
  reported.

## Data sources

NOAA NCEI GHCN-Daily · NOAA NCEI StormEvents · NHC HURDAT2 Atlantic best track ·
FEMA Disaster Declarations Summaries (secondary target only) ·
CDC/ATSDR Social Vulnerability Index · Open-Meteo (forecast and ERA5 archive) ·
NWS api.weather.gov (active alerts)
