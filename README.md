# Gulf Coast Disaster Declaration Risk

**[Open the dashboard →](https://vigneshwar-lokoji.github.io/gulf-coast-disaster-risk/)**

Given that a Gulf Coast county has just exceeded its own 1-in-100-day rainfall
threshold, how likely is a federal disaster declaration within the next 14 days?

The dashboard runs entirely in the browser from a 20 KB model file. No server,
no tracking, no build step.

## What this is, and what it is not

This is **not a weather forecaster**. It does not predict rain. It consumes the
National Weather Service's rainfall forecast as an input and maps it to
declaration risk.

It is also **not a warning system**. The highest probability the model has ever
produced is **15.7%**. It is a ranked priority list across 218 counties — useful
for deciding where to look first, useless for deciding whether to evacuate.

The model is **conditional**. A county only enters the data once it has already
exceeded its own 99th-percentile three-day rainfall. Below that line the model
has nothing to say — not "low risk", but no opinion at all. The dashboard shows
this explicitly rather than rendering a quiet day as a blank screen.

## Results

Held-out test period **2020–2026**, 5,260 hazard county-days the model never saw.

| | |
|---|---|
| Average precision | 0.0929 |
| Base rate (prevalence) | 4.60% |
| Lift over base rate | 2.02× |
| Highest probability ever produced | 15.7% |

Average precision and lift are reported rather than accuracy or ROC-AUC. At a
4.6% base rate, a model that predicts "no declaration" every time scores 95.4%
accuracy while being entirely useless.

### Operating points

| Flag when P ≥ | Counties flagged / year | Precision | Recall | Lift |
|---|---|---|---|---|
| 5% | 342 | 7.9% | 77.7% | 1.71× |
| 10% | 86 | 11.8% | 29.3% | 2.56× |

There is no free lunch: raising the threshold cuts workload and raises
precision, but misses more declarations.

### Where it works, and where it does not

Lift when each state is held out of training entirely:

| FL | LA | MS | AL | TX |
|---|---|---|---|---|
| 1.98× | 1.56× | 1.44× | 1.23× | **1.11×** |

**Texas output is effectively unvalidated.** The model barely beats chance there.
The dashboard says so on every screen that can show a Texas row.

## The three views

- **Historical replay** — every hazard day in the held-out period, with the
  model's ranking beside the real outcome. Ground truth is in the table, so
  every number is checkable. Best single day: **2021-05-17** (Louisiana floods),
  where 5 of the top 6 ranked counties were declared.
- **Live forecast** — a rolling window from recent observed rainfall out to the
  NWS forecast horizon, roughly seven days. No ground truth exists for these
  days, so unlike the replay, nothing here is checkable.
- **Operating point** — recomputes precision, recall and lift in the browser
  over all 5,260 held-out rows as you move the threshold.

## How the live path works

The hard part is not the forward forecast. It is the *antecedent* features —
`prev7` and `prev30` need rainfall that has already fallen.

| Source | Role | Latency |
|---|---|---|
| NCEI GHCN-Daily | observed rainfall, the same station network used in training | ~2 days |
| NWS gridpoint QPF | forecast rainfall, ~7 days ahead | live |

`stage4_live.py --verify` checks the live fetch against the stored training
panel. On shared station-days the values agree to 0.05 mm **98.76%** of the
time. Coverage differs because GHCN backfills and quality-controls recent data
for weeks after publication, which is why live output will never be
bit-identical to a replay row.

A GitHub Action regenerates `docs/data/live.json` daily, so the live tab does
not silently freeze on the day it was last run.

## Reproducing

```bash
pip install -r requirements.txt

python stage2.py          # build the hazard-day panel and fit the model
python stage3.py          # calibration, operating points, leave-one-state-out
python stage4_export.py   # freeze the model to docs/data/model.json
python stage4_geo.py      # project county geometry to SVG paths
python stage4_live.py     # fetch live rainfall and predict the current window

python serve.py           # http://localhost:8787
```

The exported model is pure arithmetic — scaler, logistic coefficients and Platt
calibration — verified against the sklearn pipeline to 2e-9. The paper and the
dashboard cannot drift apart.

The raw rainfall panel (25 MB) is not committed. Everything the dashboard and
the daily refresh need is in `docs/data/` and `refdata/`.

## Known limits

- NWS forecasts a grid cell at the county centroid, while training used the mean
  over the county's reporting stations. This representativeness mismatch is
  real and unquantified.
- Declarations are a political and administrative outcome, not purely a physical
  one. The model sees rainfall, not county emergency-management capacity.
- Feature coefficients are standardised logistic weights. Sign and relative size
  are meaningful; they are not causal effects.

## Data sources

NOAA NCEI GHCN-Daily · NWS api.weather.gov · FEMA Disaster Declarations
Summaries · CDC/ATSDR Social Vulnerability Index
