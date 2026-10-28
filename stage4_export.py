"""Stage 4a: freeze the Stage 3 model into a standalone artifact.

The dashboard must not import sklearn, pandas or the training data. It needs a
small file of numbers and about twenty lines of arithmetic. This script writes
that file and then proves the arithmetic reproduces sklearn's output on all
5,260 test rows, so a discrepancy between the paper and the dashboard is
impossible rather than merely unlikely.

Exported pipeline, in order:
    standardise -> linear score -> logistic -> Platt rescale

The ranker is fitted on 2000-2014 and the Platt calibrator on 2015-2019,
exactly as in stage3.py, so the exported model is the one whose 2020-2026
numbers were reported. Refitting on train+val would use more data but would
leave the calibrator without a clean holdout, so it is deliberately not done
here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, FILES, OUT_DIR

STAGE4_DIR = OUT_DIR / "stage4"
STAGE4_DIR.mkdir(parents=True, exist_ok=True)
MODEL_JSON = STAGE4_DIR / "model.json"

EVENTS = OUT_DIR / "stage2" / "stage2_events_broad.parquet"
TRAIN_END, VAL_END = 2014, 2019
FEATS = ["ratio", "log_fwd3", "max1", "prev7", "prev30", "month_sin", "month_cos",
         "other_extent", "gulf_extent"]
STAGE3_TEST_AP = 0.0929

ev = pd.read_parquet(EVENTS)
med = ev.loc[ev.year <= TRAIN_END, FEATS].median()
ev[FEATS] = ev[FEATS].fillna(med)

tr = ev[ev.year <= TRAIN_END]
va = ev[(ev.year > TRAIN_END) & (ev.year <= VAL_END)]
te = ev[ev.year > VAL_END].reset_index(drop=True)

ranker = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
).fit(tr[FEATS], tr.y)

s_va = ranker.predict_proba(va[FEATS])[:, 1]
s_te = ranker.predict_proba(te[FEATS])[:, 1]

platt = LogisticRegression(C=1e6, max_iter=2000).fit(s_va.reshape(-1, 1), va.y)
p_te = platt.predict_proba(s_te.reshape(-1, 1))[:, 1]

ap = average_precision_score(te.y, p_te)
assert abs(ap - STAGE3_TEST_AP) < 5e-4, f"AP {ap:.4f} != Stage 3 {STAGE3_TEST_AP}"

sc = ranker.named_steps["standardscaler"]
lr = ranker.named_steps["logisticregression"]

# Per-county exceedance threshold: p99 of the forward 3-day total over
# 2000-2014. Already carried on the event table, one constant per county.
thr = ev.groupby("county_fips")["thr"].first()

# Centroid = mean of the county's reporting stations. Approximate, but it only
# has to pick the right NWS forecast grid cell.
mapping = pd.read_csv(DATA_DIR / FILES["mapping"])
mapping["county_fips"] = mapping["county_fips"].astype(str).str.zfill(5)
geo = mapping.groupby("county_fips").agg(
    lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"),
    name=("county_name", "first"), state=("state_abbr", "first"),
)

counties = {}
for fips_code, t in thr.items():
    g = geo.loc[fips_code] if fips_code in geo.index else None
    counties[fips_code] = {
        "threshold_mm": round(float(t), 4),
        "name": None if g is None else str(g["name"]),
        "state": None if g is None else str(g["state"]),
        "lat": None if g is None else round(float(g["lat"]), 4),
        "lon": None if g is None else round(float(g["lon"]), 4),
    }
assert all(c["lat"] is not None for c in counties.values()), "missing centroids"

model = {
    "schema": 1,
    "trained_on": f"2000-{TRAIN_END}",
    "calibrated_on": f"{TRAIN_END + 1}-{VAL_END}",
    "evaluated_on": f"{int(te.year.min())}-{int(te.year.max())}",
    "test_ap": round(float(ap), 6),
    "test_prevalence": round(float(te.y.mean()), 6),
    "max_observed_probability": round(float(p_te.max()), 6),
    "features": FEATS,
    "feature_mean": [round(float(v), 8) for v in sc.mean_],
    "feature_scale": [round(float(v), 8) for v in sc.scale_],
    "feature_median_fallback": {k: round(float(med[k]), 8) for k in FEATS},
    "coef": [round(float(v), 8) for v in lr.coef_[0]],
    "intercept": round(float(lr.intercept_[0]), 8),
    "platt_coef": round(float(platt.coef_[0][0]), 8),
    "platt_intercept": round(float(platt.intercept_[0]), 8),
    # From stage3.py. Shipped so the dashboard states workload and hit rate
    # rather than letting a user invent their own threshold.
    "operating_points": [
        {"threshold": 0.05, "flagged_per_year": 341.7, "precision": 0.079,
         "recall": 0.777, "lift": 1.71},
        {"threshold": 0.10, "flagged_per_year": 86.0, "precision": 0.118,
         "recall": 0.293, "lift": 2.56},
    ],
    # stage3.py leave-one-state-out. TX fell to 1.11x when held out, so the
    # dashboard must degrade its own claims there rather than show one number.
    "state_transfer_lift": {"AL": 1.23, "FL": 1.98, "LA": 1.56, "MS": 1.44, "TX": 1.11},
    "counties": counties,
}

MODEL_JSON.write_text(json.dumps(model, indent=1))
print(f"wrote {MODEL_JSON}  ({MODEL_JSON.stat().st_size / 1024:.1f} KB, "
      f"{len(counties)} counties)")


# ----------------------------------------------- the dashboard's entire model

def predict(m, feats):
    """Calibrated declaration probability. Pure arithmetic, no sklearn."""
    z = m["intercept"]
    for name, c, mu, sd in zip(m["features"], m["coef"],
                               m["feature_mean"], m["feature_scale"]):
        x = feats.get(name)
        if x is None:
            x = m["feature_median_fallback"][name]
        z += c * (x - mu) / sd
    s = 1.0 / (1.0 + np.exp(-z))
    return 1.0 / (1.0 + np.exp(-(m["platt_coef"] * s + m["platt_intercept"])))


reloaded = json.loads(MODEL_JSON.read_text())
mine = np.array([predict(reloaded, r) for r in te[FEATS].to_dict("records")])
worst = np.abs(mine - p_te).max()
print(f"exported arithmetic vs sklearn over {len(te):,} test rows: "
      f"max abs diff {worst:.3e}")
assert worst < 1e-6, "exported model does not reproduce the sklearn pipeline"
print(f"AP from exported path {average_precision_score(te.y, mine):.4f} "
      f"(Stage 3: {STAGE3_TEST_AP})")
print(f"probability range {mine.min():.4f} - {mine.max():.4f}")

# ------------------------------------------------------- dashboard payloads

# Replay data: what the model said on every real hazard day in the test period,
# alongside what actually happened. This is the only mode that can be checked
# against ground truth, so it is the dashboard's default.
DASH_DATA = Path(__file__).parent / "docs" / "data"
DASH_DATA.mkdir(parents=True, exist_ok=True)

rep = te[["county_fips", "DATE", "y", "ratio", "fwd3", "other_extent"]].copy()
rep["p"] = mine
rep["day"] = rep["DATE"].dt.strftime("%Y-%m-%d")
by_day = {}
for day, grp in rep.sort_values("p", ascending=False).groupby("day", sort=True):
    by_day[day] = [
        [r.county_fips, round(float(r.p), 4), int(r.y),
         round(float(r.ratio), 3), round(float(r.fwd3), 1)]
        for r in grp.itertuples()
    ]

(DASH_DATA / "replay.json").write_text(json.dumps({
    "columns": ["county_fips", "p", "declared", "ratio", "fwd3_mm"],
    "period": f"{int(te.year.min())}-{int(te.year.max())}",
    "n_rows": len(rep),
    "n_days": len(by_day),
    "n_declared": int(rep.y.sum()),
    "days": by_day,
}, separators=(",", ":")))

(DASH_DATA / "model.json").write_text(json.dumps(model, separators=(",", ":")))

for f in ("model.json", "replay.json"):
    print(f"wrote docs/data/{f}  "
          f"({(DASH_DATA / f).stat().st_size / 1024:.0f} KB)")
