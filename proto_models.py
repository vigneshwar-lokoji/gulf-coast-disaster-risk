"""Prototype hazard models, and the single place that defines how to score them.

Baseline testing picked a different model class per hazard, which is the main
methodological result of the prototype:

  flood  -> a one-predictor calibrated logistic on ratio_fwd3. Gradient
            boosting loses to this by 4x out of sample. With ~1,400 training
            positives across 356 counties the trees fit thresholds specific to
            individual historical events and do not transfer.
  storm  -> XGBoost with monotone constraints on the rain features, which
            beats its best single-feature baseline by 1.7x.

Both consumers (export_proto, fetch_live) import load_scorers from here so the
dashboard can never drift from what was actually trained.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from config import OUT_DIR

FLOOD_FEATS = ["ratio_fwd3"]

STORM_FEATS = [
    "prcp", "prcp_max_stn", "prcp_prev3", "prcp_prev7", "prcp_prev30",
    "prcp_prev60", "prcp_fwd1", "prcp_fwd3", "prcp_fwd3_max", "ratio_prev3",
    "ratio_fwd3", "ratio_prev30", "log_fwd3", "tmax", "tmin", "month_sin",
    "month_cos", "hurricane_season", "lat", "lon", "pop_density",
]
_STORM_UP = {
    "prcp", "prcp_max_stn", "prcp_prev3", "prcp_prev7", "prcp_prev30",
    "prcp_prev60", "prcp_fwd1", "prcp_fwd3", "prcp_fwd3_max", "ratio_prev3",
    "ratio_fwd3", "ratio_prev30", "log_fwd3",
}

FEATS = {"flood": FLOOD_FEATS, "storm": STORM_FEATS}


def _clean(df, feats):
    return (
        df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype="float64")
    )


def fit_flood(tr, va, save=True):
    X = _clean(tr, FLOOD_FEATS)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit((X - mu) / sd, tr["y_flood"].to_numpy())
    spec = {
        "type": "logistic",
        "features": FLOOD_FEATS,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": m.coef_[0].tolist(),
        "intercept": float(m.intercept_[0]),
    }
    if save:
        (OUT_DIR / "proto_flood.json").write_text(json.dumps(spec, indent=2))
    return spec


def fit_storm(tr, va, save=True):
    m = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        monotone_constraints=tuple(1 if f in _STORM_UP else 0 for f in STORM_FEATS),
        eval_metric="aucpr",
        early_stopping_rounds=40,
        tree_method="hist",
        n_jobs=-1,
    )
    m.fit(
        tr[STORM_FEATS], tr["y_storm"].to_numpy(),
        eval_set=[(va[STORM_FEATS], va["y_storm"].to_numpy())], verbose=False,
    )
    if save:
        m.save_model(OUT_DIR / "proto_storm.json")
    return m


def flood_scorer(spec):
    mu = np.asarray(spec["mu"])
    sd = np.asarray(spec["sd"])
    coef = np.asarray(spec["coef"])
    b = spec["intercept"]

    def flood(df):
        z = (_clean(df, spec["features"]) - mu) / sd
        return 1.0 / (1.0 + np.exp(-(z @ coef + b)))

    return flood


def storm_scorer(model):
    return lambda df: model.predict_proba(df[STORM_FEATS])[:, 1]


def load_scorers():
    """Return {hazard: callable(DataFrame) -> probability array}."""
    spec = json.loads((OUT_DIR / "proto_flood.json").read_text())
    storm_model = xgb.XGBClassifier()
    storm_model.load_model(OUT_DIR / "proto_storm.json")
    return {"flood": flood_scorer(spec), "storm": storm_scorer(storm_model)}
