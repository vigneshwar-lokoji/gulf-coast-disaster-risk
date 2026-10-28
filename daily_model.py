"""Daily extreme-rainfall model: build features, train XGBoost, compare to baselines.

Target is strictly future precipitation, so no part of the label window overlaps
the feature window. Prediction is made at the end of day t for days t+1 onward.

Run: python daily_model.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from build_panel import build_county_daily, load_static
from config import CLIMATOLOGY_YEARS, OUT_DIR, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

HORIZONS = [3, 7]
TRAIL_WINDOWS = [3, 7, 14, 30, 60, 90]
API_DECAYS = [0.85, 0.90, 0.95]
HEAVY_MM = 25.4


def wide(daily, col):
    return daily.pivot(index="DATE", columns="county_fips", values=col).sort_index()


def build():
    daily = build_county_daily()[["county_fips", "DATE", "prcp", "tmax", "tmin"]]
    P = wide(daily, "prcp")
    TX = wide(daily, "tmax")
    TN = wide(daily, "tmin")
    P0 = P.fillna(0.0)

    feats = {}

    # Trailing accumulations: all known by end of day t.
    for w in TRAIL_WINDOWS:
        feats[f"trail{w}"] = P.rolling(w, min_periods=max(2, w // 3)).sum()
    feats["prcp_today"] = P
    feats["max1d_7"] = P.rolling(7, min_periods=3).max()
    feats["max1d_30"] = P.rolling(30, min_periods=10).max()
    feats["nwet_30"] = (P >= 1.0).rolling(30, min_periods=10).sum()
    feats["nheavy_30"] = (P >= HEAVY_MM).rolling(30, min_periods=10).sum()
    feats["obs_30"] = P.notna().rolling(30, min_periods=1).sum()

    # Antecedent Precipitation Index, the standard hydrological soil-moisture
    # proxy: API_t = k*API_{t-1} + P_t. Scale is irrelevant to trees, so the
    # ewm form is used directly.
    for k in API_DECAYS:
        feats[f"api{int(k*100)}"] = P0.ewm(alpha=1 - k, adjust=False).mean()

    feats["tmax_7"] = TX.rolling(7, min_periods=3).mean()
    feats["tmin_7"] = TN.rolling(7, min_periods=3).mean()
    feats["tmax_30"] = TX.rolling(30, min_periods=10).mean()

    # Day-of-year climatology of the 30-day accumulation, fit on the reference
    # period only so later years never inform their own anomaly.
    lo, hi = CLIMATOLOGY_YEARS
    t30 = feats["trail30"]
    ref = t30[(t30.index.year >= lo) & (t30.index.year <= hi)]
    doy_mean = ref.groupby(ref.index.dayofyear).mean()
    doy_std = ref.groupby(ref.index.dayofyear).std().replace(0, np.nan)
    feats["trail30_anom"] = (
        t30 - doy_mean.reindex(t30.index.dayofyear).set_axis(t30.index)
    ) / doy_std.reindex(t30.index.dayofyear).set_axis(t30.index)

    # Strictly-future targets. rolling(h).sum().shift(-h) at t is the sum over
    # days t+1..t+h.
    targets = {}
    for h in HORIZONS:
        fut = P0.rolling(h).sum().shift(-h)
        trail = P.rolling(h, min_periods=max(2, h // 3)).sum()
        tref = trail[(trail.index.year >= lo) & (trail.index.year <= hi)]
        thr = tref.quantile(0.99)
        targets[f"y_ext{h}"] = fut.ge(thr, axis=1).where(fut.notna())

    df = pd.concat(
        [v.stack().rename(k) for k, v in {**feats, **targets}.items()], axis=1
    ).reset_index()
    df.columns = ["DATE", "county_fips"] + list(df.columns[2:])

    df["doy"] = df["DATE"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month

    df = df.merge(load_static(), on="county_fips", how="inner")
    return df


DROP = {"DATE", "county_fips", "state", "year", "doy", "month"}


def fit_and_score(df, target):
    feats = [
        c
        for c in df.columns
        if c not in DROP and not c.startswith("y_") and df[c].dtype.kind in "fiub"
    ]
    ok = df[target].notna()
    tr = ok & df.year.between(*SPLIT_TRAIN)
    va = ok & df.year.between(*SPLIT_VAL)
    te = ok & df.year.between(*SPLIT_TEST)

    ytr = df.loc[tr, target].astype(int)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    m = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=60,
        tree_method="hist",
        n_jobs=-1,
        random_state=0,
    )
    m.fit(
        df.loc[tr, feats],
        ytr,
        eval_set=[(df.loc[va, feats], df.loc[va, target].astype(int))],
        verbose=False,
    )
    p = m.predict_proba(df.loc[te, feats])[:, 1]
    return m, feats, df.loc[te, target].astype(int).values, p, df.loc[te, "DATE"].values


def ap_ci_by_date(y, p, dates, n_boot=200, seed=0):
    """Bootstrap by date, not by row. One storm covers many counties on one day,
    so row-level resampling would badly understate the interval."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(dates)
    idx = pd.Series(np.arange(len(y))).groupby(pd.Series(dates)).apply(np.array)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[d] for d in pick])
        if y[sel].sum() == 0:
            continue
        out.append(average_precision_score(y[sel], p[sel]))
    return np.percentile(out, [2.5, 97.5])


def baselines(df, target, h):
    ok = df[target].notna()
    tr = df[ok & df.year.between(*SPLIT_TRAIN)]
    te = df[ok & df.year.between(*SPLIT_TEST)]
    y = te[target].astype(int).values
    gm = tr[target].astype(int).mean()

    seas = tr.groupby("month")[target].mean()
    cs = tr.groupby(["county_fips", "month"])[target].mean()

    out = {
        "constant prevalence": np.full(len(y), gm),
        "seasonal climatology": te["month"].map(seas).fillna(gm).values,
        "county x month climatology": np.where(
            np.isnan(
                v := te.set_index(["county_fips", "month"]).index.map(cs).to_numpy(float)
            ),
            gm,
            v,
        ),
        "persistence (today extreme)": te[f"trail{h}"]
        .ge(tr[f"trail{h}"].quantile(0.99))
        .astype(float)
        .values,
        "trailing 30d precip (raw)": te["trail30"].fillna(0).values,
        f"API 0.95 (raw)": te["api95"].fillna(0).values,
    }
    return y, out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building daily features (this takes a couple of minutes)...")
    df = build()
    print(f"  {df.shape[0]:,} county-days x {df.shape[1]} cols\n")

    for h in HORIZONS:
        target = f"y_ext{h}"
        y, base = baselines(df, target, h)
        prev = y.mean()

        rows = []
        for name, p in base.items():
            ap = average_precision_score(y, p)
            rows.append({"model": name, "AP": ap, "lift": ap / prev, "ROC": roc_auc_score(y, p)})

        m, feats, yte, pte, dates = fit_and_score(df, target)
        ap = average_precision_score(yte, pte)
        lo, hi = ap_ci_by_date(yte, pte, dates)
        rows.append(
            {
                "model": "XGBoost",
                "AP": ap,
                "lift": ap / prev,
                "ROC": roc_auc_score(yte, pte),
            }
        )

        print("=" * 72)
        print(f"EXTREME RAINFALL, next {h} days   (test 2020-2026)")
        print(f"positives {y.sum():,} / {len(y):,} rows   prevalence {prev:.4f}")
        print("=" * 72)
        print(
            pd.DataFrame(rows).to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )
        print(f"\nXGBoost AP 95% CI (bootstrapped by date): [{lo:.4f}, {hi:.4f}]")
        best_base = max(r["AP"] for r in rows[:-1])
        verdict = "BEATS" if ap > best_base else "LOSES TO"
        print(f"VERDICT: XGBoost {verdict} the best baseline ({ap:.4f} vs {best_base:.4f})")

        imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(
            ascending=False
        )
        print(f"\ntop features: {', '.join(imp.head(8).index)}\n")


if __name__ == "__main__":
    main()
