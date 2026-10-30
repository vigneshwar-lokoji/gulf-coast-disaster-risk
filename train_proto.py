"""Fit the prototype models and write the metrics the dashboard displays.

Baselines are reported next to the model on purpose. At 0.07-0.22% prevalence
accuracy and ROC-AUC are both uninformative, so the headline numbers are
average precision, lift over prevalence, and the ratio to the best baseline.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import OUT_DIR
from proto_models import FEATS, fit_flood, fit_storm, load_scorers

TRAIN = (2010, 2019)
VAL = (2020, 2022)
TEST = (2023, 2026)


def split(d):
    y = d["DATE"].dt.year
    return d[y.between(*TRAIN)], d[y.between(*VAL)], d[y.between(*TEST)]


def baselines(tr, te, haz):
    y = te[f"y_{haz}"].to_numpy()
    out = {"constant_prevalence": float(
        average_precision_score(y, np.full(len(y), tr[f"y_{haz}"].mean()))
    )}
    for f in ("ratio_fwd3", "prcp_fwd3", "prcp_fwd3_max", "prcp_prev30",
              "ratio_prev3", "pop_density"):
        out[f"single_{f}"] = float(
            average_precision_score(y, te[f].fillna(0).to_numpy())
        )
    rate = tr.groupby(tr["DATE"].dt.month)[f"y_{haz}"].mean()
    out["month_climatology"] = float(average_precision_score(
        y, te["DATE"].dt.month.map(rate).fillna(tr[f"y_{haz}"].mean()).to_numpy()
    ))
    return out


def main():
    d = pd.read_parquet(OUT_DIR / "daily_panel.parquet")
    tr, va, te = split(d)
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}\n")

    fit_flood(tr, va)
    storm_model = fit_storm(tr, va)
    scorers = load_scorers()

    report = {
        "splits": {"train": TRAIN, "val": VAL, "test": TEST},
        "horizon_days": 3,
        "target": "a FEMA flood/storm incident begins in this county within the next 3 days",
        "n_rows": int(len(d)),
        "n_counties": int(d["county_fips"].nunique()),
        "features": {k: v for k, v in FEATS.items()},
    }

    for haz in ("flood", "storm"):
        yte = te[f"y_{haz}"].to_numpy()
        base = baselines(tr, te, haz)
        best_name = max(base, key=base.get)
        p = scorers[haz](te)
        ap = float(average_precision_score(yte, p))
        prev = float(yte.mean())

        print(f"=== {haz} ===")
        for k, v in sorted(base.items(), key=lambda x: -x[1])[:4]:
            print(f"  baseline {k:26s} AP={v:.5f}")
        print(f"  MODEL                      AP={ap:.5f}"
              f"  prevalence={prev:.5f}  lift={ap/prev:.1f}x"
              f"  vs_best_baseline={ap/base[best_name]:.2f}x\n")

        entry = {
            "model": "logistic_1feature" if haz == "flood" else "xgboost_monotone",
            "features": FEATS[haz],
            "test_ap": ap,
            "test_prevalence": prev,
            "lift_over_prevalence": ap / prev if prev else None,
            "best_baseline": best_name,
            "best_baseline_ap": base[best_name],
            "vs_best_baseline": ap / base[best_name] if base[best_name] else None,
            "n_test_pos": int(yte.sum()),
            "n_train_pos": int(tr[f"y_{haz}"].sum()),
            "baselines": base,
        }
        if haz == "storm":
            entry["top_features"] = [
                [f, round(float(v), 4)]
                for f, v in sorted(
                    zip(FEATS["storm"], storm_model.feature_importances_),
                    key=lambda x: -x[1],
                )[:8]
            ]
            entry["best_iteration"] = int(storm_model.best_iteration)
        report[haz] = entry

    (OUT_DIR / "proto_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_DIR/'proto_metrics.json'}")


if __name__ == "__main__":
    main()
