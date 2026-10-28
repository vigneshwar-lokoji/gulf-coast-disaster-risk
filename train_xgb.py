"""XGBoost baselines for flood and storm, with honest rare-event evaluation.

Reports average precision against the prevalence floor. At 2-4% positives a
model can score ROC-AUC 0.85 and still be useless, so lift over prevalence is
the number that matters.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import OUT_DIR, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

DROP = {
    "county_fips",
    "month",
    "state",
    "year",
    "moy",
    # Excluded deliberately: the FEMA declaration base rate trends upward over
    # time for administrative reasons, so a year feature lets the model learn
    # policy drift and then fail when the trend changes.
}


def feature_cols(panel):
    return [
        c
        for c in panel.columns
        if c not in DROP and not c.startswith("y_") and panel[c].dtype.kind in "fiu"
    ]


def split_rows(panel):
    yr = panel["year"]
    return (
        yr.between(*SPLIT_TRAIN),
        yr.between(*SPLIT_VAL),
        yr.between(*SPLIT_TEST),
    )


def evaluate(y, p):
    prev = y.mean()
    ap = average_precision_score(y, p)
    return {
        "n": len(y),
        "pos": int(y.sum()),
        "prevalence": prev,
        "avg_precision": ap,
        "lift_over_prevalence": ap / prev if prev > 0 else np.nan,
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def fit_one(panel, target, feats, seed=0):
    y_all = panel[target]
    ok = y_all.notna()
    tr, va, te = split_rows(panel)
    tr, va, te = tr & ok, va & ok, te & ok

    ytr = y_all[tr].astype(int)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        panel.loc[tr, feats],
        ytr,
        eval_set=[(panel.loc[va, feats], y_all[va].astype(int))],
        verbose=False,
    )

    res = {}
    for name, mask in (("val", va), ("test", te)):
        p = model.predict_proba(panel.loc[mask, feats])[:, 1]
        res[name] = evaluate(y_all[mask].astype(int).values, p)
    return model, res


def leave_one_state_out(panel, target, feats, seed=0):
    y_all = panel[target]
    ok = y_all.notna()
    out = {}
    for st in sorted(panel["state"].unique()):
        tr = ok & (panel["state"] != st)
        te = ok & (panel["state"] == st)
        ytr = y_all[tr].astype(int)
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        m = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        m.fit(panel.loc[tr, feats], ytr, verbose=False)
        p = m.predict_proba(panel.loc[te, feats])[:, 1]
        out[st] = evaluate(y_all[te].astype(int).values, p)
    return out


def main():
    panel = pd.read_parquet(OUT_DIR / "panel.parquet")
    feats = feature_cols(panel)
    print(f"panel {panel.shape[0]:,} rows | {len(feats)} features\n")

    rows = []
    models = {}
    for tgt in ("flood", "storm"):
        for lead in (0, 1, 2, 3):
            target = f"y_{tgt}_onset_lead{lead}"
            model, res = fit_one(panel, target, feats)
            models[target] = model
            for split, mt in res.items():
                rows.append({"target": tgt, "lead": lead, "split": split, **mt})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "xgb_results.csv", index=False)

    print("=" * 78)
    print("TEMPORAL HOLDOUT  (train 1990-2014 / val 2015-2019 / test 2020-2026)")
    print("=" * 78)
    show = df[df.split == "test"][
        ["target", "lead", "pos", "prevalence", "avg_precision", "lift_over_prevalence", "roc_auc", "brier"]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("LEAVE-ONE-STATE-OUT  (spatial transfer, lead 1)")
    print("=" * 78)
    for tgt in ("flood", "storm"):
        loso = leave_one_state_out(panel, f"y_{tgt}_onset_lead1", feats)
        print(f"\n{tgt.upper()}")
        for st, mt in loso.items():
            print(
                f"  held-out {st}: AP {mt['avg_precision']:.4f} "
                f"(prev {mt['prevalence']:.4f}, lift {mt['lift_over_prevalence']:.2f}x) "
                f"ROC {mt['roc_auc']:.3f}"
            )

    print("\n" + "=" * 78)
    print("TOP FEATURES (gain, lead 1)")
    print("=" * 78)
    for tgt in ("flood", "storm"):
        m = models[f"y_{tgt}_onset_lead1"]
        imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(
            ascending=False
        )
        print(f"\n{tgt.upper()}")
        print(imp.head(12).to_string())


if __name__ == "__main__":
    main()
