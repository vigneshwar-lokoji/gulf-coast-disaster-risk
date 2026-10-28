"""Tier-0 baselines. If XGBoost cannot beat these, the weather features are noise.

Climatology here is not a strawman: P(event | county, month-of-year) already
encodes 'Louisiana in September is dangerous', which is most of what a county
risk model is asked to say.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import OUT_DIR, SPLIT_TEST, SPLIT_TRAIN

SMOOTH = 10.0


def score(y, p, name):
    prev = y.mean()
    ap = average_precision_score(y, p)
    return {
        "model": name,
        "avg_precision": ap,
        "lift": ap / prev,
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else np.nan,
    }


def main():
    panel = pd.read_parquet(OUT_DIR / "panel.parquet")
    panel["moy"] = pd.PeriodIndex(panel["month"], freq="M").month
    xgb_res = pd.read_csv(OUT_DIR / "xgb_results.csv")

    for tgt in ("flood", "storm"):
        for lead in (1, 2, 3):
            col = f"y_{tgt}_onset_lead{lead}"
            d = panel[panel[col].notna()].copy()
            d[col] = d[col].astype(int)
            tr = d[d.year.between(*SPLIT_TRAIN)]
            te = d[d.year.between(*SPLIT_TEST)].copy()
            y = te[col].values
            gm = tr[col].mean()

            # Seasonal climatology: month-of-year only.
            seas = tr.groupby("moy")[col].mean()
            p_seas = te["moy"].map(seas).fillna(gm).values

            # County x month-of-year, shrunk toward the county mean because many
            # county-month cells hold only ~25 observations.
            cnt = tr.groupby(["county_fips", "moy"])[col].agg(["sum", "count"])
            cmean = tr.groupby("county_fips")[col].mean()
            cnt = cnt.join(cmean.rename("cm"), on="county_fips")
            cnt["p"] = (cnt["sum"] + SMOOTH * cnt["cm"]) / (cnt["count"] + SMOOTH)
            p_cs = (
                te.set_index(["county_fips", "moy"])
                .index.map(cnt["p"])
                .to_numpy(dtype=float)
            )
            p_cs = np.where(np.isnan(p_cs), gm, p_cs)

            # Persistence: did it happen this month?
            p_pers = te[f"y_{tgt}_onset"].astype(float).values

            rows = [
                score(y, np.full(len(y), gm), "constant prevalence"),
                score(y, p_seas, "seasonal climatology"),
                score(y, p_cs, "county x season climatology"),
                score(y, p_pers, "persistence"),
            ]
            x = xgb_res[
                (xgb_res.target == tgt)
                & (xgb_res.lead == lead)
                & (xgb_res.split == "test")
            ]
            if len(x):
                r = x.iloc[0]
                rows.append(
                    {
                        "model": "XGBoost",
                        "avg_precision": r.avg_precision,
                        "lift": r.lift_over_prevalence,
                        "roc_auc": r.roc_auc,
                    }
                )

            print(f"\n=== {tgt.upper()}  lead {lead}  (test 2020-2026, prev {y.mean():.4f}) ===")
            print(
                pd.DataFrame(rows).to_string(
                    index=False, float_format=lambda v: f"{v:.4f}"
                )
            )


if __name__ == "__main__":
    main()
