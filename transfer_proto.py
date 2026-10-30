"""Spatial and temporal transfer validation.

The pooled test score answers "how well does this rank counties it has seen
before?" That is not the question anyone cares about. The two that matter:

  spatial  -> hold an entire state out of training, then score it. The gap
              against the pooled model on the same rows is the cost of asking
              the model to work somewhere it has never been.
  temporal -> score each test year separately. If skill decays with distance
              from the training window, the model is fitting a period rather
              than a process.

Both comparisons are made on identical row sets, so prevalence cancels and the
ratio is a clean retention figure.
"""

import json

import pandas as pd
from sklearn.metrics import average_precision_score

from config import OUT_DIR
from proto_models import (fit_flood, fit_storm, flood_scorer, load_scorers,
                          storm_scorer)
from train_proto import TEST, TRAIN, VAL, split

STATES = {"01": "Alabama", "12": "Florida", "22": "Louisiana",
          "28": "Mississippi", "48": "Texas"}
MIN_POS = 20  # below this an AP is noise, not a measurement


def ap(y, p):
    return float(average_precision_score(y, p)) if y.sum() else None


def main():
    d = pd.read_parquet(OUT_DIR / "daily_panel.parquet")
    d["state"] = d["county_fips"].str[:2]
    tr, va, te = split(d)
    pooled = load_scorers()

    out = {"splits": {"train": TRAIN, "val": VAL, "test": TEST},
           "min_positives": MIN_POS, "spatial": [], "temporal": []}

    # ---------------------------------------------------- spatial transfer
    for code, name in STATES.items():
        sub = te[te["state"] == code]
        trx, vax = tr[tr["state"] != code], va[va["state"] != code]
        print(f"=== {name} ({len(sub):,} test rows) ===", flush=True)

        held = {
            "flood": flood_scorer(fit_flood(trx, vax, save=False)),
            "storm": storm_scorer(fit_storm(trx, vax, save=False)),
        }
        row = {"state": code, "name": name, "n_test": int(len(sub))}
        for haz in ("flood", "storm"):
            y = sub[f"y_{haz}"].to_numpy()
            a_pool, a_held = ap(y, pooled[haz](sub)), ap(y, held[haz](sub))
            keep = int(y.sum()) >= MIN_POS and a_pool
            row[haz] = {
                "n_pos": int(y.sum()),
                "pooled_ap": a_pool,
                "transfer_ap": a_held,
                "retention": a_held / a_pool if keep else None,
            }
            r = row[haz]["retention"]
            print(f"  {haz:6s} n_pos={y.sum():5d}  pooled={a_pool or 0:.5f}"
                  f"  held-out={a_held or 0:.5f}"
                  f"  retention={'n/a' if r is None else f'{r:.0%}'}")
        out["spatial"].append(row)

    # --------------------------------------------------- temporal decay
    # No refit: the question is whether the single trained model holds up as
    # the test year moves further from the training window.
    print("\n=== by test year ===")
    for yr, grp in te.groupby(te["DATE"].dt.year):
        row = {"year": int(yr), "n": int(len(grp))}
        for haz in ("flood", "storm"):
            y = grp[f"y_{haz}"].to_numpy()
            row[haz] = {"n_pos": int(y.sum()),
                        "ap": ap(y, pooled[haz](grp)) if y.sum() >= MIN_POS else None}
        out["temporal"].append(row)
        f, s = row["flood"], row["storm"]
        print(f"  {yr}  flood AP={f['ap'] or float('nan'):.5f} (n={f['n_pos']:4d})"
              f"   storm AP={s['ap'] or float('nan'):.5f} (n={s['n_pos']:4d})")

    # Headline: positive-weighted mean retention, so a state with three onsets
    # cannot swing the number as hard as Texas.
    for haz in ("flood", "storm"):
        rows = [r for r in out["spatial"] if r[haz]["retention"] is not None]
        w = sum(r[haz]["n_pos"] for r in rows)
        out[f"{haz}_retention"] = (
            sum(r[haz]["retention"] * r[haz]["n_pos"] for r in rows) / w if w else None
        )
        out[f"{haz}_states_scored"] = len(rows)

    (OUT_DIR / "proto_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"\nflood retention {out['flood_retention']:.1%}"
          f"   storm retention {out['storm_retention']:.1%}")
    print(f"wrote {OUT_DIR / 'proto_transfer.json'}")


if __name__ == "__main__":
    main()
