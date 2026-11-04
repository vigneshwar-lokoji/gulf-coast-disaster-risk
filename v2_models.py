"""The V2 hazard models, and the single place that defines how to score them.

Both consumers (export_v2, fetch_live) import from here, so the dashboard can
never show risk from a model other than the one the paper reports.

The boosters are read straight out of a sealed run directory rather than from a
loose `artifacts/models/`, because a run directory is hash-verified against the
panel it was trained on and a loose directory is whatever was written last. Three
times on this project a loose path produced numbers nobody could trace.

Scoring a saved booster has one trap. Early stopping keeps training for `early`
more rounds after the best one, and `save_model` writes all of them, so the file
holds `best_iteration + early` trees. Predicting with the whole file silently
scores an overfit model -- here the flood file has 387 trees for a best
iteration of 336. `iteration_range` is therefore mandatory, and `best_iteration`
comes from the run's own results JSON rather than from the file.
"""

import json
import os
import pathlib

import pandas as pd
import xgboost as xgb

# The run this dashboard reports. Pinned, not globbed: picking "the newest run"
# would let a smoke test silently become the published model.
RUN_ID = os.environ.get("CRISK_RUN_ID", "20261104T015759Z-run")
RUN_DIR = pathlib.Path(
    os.environ.get("CRISK_RUN_DIR", f"artifacts/runs/{RUN_ID}")
)

# Order matters: a DMatrix built from a DataFrame takes its feature names from
# the column order, and xgboost rejects a layout that differs from training.
FEATS = [
    "prcp", "prcp_max_stn", "prcp_prev3", "prcp_prev7", "prcp_prev14",
    "prcp_prev30", "prcp_prev60", "prcp_prev90", "prcp_fwd1", "prcp_fwd3",
    "prcp_fwd3_max", "ratio_prev3", "ratio_fwd3", "ratio_prev30", "log_fwd3",
    "tmax", "tmin", "month_sin", "month_cos", "hurricane_season",
    "lat", "lon", "pop_density", "RPL_THEME1", "RPL_THEME4", "RPL_THEMES",
]

HAZARDS = ("flood", "storm")
LEADS = (1, 2, 3)


def run_dir():
    if not RUN_DIR.exists():
        raise SystemExit(
            f"run directory not found: {RUN_DIR}\n"
            f"set CRISK_RUN_DIR, or unpack the sealed run zip into artifacts/runs/"
        )
    return RUN_DIR


def results(name):
    return json.loads((run_dir() / "results" / name).read_text())


def _scorer(path, best_iteration):
    """Wrap one booster as callable(DataFrame) -> probability array.

    Missing values are passed through rather than filled. XGBoost learned its
    own default direction for them during training, so filling here would score
    a different function than the one that was fitted.
    """
    booster = xgb.Booster()
    booster.load_model(str(path))
    if booster.num_features() != len(FEATS):
        raise SystemExit(
            f"{path.name} expects {booster.num_features()} features, "
            f"FEATS has {len(FEATS)} -- the model and this module disagree"
        )
    rng = (0, best_iteration + 1)

    def score(df):
        missing = [f for f in FEATS if f not in df.columns]
        if missing:
            raise KeyError(f"{path.name} needs columns {missing}")
        return booster.predict(xgb.DMatrix(df[FEATS]), iteration_range=rng)

    score.best_iteration = best_iteration
    score.rounds_in_file = booster.num_boosted_rounds()
    score.name = path.name
    return score


def load_scorers():
    """Pooled models: probability that onset falls anywhere in t+1..t+3."""
    res = results("xgb_run.json")
    return {
        haz: _scorer(run_dir() / "models" / f"xgb_{haz}_run.json",
                     res[haz]["best_iteration"])
        for haz in HAZARDS
    }


def load_lead_scorers():
    """Exact-lead models, keyed (hazard, lead).

    A separate model per lead rather than one model asked three times: the
    pooled target is "onset somewhere in the next three days", which is a
    different and easier question than "onset exactly three days from now".
    Reporting the pooled number as a 3-day forecast is the overstatement this
    sweep exists to avoid.
    """
    res = results("leads_run.json")
    out = {}
    for haz in HAZARDS:
        for k in LEADS:
            key = f"{haz}_l{k}"
            out[(haz, k)] = _scorer(
                run_dir() / "models" / f"xgb_{haz}_l{k}_run.json",
                res[key]["best_iteration"],
            )
    return out


def provenance():
    """What the dashboard should say about where its numbers came from."""
    man = json.loads((run_dir() / "manifest.json").read_text())
    return {
        "run_id": man["run_id"],
        "finished_utc": man["finished_utc"],
        "wall_seconds": man["wall_seconds"],
        "self_contained": man["self_contained"],
        "panel_sha256": man["panel"]["sha256"],
        "gpu": man["versions"].get("cuda"),
        "xgboost": man["versions"].get("xgboost"),
        "torch": man["versions"].get("torch"),
    }


if __name__ == "__main__":
    # Smoke check: reload every booster and confirm it reproduces the test AP
    # the run recorded. Catches a truncated download or a drifted FEATS list
    # before either reaches the dashboard.
    from sklearn.metrics import average_precision_score

    panel = pd.read_parquet("artifacts/daily_panel.parquet")
    panel["year"] = panel["DATE"].dt.year
    panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
    for h in HAZARDS:
        g = panel.groupby("county_fips", sort=False)[f"{h}_onset"]
        for k in LEADS:
            panel[f"y_{h}_l{k}"] = g.shift(-k).fillna(0).astype("int8")

    prm = results("params_run.json")
    cap = {h: int(v) for h, v in prm.get("label_end", {}).items()}
    test = panel[panel["year"].between(*prm["years"]["test"])]

    def rows(haz):
        c = cap.get(haz)
        return test if c is None else test[test["year"] <= c]

    print(json.dumps(provenance(), indent=2))
    bad = 0
    for (label, fn, target, haz, want) in (
        [(h, load_scorers()[h], f"y_{h}", h, results("xgb_run.json")[h]["test"]["ap"])
         for h in HAZARDS]
        + [(f"{h}_l{k}", load_lead_scorers()[(h, k)], f"y_{h}_l{k}", h,
            results("leads_run.json")[f"{h}_l{k}"]["test"]["ap"])
           for h in HAZARDS for k in LEADS]
    ):
        d = rows(haz)
        got = average_precision_score(d[target].to_numpy(), fn(d))
        ok = abs(got - want) < 5e-6
        bad += not ok
        print(f"  {label:10s} AP={got:.5f} recorded={want:.5f} "
              f"{'ok' if ok else 'MISMATCH'}  "
              f"({fn.best_iteration + 1} of {fn.rounds_in_file} trees)")
    raise SystemExit(bad and f"{bad} model(s) did not reproduce")
