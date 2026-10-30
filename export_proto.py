"""Export dashboard JSON: county metadata, real historical events, and model
risk for the windows a user will actually open.

Shipping risk for every county-day would be ~2M entries. Instead we ship the
windows around real FEMA incidents (what the demo navigates to) plus the most
recent observed stretch, which keeps the payload small.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from config import DATA_DIR, FILES, OUT_DIR
from proto_models import load_scorers

DOCS = "docs/data"
WINDOW = 7  # days either side of a real incident
RECENT_DAYS = 60


def county_meta(panel):
    """Every county the map draws, flagged for whether the model covers it.

    Names and centroids come from the map geometry so the whole lower 48 is
    present; modelled counties override the centroid with their station mean,
    which is the point the features were actually built from.
    """
    geo = json.loads((pathlib.Path(DOCS) / "geo.json").read_text())
    out = {
        f: [n[0], n[1], n[2], n[3], 0] for f, n in geo["names"].items()
    }

    m = pd.read_csv(DATA_DIR / FILES["mapping"])
    m["county_fips"] = m["county_fips"].astype("int64").astype(str).str.zfill(5)
    stn = (
        m.groupby("county_fips")
        .agg(name=("county_name", "first"), state=("state_abbr", "first"),
             lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"))
        .reset_index()
    )
    covered = set(panel["county_fips"].unique())
    for r in stn.itertuples(index=False):
        if r.county_fips in covered:
            out[r.county_fips] = [r.name, r.state, round(r.lat, 4),
                                  round(r.lon, 4), 1]
    return out


def main():
    panel = pd.read_parquet(OUT_DIR / "daily_panel.parquet")
    events = pd.read_parquet(OUT_DIR / "fema_events.parquet")
    metrics = json.loads((OUT_DIR / "proto_metrics.json").read_text())
    transfer = json.loads((OUT_DIR / "proto_transfer.json").read_text())
    scorers = load_scorers()

    # --- score the rows we need -------------------------------------------
    keep = pd.Series(False, index=panel.index)
    idx = pd.MultiIndex.from_frame(panel[["county_fips", "DATE"]])
    lookup = pd.Series(np.arange(len(panel)), index=idx)

    for r in events.itertuples(index=False):
        days = pd.date_range(
            r.begin.normalize() - pd.Timedelta(days=WINDOW),
            max(r.end, r.begin).normalize() + pd.Timedelta(days=WINDOW),
            freq="D",
        )
        rows = [lookup.get((r.county_fips, t), -1) for t in days]
        rows = [i for i in rows if i >= 0]
        keep.iloc[rows] = True

    last = panel["DATE"].max()
    keep |= (panel["DATE"] > last - pd.Timedelta(days=RECENT_DAYS)).to_numpy()

    sub = panel[keep].copy()
    print(f"scoring {len(sub):,} rows of {len(panel):,}")
    for haz, fn in scorers.items():
        sub[f"p_{haz}"] = fn(sub)

    # --- per-county historical event list ---------------------------------
    ev_out = {}
    for r in events.sort_values("begin", ascending=False).itertuples(index=False):
        haz = "flood" if r.is_flood else "storm"
        if r.is_flood and r.is_storm:
            haz = "both"
        ev_out.setdefault(r.county_fips, []).append(
            {
                "begin": r.begin.strftime("%Y-%m-%d"),
                "end": max(r.end, r.begin).strftime("%Y-%m-%d"),
                "hazard": haz,
                "type": r.incidentType,
                "title": str(r.declarationTitle)[:90],
            }
        )
    for k in ev_out:
        ev_out[k] = ev_out[k][:12]

    # --- risk by date -----------------------------------------------------
    sub["d"] = sub["DATE"].dt.strftime("%Y-%m-%d")
    by_day = {}
    cols = ["county_fips", "p_flood", "p_storm", "prcp_fwd3", "ratio_fwd3",
            "flood_active", "storm_active"]
    for day, grp in sub[["d"] + cols].groupby("d", sort=True):
        by_day[day] = [
            [
                r.county_fips,
                round(float(r.p_flood), 5),
                round(float(r.p_storm), 5),
                None if pd.isna(r.prcp_fwd3) else round(float(r.prcp_fwd3), 1),
                None if pd.isna(r.ratio_fwd3) else round(float(r.ratio_fwd3), 2),
                int(r.flood_active),
                int(r.storm_active),
            ]
            for r in grp.itertuples(index=False)
        ]

    # Raw probabilities are tiny because the event itself is rare. Bands are
    # quantiles of an unbiased sample of all county-days, so "High" means high
    # relative to a normal day rather than high in absolute terms.
    samp = panel.sample(n=min(150_000, len(panel)), random_state=0)
    bands = {}
    for haz, fn in scorers.items():
        s = fn(samp)
        bands[haz] = {
            q: float(np.quantile(s, v))
            for q, v in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p999", 0.999))
        }

    meta = county_meta(panel)
    payload_meta = {
        "bands": bands,
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed_through": str(panel["DATE"].max().date()),
        "n_counties_modeled": sum(v[4] for v in meta.values()),
        "n_counties_mapped": len(meta),
        "horizon_days": 3,
        "splits": metrics["splits"],
        "target": metrics["target"],
        "transfer": transfer,
        "flood": {k: metrics["flood"][k] for k in (
            "model", "features", "test_ap", "test_prevalence",
            "lift_over_prevalence", "best_baseline", "best_baseline_ap",
            "vs_best_baseline", "n_test_pos", "n_train_pos")},
        "storm": {k: metrics["storm"][k] for k in (
            "model", "features", "test_ap", "test_prevalence",
            "lift_over_prevalence", "best_baseline", "best_baseline_ap",
            "vs_best_baseline", "n_test_pos", "n_train_pos", "top_features")},
        "counties": meta,
    }

    p = pathlib.Path(DOCS)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(json.dumps(payload_meta, separators=(",", ":")))
    (p / "events.json").write_text(json.dumps(ev_out, separators=(",", ":")))
    (p / "history.json").write_text(
        json.dumps({"days": by_day}, separators=(",", ":"))
    )

    for f in ("meta.json", "events.json", "history.json"):
        print(f"  {f}: {(p/f).stat().st_size/1024:.0f} KB")
    print(f"days covered: {len(by_day):,}")


if __name__ == "__main__":
    main()
