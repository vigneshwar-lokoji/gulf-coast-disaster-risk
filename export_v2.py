"""Export the dashboard payload for the V2 models.

Replaces export_proto.py, which served the V1 prototype: a FEMA-declaration
target, 2010-2019 training, and a one-feature logistic for flood. None of that
is what V2 trained, so the old exporter could only have published numbers the
paper does not contain.

Three things changed that matter here:

  label   NOAA StormEvents flood episodes and HURDAT2 cyclone passes, not FEMA
          declarations. A declaration is an administrative act that depends on
          damage estimates and a governor's request, so V1 scored counties for
          paperwork as much as for weather.
  models  XGBoost for both hazards, from a sealed hash-verified run, plus the
          exact-lead models the lead sweep produced.
  leads   A separate model per lead, so the dashboard can show the +1/+2/+3 day
          forecasts rather than implying the pooled 3-day number holds at day 3.

Shipping risk for every county-day would be ~3.2M rows. Instead we ship the
windows around the events the dashboard actually lists, plus the most recent
observed stretch, which is what a visitor can navigate to.
"""

import json
import pathlib

import numpy as np
import pandas as pd

import v2_models
from v2_models import FEATS, HAZARDS, LEADS

ARTIFACTS = pathlib.Path("artifacts")
DOCS = pathlib.Path("docs/data")

WINDOW = 5          # days either side of a listed event
RECENT_DAYS = 60
PER_COUNTY = {"flood": 6, "storm": 4}   # events listed per county, per hazard
BAND_SAMPLE = 150_000


def fips5(s):
    """Both event tables key on county, one as int and one already padded."""
    return s.astype("int64").astype(str).str.zfill(5) if s.dtype.kind in "iu" \
        else s.astype(str).str.zfill(5)


def load_events():
    """One table of real hazard occurrences, both hazards, with a display title.

    These are the observational records the labels were built from, so an event
    listed here is by construction an event the model was scored against -- the
    dashboard cannot show an outcome the training never saw.
    """
    fl = pd.read_parquet(ARTIFACTS / "noaa" / "noaa_flood_events.parquet")
    fl["county_fips"] = fips5(fl["county_fips"])
    fl["hazard"] = "flood"
    fl["type"] = "Flood"
    fl["title"] = ("NOAA StormEvents flood episode "
                   + fl["EPISODE_ID"].astype("int64").astype(str))

    st = pd.read_parquet(ARTIFACTS / "noaa" / "hurdat_events.parquet")
    st["county_fips"] = fips5(st["county_fips"])
    st["hazard"] = "storm"
    st["type"] = "Tropical cyclone"
    st["title"] = ("HURDAT2 track " + st["storm_id"].astype(str)
                   + ", peak wind " + st["peak_wind_kt"].round().astype("int64")
                     .astype(str) + " kt over the county")

    cols = ["county_fips", "begin", "end", "hazard", "type", "title"]
    ev = pd.concat([fl[cols], st[cols]], ignore_index=True)
    ev["begin"] = pd.to_datetime(ev["begin"]).dt.normalize()
    ev["end"] = pd.to_datetime(ev["end"]).dt.normalize()
    ev["end"] = ev[["begin", "end"]].max(axis=1)

    # One NOAA episode spans several rows per county, one per sub-event inside
    # it, and HURDAT lists a county once per six-hourly track point. Left as-is
    # the same flood turns up three times in a county's history and crowds out
    # distinct events, so collapse to one row per county, hazard and start day,
    # keeping the longest duration seen.
    ev = (ev.sort_values("end", ascending=False)
            .drop_duplicates(["county_fips", "hazard", "begin"], keep="first"))
    return ev.sort_values("begin", ascending=False).reset_index(drop=True)


def listed(ev, covered):
    """The most recent events per county and hazard, capped.

    Capped per hazard rather than overall: a Gulf county logs far more flood
    episodes than cyclone passes, so a single recency cap would list floods only
    and hide the hurricane side of the model entirely.
    """
    ev = ev[ev["county_fips"].isin(covered)]
    keep = [g.head(PER_COUNTY[h]) for (_, h), g
            in ev.groupby(["county_fips", "hazard"], sort=False)]
    return pd.concat(keep, ignore_index=True)


def county_meta(panel, geo):
    """Every county the map draws, flagged for whether a model covers it.

    Names and centroids come from the map geometry so the whole lower 48 is
    present; modelled counties override the centroid with the mean position of
    their rainfall stations, which is the point the features were built from.
    """
    out = {f: [n[0], n[1], n[2], n[3], 0] for f, n in geo["names"].items()}
    stn = panel.groupby("county_fips")[["lat", "lon"]].mean()
    for fips, r in stn.iterrows():
        base = out.get(fips)
        name = base[0] if base else fips
        state = base[1] if base else ""
        out[fips] = [name, state, round(float(r.lat), 4),
                     round(float(r.lon), 4), 1]
    return out


def main():
    prm = v2_models.results("params_run.json")
    xgb_res = v2_models.results("xgb_run.json")
    tft_res = v2_models.results("tft_run.json")
    lead_res = v2_models.results("leads_run.json")
    tft_lead_res = v2_models.results("tft_leads_run.json")

    pooled = v2_models.load_scorers()
    per_lead = v2_models.load_lead_scorers()

    panel = pd.read_parquet(ARTIFACTS / "daily_panel.parquet")
    panel["year"] = panel["DATE"].dt.year
    lo, hi = prm["years"]["train"][0], prm["years"]["test"][1]
    panel = panel[panel["year"].between(lo, hi)]
    panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
    print(f"panel {len(panel):,} rows  {panel['year'].min()}-{panel['year'].max()}"
          f"  {panel['county_fips'].nunique()} counties")

    geo = json.loads((DOCS / "geo.json").read_text())
    covered = set(panel["county_fips"].unique())
    events = load_events()
    shown = listed(events, covered)
    print(f"events: {len(events):,} on record, {len(shown):,} listed "
          f"across {shown['county_fips'].nunique()} counties")

    # --- pick the rows worth scoring ---------------------------------------
    pos = pd.Series(np.arange(len(panel)),
                    index=pd.MultiIndex.from_arrays(
                        [panel["county_fips"], panel["DATE"]]))
    want = np.zeros(len(panel), dtype=bool)
    for r in shown.itertuples(index=False):
        days = pd.date_range(r.begin - pd.Timedelta(days=WINDOW),
                             r.end + pd.Timedelta(days=WINDOW), freq="D")
        idx = [i for i in (pos.get((r.county_fips, d), -1) for d in days) if i >= 0]
        want[idx] = True
    last = panel["DATE"].max()
    want |= (panel["DATE"] > last - pd.Timedelta(days=RECENT_DAYS)).to_numpy()

    sub = panel[want].copy()
    print(f"scoring {len(sub):,} of {len(panel):,} county-days")
    for haz in HAZARDS:
        sub[f"p_{haz}"] = pooled[haz](sub)
        for k in LEADS:
            sub[f"p_{haz}_l{k}"] = per_lead[(haz, k)](sub)

    # --- ground truth: was an event in progress on this day? ---------------
    # Taken from the event intervals rather than from a label column: y_{haz}
    # is forward-looking (onset within three days), so it is a target, not a
    # statement about what was happening on the day itself.
    for haz in HAZARDS:
        sub[f"{haz}_active"] = False
    key = pd.MultiIndex.from_arrays([sub["county_fips"], sub["DATE"]])
    at = pd.Series(np.arange(len(sub)), index=key)
    for r in events[events["county_fips"].isin(covered)].itertuples(index=False):
        days = pd.date_range(r.begin, r.end, freq="D")
        idx = [i for i in (at.get((r.county_fips, d), -1) for d in days) if i >= 0]
        if idx:
            sub.iloc[idx, sub.columns.get_loc(f"{r.hazard}_active")] = True

    # --- per-county event list, with what the models said beforehand -------
    split_of = {y: name for name, (a, b) in prm["years"].items()
                for y in range(a, b + 1)}
    scored = sub.set_index(["county_fips", "DATE"])

    def lead_up(fips, begin, haz):
        """Two readings of the same three days before onset.

        `pre` is the pooled model, which answers "onset within three days" and
        so should rise as onset approaches. `lead` is the exact-lead model that
        would actually have been run that morning -- the +3 day model three days
        out, the +1 day model the day before. The second is the operationally
        honest number and is usually the lower of the two.
        """
        pre, lead = [], []
        for k in (3, 2, 1):
            day = begin - pd.Timedelta(days=k)
            try:
                row = scored.loc[(fips, day)]
            except KeyError:
                pre.append(None)
                lead.append(None)
                continue
            pre.append(round(float(row[f"p_{haz}"]), 5))
            lead.append(round(float(row[f"p_{haz}_l{k}"]), 5))
        try:
            rain = scored.loc[(fips, begin - pd.Timedelta(days=1)), "prcp_fwd3"]
            rain = None if pd.isna(rain) else round(float(rain), 1)
        except KeyError:
            rain = None
        return pre, lead, rain

    ev_out = {}
    for r in shown.sort_values("begin", ascending=False).itertuples(index=False):
        pre, lead, rain = lead_up(r.county_fips, r.begin, r.hazard)
        e = {"begin": r.begin.strftime("%Y-%m-%d"),
             "end": r.end.strftime("%Y-%m-%d"),
             "hazard": r.hazard, "type": r.type, "title": r.title[:110],
             "split": split_of.get(r.begin.year)}
        if any(v is not None for v in pre):
            e["pre"] = {r.hazard: pre}
            e["lead"] = {r.hazard: lead}
            e["rain"] = rain
        ev_out.setdefault(r.county_fips, []).append(e)

    n_listed = sum(len(v) for v in ev_out.values())
    n_with = sum(1 for v in ev_out.values() for e in v if "pre" in e)
    print(f"events with a model lead-up: {n_with:,} of {n_listed:,}")

    # A county's panel only holds days its rainfall stations reported, and some
    # stations have multi-year outages -- one county has 3,702 of 9,645 days.
    # Those gaps are why some listed events carry no lead-up, so the number is
    # published rather than left for a visitor to discover as a blank card.
    days_per = panel.groupby("county_fips").size()
    span = int((panel["DATE"].max() - panel["DATE"].min()).days) + 1
    coverage = {
        "calendar_days": span,
        "median_days_per_county": int(days_per.median()),
        "min_days_per_county": int(days_per.min()),
        "counties_under_80pct": int((days_per < 0.8 * span).sum()),
        "events_listed_with_leadup": n_with,
        "events_listed": n_listed,
    }

    # --- risk by date ------------------------------------------------------
    sub["d"] = sub["DATE"].dt.strftime("%Y-%m-%d")
    cols = (["county_fips", "p_flood", "p_storm", "prcp_fwd3", "ratio_fwd3",
             "flood_active", "storm_active"]
            + [f"p_flood_l{k}" for k in LEADS] + [f"p_storm_l{k}" for k in LEADS])
    by_day = {}
    for day, grp in sub[["d"] + cols].groupby("d", sort=True):
        by_day[day] = [
            [r[0], round(float(r[1]), 5), round(float(r[2]), 5),
             None if pd.isna(r[3]) else round(float(r[3]), 1),
             None if pd.isna(r[4]) else round(float(r[4]), 2),
             int(r[5]), int(r[6])]
            + [round(float(v), 5) for v in r[7:]]
            for r in grp[cols].itertuples(index=False, name=None)
        ]

    # Raw probabilities are tiny because the event itself is rare. Bands are
    # quantiles of an unbiased sample of all county-days, so "High" means high
    # relative to a normal day rather than high in absolute terms. Each lead has
    # its own quantiles: a +3 day model sees roughly a third of the pooled
    # prevalence, so reusing the pooled cut-offs would paint every day green.
    samp = panel.sample(n=min(BAND_SAMPLE, len(panel)), random_state=0)
    qs = (("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p999", 0.999))
    bands = {}
    for haz in HAZARDS:
        s = pooled[haz](samp)
        bands[haz] = {q: float(np.quantile(s, v)) for q, v in qs}
        for k in LEADS:
            s = per_lead[(haz, k)](samp)
            bands[f"{haz}_l{k}"] = {q: float(np.quantile(s, v)) for q, v in qs}

    # --- metrics the dashboard reports -------------------------------------
    def model_block(haz):
        x, t = xgb_res[haz], tft_res[haz]
        return {
            "model": "xgboost_monotone",
            "n_features": len(FEATS),
            "test_ap": x["test"]["ap"],
            "test_prevalence": x["test"]["prevalence"],
            "lift_over_prevalence": x["test"]["lift"],
            "val_ap": x["val"]["ap"],
            "best_baseline": x["best_baseline"],
            "best_baseline_ap": x["best_baseline_ap"],
            "vs_best_baseline": x["vs_best_baseline"],
            "n_train_pos": x["n_train_pos"],
            "n_test_pos": x["n_test_pos"],
            "best_iteration": x["best_iteration"],
            "top_features": x["top_features"][:8],
            "baselines": dict(sorted(x["baselines"].items(),
                                     key=lambda kv: -kv[1])[:6]),
            "tft": {"test_ap": t["test"]["ap"],
                    "lift_over_prevalence": t["test"]["lift"],
                    "vs_best_baseline": t["vs_best_baseline"],
                    "epochs_run": t["epochs_run"],
                    "best_val_loss": t["best_val_loss"],
                    "n_test_windows": t["n_test_windows"],
                    "calibration": t["calibration"]["method"]},
        }

    leads_block = [
        {"hazard": haz, "lead": k,
         "prevalence": lead_res[f"{haz}_l{k}"]["test"]["prevalence"],
         "baseline": lead_res[f"{haz}_l{k}"]["best_baseline"],
         "baseline_ap": lead_res[f"{haz}_l{k}"]["best_baseline_ap"],
         "xgb_ap": lead_res[f"{haz}_l{k}"]["test"]["ap"],
         "xgb_vs_base": lead_res[f"{haz}_l{k}"]["vs_best_baseline"],
         "xgb_lift": lead_res[f"{haz}_l{k}"]["test"]["lift"],
         "tft_ap": tft_lead_res[f"{haz}_l{k}"]["test"]["ap"],
         "tft_vs_base": tft_lead_res[f"{haz}_l{k}"]["vs_best_baseline"],
         "tft_lift": tft_lead_res[f"{haz}_l{k}"]["test"]["lift"],
         "n_test_pos": lead_res[f"{haz}_l{k}"]["n_test_pos"]}
        for haz in HAZARDS for k in LEADS
    ]

    meta = county_meta(panel, geo)
    payload = {
        "version": "v2",
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed_through": str(panel["DATE"].max().date()),
        "run": v2_models.provenance(),
        "bands": bands,
        "horizon_days": prm["horizon_days"],
        "leads": list(LEADS),
        "splits": prm["years"],
        "label_end": prm.get("label_end", {}),
        "target": ("a NOAA-recorded flood episode or HURDAT2 cyclone pass "
                   "begins in this county within the next 3 days"),
        "label_source": {
            "flood": "NOAA StormEvents flood episodes, 2000-2026",
            "storm": "HURDAT2 Atlantic best track, county passes through 2024",
        },
        "n_counties_modeled": sum(v[4] for v in meta.values()),
        "n_counties_mapped": len(meta),
        "n_events_listed": int(len(shown)),
        "n_events_total": int(len(events)),
        "coverage": coverage,
        "flood": model_block("flood"),
        "storm": model_block("storm"),
        "lead_table": leads_block,
        "counties": meta,
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "meta.json").write_text(json.dumps(payload, separators=(",", ":")))
    (DOCS / "events.json").write_text(json.dumps(ev_out, separators=(",", ":")))
    (DOCS / "history.json").write_text(
        json.dumps({"days": by_day}, separators=(",", ":")))

    for f in ("meta.json", "events.json", "history.json"):
        print(f"  {f}: {(DOCS / f).stat().st_size / 1024:.0f} KB")
    print(f"days covered: {len(by_day):,}   "
          f"counties modelled: {payload['n_counties_modeled']}")


if __name__ == "__main__":
    main()
