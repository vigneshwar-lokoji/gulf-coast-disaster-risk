"""Build the county-DAY panel for the prototype.

V1 aggregated to county-month, which cannot answer "is there a warning today".
This builds the same feature families at daily resolution and defines the
forecast target as "does a FEMA flood/storm incident BEGIN in the next 3 days",
which is well posed in a way that an active-window label is not: an active
window is trivially predictable from yesterday's label.

Writes artifacts/daily_panel.parquet.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, FILES, DROP_INCIDENT_TYPES, STORM_TYPES, OUT_DIR

START = 2010
CLIM_YEARS = (2010, 2018)  # frozen; never sees val/test distribution
HORIZON = 3  # predict an incident beginning within the next N days


def _fips(s):
    return s.astype("int64").astype(str).str.zfill(5)


def county_daily():
    w = pd.read_parquet(
        DATA_DIR / FILES["weather"], columns=["STATION", "DATE", "PRCP", "TMAX", "TMIN"]
    )
    w = w[w["DATE"].dt.year >= START]
    m = pd.read_csv(DATA_DIR / FILES["mapping"])
    m["county_fips"] = _fips(m["county_fips"])
    w = w.merge(m[["STATION", "county_fips"]], on="STATION", how="inner")

    d = (
        w.groupby(["county_fips", "DATE"])
        .agg(
            prcp=("PRCP", "mean"),
            prcp_max_stn=("PRCP", "max"),
            tmax=("TMAX", "mean"),
            tmin=("TMIN", "mean"),
        )
        .reset_index()
    )

    # Complete the grid so rolling windows span calendar time, not report days.
    full = pd.MultiIndex.from_product(
        [
            sorted(d["county_fips"].unique()),
            pd.date_range(d["DATE"].min(), d["DATE"].max(), freq="D"),
        ],
        names=["county_fips", "DATE"],
    )
    return d.set_index(["county_fips", "DATE"]).reindex(full).reset_index()


def add_features(d):
    d = d.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
    g = d.groupby("county_fips", sort=False)["prcp"]

    for w in (3, 7, 14, 30, 60, 90):
        d[f"prcp_prev{w}"] = g.transform(
            lambda s, w=w: s.rolling(w, min_periods=max(1, w // 3)).sum()
        )

    # Forward-looking rain. At inference this comes from the weather forecast,
    # which is what makes a lead-time claim possible at all.
    for w in (1, 3):
        d[f"prcp_fwd{w}"] = g.transform(
            lambda s, w=w: s.shift(-w).rolling(w, min_periods=1).sum()
        )
    d["prcp_fwd3_max"] = g.transform(
        lambda s: s.shift(-3).rolling(3, min_periods=1).max()
    )

    d["doy"] = d["DATE"].dt.dayofyear
    d["month"] = d["DATE"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["hurricane_season"] = d["month"].between(6, 11).astype("int8")

    # Anomaly ratios against a frozen climatology, so "a lot of rain" is
    # relative to what that county normally gets rather than absolute mm.
    clim_mask = d["DATE"].dt.year.between(*CLIM_YEARS)
    clim = (
        d[clim_mask]
        .groupby("county_fips")
        .agg(
            p95_prev3=("prcp_prev3", lambda s: s.quantile(0.95)),
            p95_fwd3=("prcp_fwd3", lambda s: s.quantile(0.95)),
            mean_prev30=("prcp_prev30", "mean"),
        )
        .reset_index()
    )
    d = d.merge(clim, on="county_fips", how="left")
    d["ratio_prev3"] = d["prcp_prev3"] / d["p95_prev3"].replace(0, np.nan)
    d["ratio_fwd3"] = d["prcp_fwd3"] / d["p95_fwd3"].replace(0, np.nan)
    d["ratio_prev30"] = d["prcp_prev30"] / d["mean_prev30"].replace(0, np.nan)
    d["log_fwd3"] = np.log1p(d["prcp_fwd3"].clip(lower=0))
    return d


def static_covariates():
    svi = pd.read_csv(DATA_DIR / FILES["svi"])
    svi["county_fips"] = _fips(svi["FIPS"])
    svi = svi[
        [
            "county_fips",
            "ST_ABBR",
            "AREA_SQMI",
            "E_TOTPOP",
            "RPL_THEME1",
            "RPL_THEME4",
            "RPL_THEMES",
        ]
    ].rename(columns={"ST_ABBR": "state"})
    svi["pop_density"] = svi["E_TOTPOP"] / svi["AREA_SQMI"].replace(0, np.nan)

    m = pd.read_csv(DATA_DIR / FILES["mapping"])
    m["county_fips"] = _fips(m["county_fips"])
    geo = (
        m.groupby("county_fips")
        .agg(lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"),
             n_stations=("STATION", "nunique"))
        .reset_index()
    )
    return geo.merge(svi, on="county_fips", how="inner")


def fema_events():
    """Return one row per (county, incident) with hazard flags and dates."""
    f = pd.read_csv(DATA_DIR / FILES["fema"], low_memory=False)
    f["county_fips"] = _fips(f["county_fips"])
    f = f[~f["incidentType"].isin(DROP_INCIDENT_TYPES)].copy()
    f["begin"] = pd.to_datetime(f["incidentBeginDate"], errors="coerce").dt.tz_localize(None)
    f["end"] = pd.to_datetime(f["incidentEndDate"], errors="coerce").dt.tz_localize(None)
    f = f.dropna(subset=["begin"])
    f["end"] = f["end"].fillna(f["begin"])
    f = f[f["begin"].dt.year >= START]

    title = f["declarationTitle"].fillna("").str.upper()
    f["is_flood"] = ((f["incidentType"] == "Flood") | title.str.contains("FLOOD")).astype(int)
    f["is_storm"] = f["incidentType"].isin(STORM_TYPES).astype(int)
    f = f[(f["is_flood"] == 1) | (f["is_storm"] == 1)]

    cols = ["county_fips", "begin", "end", "is_flood", "is_storm",
            "incidentType", "declarationTitle"]
    return (
        f[cols]
        .drop_duplicates(subset=["county_fips", "begin", "declarationTitle"])
        .reset_index(drop=True)
    )


def add_labels(d, ev):
    """Active-window labels, plus 'incident begins within next HORIZON days'."""
    d = d.copy()
    for c in ("flood_active", "storm_active", "flood_onset", "storm_onset"):
        d[c] = 0

    idx = pd.MultiIndex.from_frame(d[["county_fips", "DATE"]])
    pos = pd.Series(np.arange(len(d)), index=idx)

    for r in ev.itertuples(index=False):
        days = pd.date_range(r.begin.normalize(), max(r.end, r.begin).normalize(), freq="D")
        try:
            rows = pos.loc[[(r.county_fips, t) for t in days]].to_numpy()
        except KeyError:
            rows = np.array(
                [pos.get((r.county_fips, t), -1) for t in days]
            )
            rows = rows[rows >= 0]
        if len(rows) == 0:
            continue
        if r.is_flood:
            d.iloc[rows, d.columns.get_loc("flood_active")] = 1
            d.iloc[rows[0], d.columns.get_loc("flood_onset")] = 1
        if r.is_storm:
            d.iloc[rows, d.columns.get_loc("storm_active")] = 1
            d.iloc[rows[0], d.columns.get_loc("storm_onset")] = 1

    # Target: does an incident begin in [t+1, t+HORIZON]?
    d = d.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
    for haz in ("flood", "storm"):
        g = d.groupby("county_fips", sort=False)[f"{haz}_onset"]
        d[f"y_{haz}"] = (
            g.transform(
                lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).max()
            )
            .fillna(0)
            .astype("int8")
        )
    return d


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("aggregating stations to county-day ...", flush=True)
    d = county_daily()
    print(f"  {d.shape[0]:,} county-days, {d['county_fips'].nunique()} counties")

    print("features ...", flush=True)
    d = add_features(d)

    print("static covariates ...", flush=True)
    d = d.merge(static_covariates(), on="county_fips", how="inner")

    print("labels ...", flush=True)
    ev = fema_events()
    print(f"  {len(ev):,} county-incidents")
    d = add_labels(d, ev)

    # Drop the warm-up period where antecedent windows are incomplete.
    d = d[d["DATE"] >= pd.Timestamp(f"{START}-04-01")]
    d = d[d["prcp"].notna() | d["prcp_prev30"].notna()]

    print(f"\npanel: {d.shape[0]:,} rows x {d.shape[1]} cols")
    for haz in ("flood", "storm"):
        print(f"  y_{haz}: {d[f'y_{haz}'].mean()*100:.3f}%  ({int(d[f'y_{haz}'].sum()):,} positives)")

    d.to_parquet(OUT_DIR / "daily_panel.parquet", index=False)
    ev.to_parquet(OUT_DIR / "fema_events.parquet", index=False)
    print(f"\nwrote {OUT_DIR/'daily_panel.parquet'}")


if __name__ == "__main__":
    main()
