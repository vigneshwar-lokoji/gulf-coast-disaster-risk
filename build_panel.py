"""Build the county-month panel: hazard features, FEMA flood/storm labels, SVI.

Run once. Writes artifacts/panel.parquet, which is small enough to upload to
Colab directly (a few MB versus the 72MB raw weather parquet).
"""

import numpy as np
import pandas as pd

from config import (
    CLIMATOLOGY_YEARS,
    DATA_DIR,
    DROP_INCIDENT_TYPES,
    FILES,
    HEAVY_DAY_MM,
    LEADS,
    OUT_DIR,
    START_YEAR,
    STORM_TYPES,
    WET_DAY_MM,
)


def _fips(s):
    return s.astype("int64").astype(str).str.zfill(5)


def load_mapping():
    m = pd.read_csv(DATA_DIR / FILES["mapping"])
    m["county_fips"] = _fips(m["county_fips"])
    return m


def load_static():
    svi = pd.read_csv(DATA_DIR / FILES["svi"])
    svi["county_fips"] = _fips(svi["FIPS"])
    svi = svi[
        [
            "county_fips",
            "ST_ABBR",
            "AREA_SQMI",
            "E_TOTPOP",
            "RPL_THEME1",
            "RPL_THEME2",
            "RPL_THEME3",
            "RPL_THEME4",
            "RPL_THEMES",
        ]
    ].rename(columns={"ST_ABBR": "state"})
    svi["pop_density"] = svi["E_TOTPOP"] / svi["AREA_SQMI"].replace(0, np.nan)

    m = load_mapping()
    geo = (
        m.groupby("county_fips")
        .agg(
            lat=("LATITUDE", "mean"),
            lon=("LONGITUDE", "mean"),
            n_stations=("STATION", "nunique"),
        )
        .reset_index()
    )

    st = pd.read_csv(DATA_DIR / FILES["stations"])
    st = st.merge(
        m[["STATION", "county_fips"]], left_on="ID", right_on="STATION", how="inner"
    )
    elev = (
        st.groupby("county_fips")["ELEVATION"].mean().rename("elevation").reset_index()
    )

    return geo.merge(elev, on="county_fips", how="left").merge(
        svi, on="county_fips", how="inner"
    )


def load_labels():
    f = pd.read_csv(DATA_DIR / FILES["fema"], low_memory=False)
    f["county_fips"] = _fips(f["county_fips"])
    f = f[~f["incidentType"].isin(DROP_INCIDENT_TYPES)].copy()

    f["begin"] = pd.to_datetime(f["incidentBeginDate"], utc=True, errors="coerce")
    f["end"] = pd.to_datetime(f["incidentEndDate"], utc=True, errors="coerce")
    f = f.dropna(subset=["begin"])
    f["end"] = f["end"].fillna(f["begin"])
    f = f[f["begin"].dt.year >= START_YEAR]

    title = f["declarationTitle"].fillna("").str.upper()
    # FEMA files most Gulf Coast floods under 'Severe Storm' with flooding named
    # only in the title; incidentType alone recovers barely half of them.
    f["is_flood"] = (f["incidentType"] == "Flood") | title.str.contains("FLOOD")
    f["is_storm"] = f["incidentType"].isin(STORM_TYPES)

    # Two label views. 'active' marks every month a declaration spans, which
    # makes continuation trivially predictable from the previous month. 'onset'
    # marks only the month a declaration begins, which is the quantity a
    # decision-maker actually needs and the only well-posed forecast target.
    rows = []
    for r in f.itertuples(index=False):
        months = pd.period_range(
            r.begin.to_period("M"), max(r.end, r.begin).to_period("M"), freq="M"
        )
        for i, mo in enumerate(months):
            rows.append((r.county_fips, mo, r.is_flood, r.is_storm, i == 0))

    lab = pd.DataFrame(
        rows, columns=["county_fips", "month", "is_flood", "is_storm", "is_onset"]
    )
    lab["onset_flood"] = lab["is_flood"] & lab["is_onset"]
    lab["onset_storm"] = lab["is_storm"] & lab["is_onset"]

    return (
        lab.groupby(["county_fips", "month"])
        .agg(
            y_flood=("is_flood", "max"),
            y_storm=("is_storm", "max"),
            y_flood_onset=("onset_flood", "max"),
            y_storm_onset=("onset_storm", "max"),
        )
        .astype("int8")
        .reset_index()
    )


def build_county_daily():
    w = pd.read_parquet(
        DATA_DIR / FILES["weather"],
        columns=["STATION", "DATE", "PRCP", "TMAX", "TMIN"],
    )
    w = w[w["DATE"].dt.year >= START_YEAR]

    m = load_mapping()[["STATION", "county_fips"]]
    w = w.merge(m, on="STATION", how="inner")

    daily = (
        w.groupby(["county_fips", "DATE"])
        .agg(
            prcp=("PRCP", "mean"),
            prcp_max_stn=("PRCP", "max"),
            tmax=("TMAX", "mean"),
            tmin=("TMIN", "mean"),
        )
        .reset_index()
    )

    # Complete the daily grid so rolling windows span real calendar time rather
    # than counting only the days a station happened to report.
    full = pd.MultiIndex.from_product(
        [
            sorted(daily["county_fips"].unique()),
            pd.date_range(daily["DATE"].min(), daily["DATE"].max(), freq="D"),
        ],
        names=["county_fips", "DATE"],
    )
    return (
        daily.set_index(["county_fips", "DATE"]).reindex(full).reset_index()
    )


def add_daily_rollings(daily):
    daily = daily.sort_values(["county_fips", "DATE"])
    g = daily.groupby("county_fips", sort=False)["prcp"]
    for w in (3, 7, 30, 60, 90):
        daily[f"prcp_roll{w}"] = g.transform(
            lambda s, w=w: s.rolling(w, min_periods=max(1, w // 3)).sum()
        )
    return daily


def aggregate_monthly(daily, p95):
    daily = daily.merge(p95, on="county_fips", how="left")
    daily["is_wet"] = (daily["prcp"] >= WET_DAY_MM).astype("float32")
    daily["is_heavy"] = (daily["prcp"] >= HEAVY_DAY_MM).astype("float32")
    daily["is_p95"] = (daily["prcp"] >= daily["prcp_p95"]).astype("float32")
    daily.loc[daily["prcp"].isna(), ["is_wet", "is_heavy", "is_p95"]] = np.nan
    daily["month"] = daily["DATE"].dt.to_period("M")

    out = (
        daily.groupby(["county_fips", "month"])
        .agg(
            prcp_sum=("prcp", "sum"),
            prcp_max=("prcp", "max"),
            prcp_mean=("prcp", "mean"),
            prcp_std=("prcp", "std"),
            prcp_max3d=("prcp_roll3", "max"),
            prcp_max7d=("prcp_roll7", "max"),
            n_wet=("is_wet", "sum"),
            n_heavy=("is_heavy", "sum"),
            n_p95=("is_p95", "sum"),
            tmax_mean=("tmax", "mean"),
            tmax_max=("tmax", "max"),
            tmin_mean=("tmin", "mean"),
            tmin_min=("tmin", "min"),
            obs_days=("prcp", "count"),
        )
        .reset_index()
    )

    # Antecedent wetness as of the final day of the month.
    eom = (
        daily.sort_values("DATE")
        .groupby(["county_fips", "month"])[["prcp_roll30", "prcp_roll60", "prcp_roll90"]]
        .last()
        .reset_index()
        .rename(
            columns={
                "prcp_roll30": "antecedent_30d",
                "prcp_roll60": "antecedent_60d",
                "prcp_roll90": "antecedent_90d",
            }
        )
    )
    return out.merge(eom, on=["county_fips", "month"], how="left")


def daily_p95(daily):
    lo, hi = CLIMATOLOGY_YEARS
    ref = daily[(daily["DATE"].dt.year >= lo) & (daily["DATE"].dt.year <= hi)]
    wet = ref[ref["prcp"] >= WET_DAY_MM]
    return (
        wet.groupby("county_fips")["prcp"]
        .quantile(0.95)
        .rename("prcp_p95")
        .reset_index()
    )


def add_climatology_anomalies(monthly):
    lo, hi = CLIMATOLOGY_YEARS
    monthly["moy"] = monthly["month"].dt.month
    ref = monthly[(monthly["month"].dt.year >= lo) & (monthly["month"].dt.year <= hi)]

    cols = ["prcp_sum", "prcp_max", "tmax_mean", "tmin_mean", "antecedent_90d"]
    clim = (
        ref.groupby(["county_fips", "moy"])[cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    clim.columns = ["county_fips", "moy"] + [
        f"{c}_clim_{s}" for c in cols for s in ("mean", "std")
    ]

    monthly = monthly.merge(clim, on=["county_fips", "moy"], how="left")
    for c in cols:
        sd = monthly[f"{c}_clim_std"].replace(0, np.nan)
        monthly[f"{c}_anom"] = (monthly[c] - monthly[f"{c}_clim_mean"]) / sd
    return monthly.drop(
        columns=[f"{c}_clim_{s}" for c in cols for s in ("mean", "std")]
    )


def add_lags(monthly):
    monthly = monthly.sort_values(["county_fips", "month"])
    g = monthly.groupby("county_fips", sort=False)
    for col in ["prcp_sum", "prcp_max", "n_heavy", "prcp_sum_anom"]:
        for lag in (1, 2, 3, 6, 12):
            monthly[f"{col}_lag{lag}"] = g[col].shift(lag)
    return monthly


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("building county-daily series...")
    daily = build_county_daily()
    daily = add_daily_rollings(daily)
    print(f"  {len(daily):,} county-days, {daily.county_fips.nunique()} counties")

    print("aggregating to county-month...")
    monthly = aggregate_monthly(daily, daily_p95(daily))
    monthly = add_climatology_anomalies(monthly)
    monthly = add_lags(monthly)
    del daily

    print("joining static + labels...")
    static = load_static()
    labels = load_labels()

    panel = monthly.merge(static, on="county_fips", how="inner")
    panel = panel.merge(labels, on=["county_fips", "month"], how="left")
    lab_cols = ["y_flood", "y_storm", "y_flood_onset", "y_storm_onset"]
    panel[lab_cols] = panel[lab_cols].fillna(0).astype("int8")

    panel["year"] = panel["month"].dt.year
    panel["month_sin"] = np.sin(2 * np.pi * panel["moy"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["moy"] / 12)
    panel["hurricane_season"] = panel["moy"].between(6, 11).astype("int8")

    # Lead targets. Features for row t describe months <= t; the lead-L target is
    # the label at t+L, so only lead >= 1 is a genuine forecast. Lead 0 is
    # detection and is kept purely as a reference ceiling.
    panel = panel.sort_values(["county_fips", "month"])
    g = panel.groupby("county_fips", sort=False)
    for lead in LEADS:
        for tgt in ("flood", "storm"):
            panel[f"y_{tgt}_lead{lead}"] = g[f"y_{tgt}"].shift(-lead)
            panel[f"y_{tgt}_onset_lead{lead}"] = g[f"y_{tgt}_onset"].shift(-lead)
    panel["y_compound_onset_lead1"] = (
        (panel["y_flood_onset_lead1"] == 1) & (panel["y_storm_onset_lead1"] == 1)
    ).astype("int8")

    panel["month"] = panel["month"].astype(str)
    panel.to_parquet(OUT_DIR / "panel.parquet", index=False)

    print(f"\npanel: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"written to {OUT_DIR / 'panel.parquet'}")
    print("\npositive rates (active / onset):")
    for lead in LEADS:
        print(
            f"  lead {lead}:  flood {panel[f'y_flood_lead{lead}'].mean()*100:5.2f}%"
            f" / {panel[f'y_flood_onset_lead{lead}'].mean()*100:5.2f}%"
            f"    storm {panel[f'y_storm_lead{lead}'].mean()*100:5.2f}%"
            f" / {panel[f'y_storm_onset_lead{lead}'].mean()*100:5.2f}%"
        )
    print(f"  compound onset lead1: {panel['y_compound_onset_lead1'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
