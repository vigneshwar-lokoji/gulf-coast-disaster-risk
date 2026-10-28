"""Stage 2 feasibility count, run before any Stage 2 modelling.

Stage 1 predicts the physical hazard (3-day precipitation above the county's
training-period p99). Stage 2 asks the conditional question: given that the
hazard occurred, which counties were federally declared?

The original FEMA-label design died on sample size: ~126 flood disasters over
~115 distinct begin-dates against millions of county-days. Conditioning on
hazard is supposed to fix the *denominator*. It cannot fix the numerator. This
script measures what is actually left so the model class is chosen from the
count rather than from hope.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, FILES

START_YEAR = 2000
MIN_COVERAGE = 0.95
TRAIN_END = 2014
# FEMA declarations lag the rain. A county flooded on day t is declared days to
# weeks later, so the hazard is matched to any qualifying declaration whose
# incident *begins* within this window of the hazard date.
LAG_WINDOW_DAYS = 14

FLOOD_HINTS = ("flood", "rain", "storm", "hurricane", "tropical")
DROP_TYPES = {"Biological", "Human Cause", "Toxic Substances", "Fire",
              "Drought", "Freezing", "Snowstorm", "Winter Storm", "Earthquake"}


def fips(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def county_daily():
    w = pd.read_parquet(DATA_DIR / FILES["weather"], columns=["STATION", "DATE", "PRCP"])
    w = w[w["DATE"].dt.year >= START_YEAR]

    mapping = pd.read_csv(DATA_DIR / FILES["mapping"])
    mapping["county_fips"] = fips(mapping["county_fips"])
    w = w.merge(mapping[["STATION", "county_fips"]], on="STATION", how="inner")

    daily = (
        w.groupby(["county_fips", "DATE"])["PRCP"].mean()
        .rename("prcp").reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(daily["county_fips"].unique()),
         pd.date_range(daily["DATE"].min(), daily["DATE"].max(), freq="D")],
        names=["county_fips", "DATE"],
    )
    daily = daily.set_index(["county_fips", "DATE"]).reindex(grid).reset_index()

    cov = daily.groupby("county_fips")["prcp"].apply(lambda s: s.notna().mean())
    keep = cov[cov >= MIN_COVERAGE].index
    daily = daily[daily["county_fips"].isin(keep)].copy()
    daily["prcp"] = daily["prcp"].fillna(0.0)
    daily["year"] = daily["DATE"].dt.year

    g = daily.sort_values(["county_fips", "DATE"]).groupby("county_fips", sort=False)["prcp"]
    daily["fwd3"] = g.transform(lambda s: s.rolling(3).sum().shift(-2))
    return daily


def declarations():
    f = pd.read_csv(DATA_DIR / FILES["fema"], low_memory=False)
    f["county_fips"] = fips(f["county_fips"])
    f = f[~f["incidentType"].isin(DROP_TYPES)]

    title = f["declarationTitle"].fillna("").str.lower()
    itype = f["incidentType"].fillna("").str.lower()
    blob = itype + " " + title
    f = f[blob.str.contains("|".join(FLOOD_HINTS), regex=True)]

    f["begin"] = pd.to_datetime(f["incidentBeginDate"], errors="coerce", utc=True).dt.tz_localize(None)
    f = f.dropna(subset=["begin"])
    f["begin"] = f["begin"].dt.normalize()
    f = f[f["begin"].dt.year >= START_YEAR]

    # One disaster covers many counties on one date. Collapse to the unique
    # (county, onset) pairs that Stage 2 can actually treat as observations.
    return f[["county_fips", "begin", "disasterNumber", "incidentType"]].drop_duplicates()


def main():
    daily = county_daily()
    print(f"counties {daily.county_fips.nunique()}  county-days {len(daily):,}")

    thr = (
        daily[daily.year <= TRAIN_END]
        .groupby("county_fips")["fwd3"].quantile(0.99).rename("thr")
    )
    daily = daily.join(thr, on="county_fips").dropna(subset=["fwd3", "thr"])
    haz = daily[daily["fwd3"] >= daily["thr"]].copy()

    print(f"\nhazard events {len(haz):,} over {haz.DATE.nunique():,} distinct dates "
          f"({haz.county_fips.nunique()} counties)")

    dec = declarations()
    print(f"qualifying declarations: {len(dec):,} county-onset rows, "
          f"{dec.begin.nunique()} distinct onsets, "
          f"{dec.disasterNumber.nunique()} disasters")

    # Label each hazard event: any qualifying declaration for that county with
    # onset in [hazard_date, hazard_date + LAG_WINDOW_DAYS].
    haz = haz.sort_values("DATE")
    dec = dec.sort_values("begin")
    m = pd.merge_asof(
        haz[["county_fips", "DATE", "fwd3", "thr"]],
        dec[["county_fips", "begin", "disasterNumber"]].rename(columns={"begin": "DATE"}),
        on="DATE", by="county_fips",
        direction="forward",
        tolerance=pd.Timedelta(days=LAG_WINDOW_DAYS),
    )
    m["y"] = m["disasterNumber"].notna().astype(int)

    pos = m[m.y == 1]
    print(f"\n--- Stage 2 label, lag window {LAG_WINDOW_DAYS}d ---")
    print(f"rows {len(m):,}  positives {m.y.sum():,}  prevalence {m.y.mean():.4f}")
    print(f"positives span {pos.DATE.nunique()} distinct hazard dates, "
          f"{pos.disasterNumber.nunique()} disasters, {pos.county_fips.nunique()} counties")
    print(f"effective independent positive units (date-grouped): {pos.DATE.nunique()}")

    print("\nprevalence by lag window:")
    for lag in (3, 7, 14, 30):
        mm = pd.merge_asof(
            haz[["county_fips", "DATE"]],
            dec[["county_fips", "begin", "disasterNumber"]].rename(columns={"begin": "DATE"}),
            on="DATE", by="county_fips", direction="forward",
            tolerance=pd.Timedelta(days=lag),
        )
        yy = mm["disasterNumber"].notna().astype(int)
        print(f"  {lag:>3}d: positives {yy.sum():>5,}  prevalence {yy.mean():.4f}  "
              f"dates {mm.loc[yy == 1, 'DATE'].nunique():>4}")

    print("\nhazard severity, declared vs not (14d window):")
    m["ratio"] = m["fwd3"] / m["thr"]
    print(m.groupby("y")["ratio"].describe()[["count", "mean", "50%", "max"]])

    print("\npositives per split:")
    m["year"] = m["DATE"].dt.year
    split = pd.cut(m["year"], [1999, 2014, 2019, 2026],
                   labels=["train 2000-2014", "val 2015-2019", "test 2020-2026"])
    print(pd.crosstab(split, m["y"]))


if __name__ == "__main__":
    main()
