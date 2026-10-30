"""Fetch live weather per county and score today + next 3 days.

Open-Meteo's forecast endpoint returns up to 92 past days alongside the
forecast, so a single request per county supplies both the antecedent windows
and the forward-looking rain the models need. No API key required.

Writes docs/data/live.json.
"""

import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from config import OUT_DIR
from proto_models import FLOOD_FEATS, STORM_FEATS, load_scorers

API = "https://api.open-meteo.com/v1/forecast"
PAST = 92
FWD = 4
STATIC = ["lat", "lon", "n_stations", "AREA_SQMI", "E_TOTPOP", "RPL_THEME1",
          "RPL_THEME4", "RPL_THEMES", "pop_density", "p95_prev3", "p95_fwd3",
          "mean_prev30"]


def county_static(panel):
    return (
        panel.sort_values("DATE")
        .groupby("county_fips")[STATIC]
        .last()
        .reset_index()
    )


def fetch(row):
    try:
        r = requests.get(
            API,
            params={
                "latitude": round(row.lat, 4),
                "longitude": round(row.lon, 4),
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                         "wind_speed_10m_max,wind_gusts_10m_max",
                "timezone": "America/Chicago",
                "past_days": PAST,
                "forecast_days": FWD,
            },
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        return row.county_fips, d
    except Exception as e:
        return row.county_fips, None


def build_rows(fips, daily, static, feats):
    """One feature row per forecast day for this county."""
    t = pd.Series(pd.to_datetime(daily["time"]))
    prcp = pd.Series(daily["precipitation_sum"], dtype="float64").fillna(0.0)
    tmax = pd.Series(daily["temperature_2m_max"], dtype="float64")
    tmin = pd.Series(daily["temperature_2m_min"], dtype="float64")

    # Index of "today" = first forecast day.
    i0 = PAST
    out = []
    for k in range(FWD):
        i = i0 + k
        if i >= len(t):
            break
        past = prcp.iloc[:i + 1]
        fwd = prcp.iloc[i + 1:i + 4]

        f = {
            "prcp": float(prcp.iloc[i]),
            "prcp_max_stn": float(prcp.iloc[i]),
            "tmax": float(tmax.iloc[i]) if pd.notna(tmax.iloc[i]) else np.nan,
            "tmin": float(tmin.iloc[i]) if pd.notna(tmin.iloc[i]) else np.nan,
        }
        for w in (3, 7, 14, 30, 60, 90):
            f[f"prcp_prev{w}"] = float(past.iloc[-w:].sum())
        f["prcp_fwd1"] = float(prcp.iloc[i + 1]) if i + 1 < len(prcp) else 0.0
        f["prcp_fwd3"] = float(fwd.sum())
        f["prcp_fwd3_max"] = float(fwd.max()) if len(fwd) else 0.0

        day = t.iloc[i]
        f["month_sin"] = float(np.sin(2 * np.pi * day.month / 12))
        f["month_cos"] = float(np.cos(2 * np.pi * day.month / 12))
        f["hurricane_season"] = int(6 <= day.month <= 11)

        for c in STATIC:
            f[c] = float(static[c]) if pd.notna(static[c]) else np.nan

        p95_3 = f["p95_prev3"] or np.nan
        p95_f = f["p95_fwd3"] or np.nan
        m30 = f["mean_prev30"] or np.nan
        f["ratio_prev3"] = f["prcp_prev3"] / p95_3 if p95_3 else np.nan
        f["ratio_fwd3"] = f["prcp_fwd3"] / p95_f if p95_f else np.nan
        f["ratio_prev30"] = f["prcp_prev30"] / m30 if m30 else np.nan
        f["log_fwd3"] = float(np.log1p(max(f["prcp_fwd3"], 0)))

        out.append((day.strftime("%Y-%m-%d"), f,
                    {"precip": round(float(prcp.iloc[i]), 1),
                     "tmax": None if pd.isna(tmax.iloc[i]) else round(float(tmax.iloc[i]), 1),
                     "tmin": None if pd.isna(tmin.iloc[i]) else round(float(tmin.iloc[i]), 1),
                     "wind": round(float(daily["wind_speed_10m_max"][i] or 0), 1),
                     "gust": round(float(daily["wind_gusts_10m_max"][i] or 0), 1),
                     "fwd3": round(f["prcp_fwd3"], 1)}))
    return out


def main():
    panel = pd.read_parquet(
        OUT_DIR / "daily_panel.parquet",
        columns=["county_fips", "DATE"] + STATIC,
    )
    feats = sorted(set(FLOOD_FEATS) | set(STORM_FEATS))
    scorers = load_scorers()
    cs = county_static(panel)
    print(f"fetching live weather for {len(cs)} counties ...", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch, list(cs.itertuples(index=False))))

    ok = {f: d for f, d in results if d}
    print(f"  {len(ok)}/{len(cs)} succeeded")

    static_map = cs.set_index("county_fips")
    rows, keys = [], []
    weather = {}
    for fips, daily in ok.items():
        for day, f, w in build_rows(fips, daily, static_map.loc[fips], feats):
            rows.append(f)
            keys.append((fips, day))
            weather.setdefault(fips, {})[day] = w

    X = pd.DataFrame(rows).reindex(columns=feats)
    for haz, fn in scorers.items():
        X[f"p_{haz}"] = fn(X)

    days = {}
    for (fips, day), pf, ps in zip(keys, X["p_flood"], X["p_storm"]):
        days.setdefault(day, []).append(
            [fips, round(float(pf), 5), round(float(ps), 5)]
        )

    payload = {
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Open-Meteo forecast, scored by the prototype XGBoost models",
        "n_counties": len(ok),
        "days": days,
        "weather": weather,
    }
    p = pathlib.Path("docs/data/live.json")
    p.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)  days={sorted(days)}")


if __name__ == "__main__":
    main()
