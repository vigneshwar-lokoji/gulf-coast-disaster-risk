"""Stage 4b: run the exported model on dates beyond the training data.

The replay in the dashboard stops at 2026-08-19 because that is where the stored
GHCN panel ends. This script goes past it by assembling the same nine features
from two live public sources:

    observed rainfall   NCEI GHCN-Daily access service  (~2 day latency)
                        -> the exact station network the model was trained on
    forecast rainfall   NWS api.weather.gov gridpoint QPF  (~7 days ahead)

A county's daily rainfall is the mean over its reporting stations, and
county-days with no reporting station are zero -- both exactly as build_panel.py
and stage2.py did it, so the live features are built the same way as the
training features.

Feature provenance, which is the thing that actually matters here:

    fwd3, max1, ratio   forecast only, by construction (they look forward)
    prev7, prev30       observed, with the last ~2 days before the target taken
                        from forecast because GHCN has not published them yet
    extents, month      derived

Every printed row carries the share of its antecedent window that was observed
rather than forecast, so a degraded row is visible rather than implied.

Dates run from any past day (fully observed) up to the NWS horizon, about seven
days out. Past the horizon the 72h window is incomplete, so those days are
dropped rather than reported as quiet -- "not predictable" and "no county
crossed its threshold" are different claims.

    python stage4_live.py                 rolling window, today-30 .. horizon
    python stage4_live.py 2026-09-01 2026-09-30     an explicit past range
    python stage4_live.py --verify                  check the live fetch against
                                                    the stored training panel
"""

import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from math import cos, exp, log1p, pi, sin
from pathlib import Path

import pandas as pd

from config import DATA_DIR, FILES

HERE = Path(__file__).parent
MODEL = json.loads((HERE / "docs" / "data" / "model.json").read_text())
COUNTIES = MODEL["counties"]
PTS_CACHE = HERE / "artifacts" / "stage4" / "nws_points.json"
MAPPING = HERE / "refdata" / "station_county_mapping.csv"
# NWS asks for a contact string so they can reach you during a security event.
UA = {"User-Agent": "(gulf-coast-disaster-risk research, "
                    "github.com/vigneshwar-lokoji/gulf-coast-disaster-risk)"}
NCEI = "https://www.ncei.noaa.gov/access/services/data/v1"

# GHCN days are local calendar days; the Gulf is mostly US Central. One hour of
# day-boundary error on the Florida peninsula is well below the other errors.
LOCAL_OFFSET = timedelta(hours=-6)
OP = next(o for o in MODEL["operating_points"] if o["threshold"] == 0.10)


def get(url, timeout=90):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


# ----------------------------------------------------- observed (NCEI GHCN)

def station_map():
    # The refresh job runs on a clean checkout with no access to the private
    # data directory, so the station list is bundled in the repo.
    src = MAPPING if MAPPING.exists() else DATA_DIR / FILES["mapping"]
    m = pd.read_csv(src)
    m["county_fips"] = m["county_fips"].astype(str).str.zfill(5)
    return m[m["county_fips"].isin(COUNTIES)][["STATION", "county_fips"]]


def fetch_observed(stations, start, end):
    """Daily station rainfall in mm. GHCN publishes PRCP in tenths of a mm."""
    out = []

    def one(batch):
        u = (f"{NCEI}?dataset=daily-summaries&stations={','.join(batch)}"
             f"&dataTypes=PRCP&startDate={start}&endDate={end}&format=json")
        try:
            body = get(u)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return []
        return json.loads(body) if body.strip() else []

    chunks = [stations[i:i + 50] for i in range(0, len(stations), 50)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for rows in pool.map(one, chunks):
            out.extend(rows)

    if not out:
        return pd.DataFrame(columns=["STATION", "DATE", "prcp"])
    df = pd.DataFrame(out)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["prcp"] = pd.to_numeric(df["PRCP"], errors="coerce") / 10.0
    return df.dropna(subset=["prcp"])[["STATION", "DATE", "prcp"]]


def county_observed(start, end):
    m = station_map()
    obs = fetch_observed(sorted(m["STATION"].unique()), start, end)
    if obs.empty:
        return {}, 0
    obs = obs.merge(m, on="STATION", how="inner")
    g = obs.groupby(["county_fips", "DATE"])["prcp"].mean()
    out = {}
    for (fips, d), v in g.items():
        out.setdefault(fips, {})[d.date()] = float(v)
    return out, obs["DATE"].max().date()


# -------------------------------------------------------- forecast (NWS)

def duration_hours(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", iso)
    d, h, mi = (int(x) if x else 0 for x in m.groups())
    return d * 24 + h + mi / 60


def county_forecast(points):
    """Local-day rainfall totals in mm per county, from NWS gridpoint QPF."""
    def one(item):
        fips, c = item
        try:
            if fips not in points:
                p = json.loads(get(f"https://api.weather.gov/points/{c['lat']},{c['lon']}", 30))
                points[fips] = p["properties"]["forecastGridData"]
            g = json.loads(get(points[fips], 30))
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, OSError):
            return fips, None
        days = {}
        for v in g["properties"]["quantitativePrecipitation"]["values"]:
            iso, dur = v["validTime"].split("/")
            mm = v["value"]
            if mm is None:
                continue
            start = datetime.fromisoformat(iso).astimezone(timezone.utc)
            hours = duration_hours(dur)
            t = 0.0
            while t < hours:
                chunk = min(1.0, hours - t)
                day = (start + timedelta(hours=t) + LOCAL_OFFSET).date()
                days[day] = days.get(day, 0.0) + mm * chunk / hours
                t += chunk
        return fips, days

    out = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fips, days in pool.map(one, COUNTIES.items()):
            if days is not None:
                out[fips] = days
    return out


# --------------------------------------------------------------- the model

def predict(feats):
    z = MODEL["intercept"]
    for name, co, mu, sd in zip(MODEL["features"], MODEL["coef"],
                                MODEL["feature_mean"], MODEL["feature_scale"]):
        x = feats.get(name)
        if x is None:
            x = MODEL["feature_median_fallback"][name]
        z += co * (x - mu) / sd
    s = 1.0 / (1.0 + exp(-z))
    return 1.0 / (1.0 + exp(-(MODEL["platt_coef"] * s + MODEL["platt_intercept"])))


# ------------------------------------------------------------- verify mode

def verify():
    """Does the live NCEI fetch reproduce the stored training panel?

    If it does not, nothing downstream can be trusted, because the live features
    would be built from a different quantity than the one the model learned on.
    """
    w = pd.read_parquet(DATA_DIR / FILES["weather"], columns=["STATION", "DATE", "PRCP"])
    end = w["DATE"].max().date()
    start = end - timedelta(days=27)
    print(f"comparing live NCEI fetch against the stored panel, {start} .. {end}")

    m = station_map()
    stored = (w[(w.DATE.dt.date >= start) & (w.DATE.dt.date <= end)][["STATION", "DATE", "PRCP"]]
              .merge(m, on="STATION", how="inner"))
    live = fetch_observed(sorted(m["STATION"].unique()),
                          start.isoformat(), end.isoformat()).merge(m, on="STATION")

    # Two different things can go wrong and they have different remedies, so
    # they are measured separately rather than blended into one number.
    j = stored.merge(live, on=["STATION", "DATE"], how="outer", indicator=True)
    both = j[j._merge == "both"]
    diff = (both["PRCP"] - both["prcp"]).abs()
    agree = (diff < 0.05).mean()

    print(f"\n  1. same station, same day -- is the value identical?")
    print(f"     {len(both):,} station-days compared")
    print(f"     identical to 0.05 mm: {agree:.2%}   mean abs diff {diff.mean():.4f} mm")
    revised = both[diff >= 0.05]
    print(f"     {len(revised)} revised, on {revised.STATION.nunique()} stations "
          f"({revised.STATION.str.startswith('USC').mean():.0%} manual COOP gauges)")

    print(f"\n  2. coverage -- does NCEI now hold days the stored panel lacks?")
    print(f"     only in NCEI (backfilled since the snapshot): "
          f"{(j._merge == 'right_only').sum():,}")
    print(f"     only in the stored panel: {(j._merge == 'left_only').sum():,}")
    print(f"     stations: stored {stored.STATION.nunique()}, live {live.STATION.nunique()}")

    print()
    if agree > 0.98:
        print("  PASS on values: the live path reads the same quantity the model "
              "was trained on.")
        print("  Coverage differs because GHCN backfills and quality-controls "
              "recent data for weeks after")
        print("  publication. Live county means are therefore taken over a "
              "slightly more complete station")
        print("  set than training used. That is a small benign difference, not "
              "a defect -- but it is the")
        print("  reason live output will never be bit-identical to a replay row.")
    else:
        print("  FAIL: values disagree on shared station-days. Do not trust live "
              "predictions until this is understood.")


# ---------------------------------------------------------------- predict

NEAR_KEEP = 8      # near-miss counties kept per day, ranked by ratio


def run(targets):
    first, last = min(targets), max(targets)
    obs_start = (first - timedelta(days=32)).isoformat()
    obs_end = date.today().isoformat()

    print(f"targets {first} .. {last}")
    print(f"fetching observed rainfall {obs_start} .. {obs_end} (NCEI GHCN-Daily) ...")
    obs, obs_last = county_observed(obs_start, obs_end)
    print(f"  {len(obs)} of {len(COUNTIES)} counties reported; "
          f"latest observation {obs_last}")

    points = json.loads(PTS_CACHE.read_text()) if PTS_CACHE.exists() else {}
    print(f"fetching NWS forecasts for {len(COUNTIES)} counties ...")
    fc = county_forecast(points)
    PTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PTS_CACHE.write_text(json.dumps(points))
    print(f"  {len(fc)} counties returned a forecast")

    fc_last = max((d for days in fc.values() for d in days), default=obs_last)
    print(f"  forecast reaches {fc_last}")

    def rain(fips, d):
        """Observed if GHCN has published it, otherwise the NWS forecast.
        Missing county-days are zero, matching build_panel.py."""
        if d <= obs_last:
            return obs.get(fips, {}).get(d, 0.0), True
        return fc.get(fips, {}).get(d, 0.0), False

    def covered(fips, d):
        """Is this county-day backed by either source? Observed days always are,
        because a non-reporting county-day is a real zero in training too."""
        return d <= obs_last or d in fc.get(fips, {})

    out, near_out, skipped = {}, {}, []
    for target in targets:
        window = [target + timedelta(days=i) for i in range(3)]
        # A day with an incomplete 72h window is not predictable at all, which
        # is a different statement from "no county crossed its threshold". Drop
        # it rather than emit an empty day the dashboard would render as quiet.
        if window[-1] > fc_last:
            skipped.append(target)
            continue
        rows, near = [], []
        for fips, c in COUNTIES.items():
            if not all(covered(fips, w) for w in window):
                continue
            vals = [rain(fips, w)[0] for w in window]
            fwd3 = sum(vals)
            if fwd3 < c["threshold_mm"]:
                near.append({"fips": fips, "fwd3": fwd3,
                             "ratio": fwd3 / c["threshold_mm"]})
                continue
            p7 = [rain(fips, target - timedelta(days=k)) for k in range(1, 8)]
            p30 = [rain(fips, target - timedelta(days=k)) for k in range(1, 31)]
            rows.append({
                "fips": fips, "fwd3": fwd3, "max1": max(vals),
                "thr": c["threshold_mm"],
                "prev7": sum(v for v, _ in p7), "prev30": sum(v for v, _ in p30),
                "obs_share": sum(o for _, o in p30) / 30,
                "fwd_obs": all(w <= obs_last for w in window),
            })

        gulf_extent = len(rows)
        per_state = {}
        for r in rows:
            per_state[r["fips"][:2]] = per_state.get(r["fips"][:2], 0) + 1
        for r in rows:
            r["p"] = predict({
                "ratio": r["fwd3"] / r["thr"],
                "log_fwd3": log1p(r["fwd3"]),
                "max1": r["max1"],
                "prev7": r["prev7"],
                "prev30": r["prev30"],
                "month_sin": sin(2 * pi * target.month / 12),
                "month_cos": cos(2 * pi * target.month / 12),
                "other_extent": per_state[r["fips"][:2]] - 1,
                "gulf_extent": gulf_extent,
            })
        rows.sort(key=lambda r: -r["p"])
        # No probability is attached to these. The model is conditional on
        # exceeding the threshold, so below it there is nothing to predict --
        # only a distance to report.
        near.sort(key=lambda r: -r["ratio"])
        out[target.isoformat()] = rows
        near_out[target.isoformat()] = near[:NEAR_KEEP]

        print(f"\n=== {target} ===  {len(rows)} counties over threshold")
        if not rows:
            print("  model is silent: no county reaches its p99. Closest:")
            for r in near[:3]:
                c = COUNTIES[r["fips"]]
                print(f"     {c['name']}, {c['state']}  {r['fwd3']:.1f} mm "
                      f"of {c['threshold_mm']:.1f} needed ({r['ratio']:.0%})")
            continue
        print(f"  {'':2} {'county':<17} {'st':<3} {'P':>6} {'72h mm':>7} "
              f"{'ratio':>6} {'prev30':>7} {'obs':>5}")
        for i, r in enumerate(rows, 1):
            c = COUNTIES[r["fips"]]
            print(f"  {i:>2} {c['name']:<17} {c['state']:<3} {r['p']:>5.1%} "
                  f"{r['fwd3']:>7.1f} {r['fwd3'] / r['thr']:>5.2f}x "
                  f"{r['prev30']:>6.1f} {r['obs_share']:>5.0%}"
                  + ("  FLAG" if r["p"] >= OP["threshold"] else ""))

    if skipped:
        print(f"\nskipped {len(skipped)} day(s) past the forecast horizon: "
              f"{skipped[0]} .. {skipped[-1]} (72h window extends beyond {fc_last})")

    days_done = sorted(out)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observed_through": obs_last.isoformat(),
        "forecast_through": fc_last.isoformat(),
        "first_day": days_done[0], "last_day": days_done[-1],
        "n_counties": len(COUNTIES),
        "threshold": OP["threshold"],
        "precision": OP["precision"], "recall": OP["recall"],
        "days": {k: [[r["fips"], round(r["p"], 4), round(r["fwd3"], 1),
                      round(r["fwd3"] / r["thr"], 3), round(r["prev30"], 1),
                      round(r["obs_share"], 3), int(r["fwd_obs"])] for r in v]
                 for k, v in out.items()},
        "near": {k: [[r["fips"], round(r["fwd3"], 1), round(r["ratio"], 3)]
                     for r in v] for k, v in near_out.items()},
    }
    f = HERE / "docs" / "data" / "live.json"
    f.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {f} ({f.stat().st_size / 1024:.0f} KB, {len(out)} days)")

    flagged = [r for v in out.values() for r in v if r["p"] >= OP["threshold"]]
    print(f"{len(flagged)} county-days clear the {OP['threshold']:.0%} operating "
          f"point ({OP['precision']:.0%} precision, {OP['recall']:.0%} recall).")
    if any(r["fips"][:2] == "48" for v in out.values() for r in v):
        print(f"Texas rows present; held-out Texas lift is "
              f"{MODEL['state_transfer_lift']['TX']:.2f}x. Treat as unvalidated.")


BACK_DEFAULT, FWD_DEFAULT = 30, 8


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--verify" in sys.argv:
        verify()
    else:
        if args:
            a = date.fromisoformat(args[0])
            b = date.fromisoformat(args[-1])
        else:
            # Rolling window: recent observed days stay selectable, and the
            # forward end is trimmed to the real forecast horizon during the run.
            today = date.today()
            a = today - timedelta(days=BACK_DEFAULT)
            b = today + timedelta(days=FWD_DEFAULT)
        run([a + timedelta(days=i) for i in range((b - a).days + 1)])
