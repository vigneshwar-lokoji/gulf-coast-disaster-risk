"""Hazard labels from observational sources, not administrative ones.

FEMA disaster declarations were the original label and they are the wrong tool
for a hazard model: a declaration needs a damage threshold met and a governor's
request, so most floods never get one. Across 2000-2026 the Gulf Coast panel had
only 82 distinct flood declarations, which made the flood metrics an estimate
from 16 test events.

Replacements, each emitting the same (county_fips, begin, end) interval shape so
build_daily.py can treat them identically:

  noaa_flood_events  NOAA Storm Events, county-coded Flood / Flash Flood.
                     ~6,700 independent episodes over the same period.
  hurdat_events      NHC HURDAT2 best track. A county-day counts as exposed when
                     a cyclone centre passes close enough to put the county
                     inside the 34-kt wind field.
  fema_events        Kept, but demoted to a secondary 'this one was costly'
                     target rather than the hazard label.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

FLOOD_TYPES = {"Flood", "Flash Flood", "Coastal Flood", "Lakeshore Flood"}

# Tropical-storm force. Below this the wind field does little damage and the
# label would be dominated by distant brushes.
TS_WIND_KT = 34
NM_TO_KM = 1.852

EARTH_KM = 6371.0


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, degrees clockwise from north."""
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    y = np.sin(dlon) * np.cos(lat2 * p)
    x = (np.cos(lat1 * p) * np.sin(lat2 * p)
         - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _runs_to_intervals(df, keys):
    """Collapse exposed days into begin/end intervals within each key group.

    Consecutive days belong to one event; a gap of a day or more starts a new
    one, which matters for slow or looping storms that leave and return.
    Grouping includes the storm, so two cyclones hitting the same county in one
    week stay two events rather than merging into one long interval.
    """
    out = []
    for k, g in df.groupby(keys, sort=False):
        days = np.sort(g["date"].unique())
        if not len(days):
            continue
        brk = np.flatnonzero(np.diff(days.astype("datetime64[D]").astype(int)) > 1)
        for s, e in zip(np.r_[0, brk + 1], np.r_[brk, len(days) - 1]):
            out.append((*(k if isinstance(k, tuple) else (k,)),
                        pd.Timestamp(days[s]), pd.Timestamp(days[e])))
    return pd.DataFrame(out, columns=[*keys, "begin", "end"])


# -----------------------------------------------------------------------------
# NOAA Storm Events -- flood
# -----------------------------------------------------------------------------
def noaa_flood_events(storm_events_dir, keep_fips, start_year, end_year):
    """County-coded flood and flash-flood events.

    Only CZ_TYPE == 'C' rows are usable. Flood and flash flood are recorded
    against counties, which is why this works here and does not work for
    tropical cyclones -- those are logged against forecast zones instead.
    """
    cols = ["BEGIN_YEARMONTH", "BEGIN_DAY", "END_YEARMONTH", "END_DAY",
            "EVENT_TYPE", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "EPISODE_ID"]
    frames = []
    for f in sorted(glob.glob(str(Path(storm_events_dir) / "*details*.csv.gz"))):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        d = d[(d["CZ_TYPE"] == "C") & d["EVENT_TYPE"].isin(FLOOD_TYPES)]
        d = d.dropna(subset=["STATE_FIPS", "CZ_FIPS"])
        if d.empty:
            continue
        d["county_fips"] = (d["STATE_FIPS"].astype(int) * 1000
                            + d["CZ_FIPS"].astype(int)).astype(str).str.zfill(5)
        frames.append(d[d["county_fips"].isin(keep_fips)])

    e = pd.concat(frames, ignore_index=True)
    e["begin"] = pd.to_datetime(
        e["BEGIN_YEARMONTH"].astype(int).astype(str)
        + e["BEGIN_DAY"].astype(int).astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce")
    e["end"] = pd.to_datetime(
        e["END_YEARMONTH"].astype(int).astype(str)
        + e["END_DAY"].astype(int).astype(str).str.zfill(2),
        format="%Y%m%d", errors="coerce")
    e["end"] = e["end"].fillna(e["begin"])
    e = e.dropna(subset=["begin"])
    e = e[(e["begin"].dt.year >= start_year) & (e["begin"].dt.year <= end_year)]
    # A handful of records have end before begin; trust the begin date.
    e.loc[e["end"] < e["begin"], "end"] = e["begin"]
    return e[["county_fips", "begin", "end", "EPISODE_ID"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# HURDAT2 -- hurricane / tropical storm exposure
# -----------------------------------------------------------------------------
def parse_hurdat2(path):
    """HURDAT2 is header lines followed by that storm's fixed-width positions."""
    rows = []
    sid = name = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if re.fullmatch(r"[A-Z]{2}\d{6}", parts[0]):
            sid, name = parts[0], parts[1]
            continue
        # date, time, record_id, status, lat, lon, wind_kt, pressure,
        # then 34/50/64-kt radii as NE,SE,SW,NW each, then radius of max wind.
        lat = float(parts[4][:-1]) * (1 if parts[4][-1] == "N" else -1)
        lon = float(parts[5][:-1]) * (1 if parts[5][-1] == "E" else -1)
        r34 = [float(parts[8 + i]) for i in range(4)]
        rows.append({
            "storm_id": sid, "name": name,
            "ts": pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M"),
            "status": parts[3], "lat": lat, "lon": lon,
            "wind_kt": float(parts[6]),
            # -999 is 'not recorded'; 0 before 2004 means the same thing, since
            # the radii were simply not archived then.
            **{f"r34_{q}": (v if v > 0 else np.nan)
               for q, v in zip(("ne", "se", "sw", "nw"), r34)},
        })
    return pd.DataFrame(rows)


def _fallback_radius(track):
    """Median observed 34-kt radius per intensity bin, measured from the data.

    The radii are only archived from 2004 onward, so earlier storms need a
    stand-in. Taking it from the later record keeps the footprint empirical
    instead of a round number chosen to make the labels come out well.
    """
    bins = [34, 50, 64, 83, 96, 113, 137, 999]
    t = track.copy()
    t["bin"] = pd.cut(t["wind_kt"], bins=bins, right=False)
    out = {}
    for b, g in t.groupby("bin", observed=True):
        vals = g[[f"r34_{q}" for q in ("ne", "se", "sw", "nw")]].to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 20:
            out[b] = float(np.median(vals))
    return out, bins


def hurdat_events(hurdat_path, counties, start_year, end_year,
                  wind_kt=TS_WIND_KT):
    """County-days where the 34-kt wind field of a cyclone covered the county.

    `counties` is a frame of county_fips / lat / lon. The track is interpolated
    to hourly before testing, because a storm moving at 20 kt covers ~130 km
    between archived 6-hourly fixes and would otherwise skip counties.
    """
    track = parse_hurdat2(hurdat_path)
    track = track[(track["ts"].dt.year >= start_year)
                  & (track["ts"].dt.year <= end_year)]

    fallback, bins = _fallback_radius(track)

    # Only positions at or above the wind threshold can produce exposure, but
    # interpolation needs the neighbouring fixes too, so filter per storm after
    # resampling rather than before.
    keep_cols = ["ts", "lat", "lon", "wind_kt",
                 "r34_ne", "r34_se", "r34_sw", "r34_nw"]
    pieces = []
    for sid, g in track.groupby("storm_id", sort=False):
        g = g.sort_values("ts").set_index("ts")[keep_cols[1:]]
        if len(g) < 2:
            continue
        g = g.resample("1h").interpolate("linear", limit_direction="both")
        g["storm_id"] = sid
        pieces.append(g.reset_index())
    pos = pd.concat(pieces, ignore_index=True)
    pos = pos[pos["wind_kt"] >= wind_kt]

    # Trim to a box around the counties plus the widest plausible wind field,
    # which removes the entire mid-Atlantic and most of the Caribbean.
    pad = 6.0
    pos = pos[(pos["lat"].between(counties["lat"].min() - pad,
                                  counties["lat"].max() + pad))
              & (pos["lon"].between(counties["lon"].min() - pad,
                                    counties["lon"].max() + pad))]
    if pos.empty:
        return pd.DataFrame(columns=["county_fips", "begin", "end"])

    # Resolve each position's quadrant radii, filling gaps from the bin medians.
    bin_idx = np.digitize(pos["wind_kt"].to_numpy(), bins) - 1
    fb = np.array([fallback.get(pd.Interval(bins[i], bins[i + 1], closed="left"),
                                np.nan) for i in bin_idx])
    R = {}
    for q in ("ne", "se", "sw", "nw"):
        r = pos[f"r34_{q}"].to_numpy()
        R[q] = np.where(np.isnan(r), fb, r) * NM_TO_KM

    plat = pos["lat"].to_numpy()
    plon = pos["lon"].to_numpy()
    pday = pos["ts"].dt.floor("D").to_numpy()
    pwind = pos["wind_kt"].to_numpy()
    pstorm = pos["storm_id"].to_numpy()

    hits = []
    for fips, clat, clon in counties[["county_fips", "lat", "lon"]].itertuples(index=False):
        dist = _haversine(clat, clon, plat, plon)
        brg = _bearing(plat, plon, clat, clon)   # county as seen from the storm
        quad = np.select(
            [brg < 90, brg < 180, brg < 270],
            [R["ne"], R["se"], R["sw"]], default=R["nw"])
        m = np.isfinite(quad) & (dist <= quad)
        if m.any():
            hits.append(pd.DataFrame({"county_fips": fips, "storm_id": pstorm[m],
                                      "date": pday[m], "wind_kt": pwind[m]}))
    if not hits:
        return pd.DataFrame(columns=["county_fips", "storm_id", "begin", "end"])

    h = pd.concat(hits, ignore_index=True)
    daily = h.groupby(["county_fips", "storm_id", "date"],
                      as_index=False)["wind_kt"].max()
    ev = _runs_to_intervals(daily, ["county_fips", "storm_id"])
    peak = (daily.groupby(["county_fips", "storm_id"])["wind_kt"].max()
            .rename("peak_wind_kt").reset_index())
    return ev.merge(peak, on=["county_fips", "storm_id"], how="left")
