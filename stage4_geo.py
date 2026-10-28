"""Stage 4c: build the dashboard's map geometry.

Turns public county/state GeoJSON into pre-projected SVG path strings so the
browser draws real boundaries instead of dots, and needs no mapping library.

Projection is Albers equal-area conic with parallels tuned to the Gulf, which
is the standard choice for the United States and keeps county areas comparable
by eye. Everything is projected here, in Python, so the payload is plain path
strings and the dashboard stays dependency-free.
"""

import json
import urllib.request
from math import asin, cos, pi, radians, sin, sqrt
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "artifacts" / "stage4" / "geo_src"
OUT = HERE / "docs" / "data" / "geo.json"
CACHE.mkdir(parents=True, exist_ok=True)

SRC = {
    "counties": "https://raw.githubusercontent.com/plotly/datasets/master/"
                "geojson-counties-fips.json",
    "states": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/"
              "data/geojson/us-states.json",
}
GULF = {"01", "12", "22", "28", "48"}
W, H, PAD = 960, 560, 10
TOL = 0.35          # px; points closer than this to the previous one are dropped
MIN_RING_AREA = 1.2  # px^2; drops slivers and specks, keeps real islands

# ------------------------------------------------------------------ source

def load(name):
    f = CACHE / f"{name}.json"
    if not f.exists():
        print(f"downloading {name} ...")
        req = urllib.request.Request(SRC[name], headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r:
            f.write_bytes(r.read())
    return json.loads(f.read_text())


# -------------------------------------------------------------- projection

LON0, PHI0, PHI1, PHI2 = radians(-89.5), radians(30.0), radians(26.0), radians(36.0)
N = (sin(PHI1) + sin(PHI2)) / 2
C = cos(PHI1) ** 2 + 2 * N * sin(PHI1)
RHO0 = sqrt(C - 2 * N * sin(PHI0)) / N


def albers(lon, lat):
    """Albers x/y, with y already negated so it grows southward like SVG."""
    phi, lam = radians(lat), radians(lon)
    rho = sqrt(max(0.0, C - 2 * N * sin(phi))) / N
    theta = N * (lam - LON0)
    return rho * sin(theta), rho * cos(theta) - RHO0


def rings(geom):
    """Every linear ring in a Polygon or MultiPolygon, outer rings only."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


# ------------------------------------------------------------------- fit

counties = load("counties")
states = load("states")

gulf_feats = [f for f in counties["features"] if f["id"][:2] in GULF]
print(f"{len(gulf_feats)} counties in the five Gulf states")

pts = [albers(x, y) for f in gulf_feats for r in rings(f["geometry"]) for x, y in r]
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
SCALE = min((W - 2 * PAD) / (x1 - x0), (H - 2 * PAD) / (y1 - y0))
OX = (W - (x1 - x0) * SCALE) / 2 - x0 * SCALE
OY = (H - (y1 - y0) * SCALE) / 2 - y0 * SCALE


def to_px(lon, lat):
    x, y = albers(lon, lat)
    return x * SCALE + OX, y * SCALE + OY


def path(geom):
    """Simplified SVG path. Returns None when nothing survives simplification."""
    out = []
    for ring in rings(geom):
        kept, area = [], 0.0
        for lon, lat in ring:
            p = to_px(lon, lat)
            if not kept or abs(p[0] - kept[-1][0]) + abs(p[1] - kept[-1][1]) > TOL:
                kept.append(p)
        if len(kept) < 4:
            continue
        for i in range(len(kept)):
            a, b = kept[i], kept[(i + 1) % len(kept)]
            area += a[0] * b[1] - b[0] * a[1]
        if abs(area) / 2 < MIN_RING_AREA:
            continue
        out.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in kept) + "Z")
    return "".join(out) or None


county_paths = {}
for f in gulf_feats:
    d = path(f["geometry"])
    if d:
        county_paths[f["id"]] = d

# Lower-48 state outlines as a faint backdrop, so the Gulf reads as part of the
# country rather than floating. Alaska and Hawaii would wreck the viewBox.
state_paths = []
for f in states["features"]:
    if f["properties"]["name"] in ("Alaska", "Hawaii", "Puerto Rico"):
        continue
    d = path(f["geometry"])
    if d:
        state_paths.append(d)

# Gulf state outlines drawn on top, emphasised.
gulf_names = {"Alabama", "Florida", "Louisiana", "Mississippi", "Texas"}
gulf_state_paths = [
    path(f["geometry"]) for f in states["features"]
    if f["properties"]["name"] in gulf_names and path(f["geometry"])
]

payload = {
    "width": W, "height": H,
    "counties": county_paths,
    "states": state_paths,
    "gulf_states": gulf_state_paths,
}
OUT.write_text(json.dumps(payload, separators=(",", ":")))

model = json.loads((HERE / "docs" / "data" / "model.json").read_text())
missing = [f for f in model["counties"] if f not in county_paths]
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
print(f"  {len(county_paths)} county shapes, {len(state_paths)} state outlines")
print(f"  modelled counties without a shape: {len(missing)}"
      + (f"  {missing[:8]}" if missing else ""))
