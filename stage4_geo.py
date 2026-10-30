"""Stage 4c: build the dashboard's map geometry.

Turns public county/state GeoJSON into pre-projected SVG path strings so the
browser draws real boundaries instead of dots, and needs no mapping library.

Scope is the lower 48. The models only cover the Gulf, but the map covers the
country so the validated region is visible as a region rather than as the whole
world — and so weather and official warnings, which work anywhere, can be
queried for any county the user clicks.

Projection is Albers equal-area conic on the standard CONUS parallels, which
keeps county areas comparable by eye. Everything is projected here, in Python,
so the payload is plain path strings and the dashboard stays dependency-free.
"""

import json
import urllib.request
from math import cos, radians, sin, sqrt
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
OFF_MAP = {"02", "15", "72", "78", "60", "66", "69"}  # AK, HI, territories
W, H, PAD = 960, 560, 8
TOL = 0.30          # px; points closer than this to the previous one are dropped
MIN_RING_AREA = 0.6  # px^2; drops slivers, keeps real islands

ABBR = {
    "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
    "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR",
    "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX",
    "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

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

LON0, PHI0, PHI1, PHI2 = radians(-96.0), radians(37.5), radians(29.5), radians(45.5)
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

conus = [f for f in counties["features"] if f["id"][:2] not in OFF_MAP]
print(f"{len(conus)} counties in the lower 48")

pts = [albers(x, y) for f in conus for r in rings(f["geometry"]) for x, y in r]
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
SCALE = min((W - 2 * PAD) / (x1 - x0), (H - 2 * PAD) / (y1 - y0))
OX = (W - (x1 - x0) * SCALE) / 2 - x0 * SCALE
OY = (H - (y1 - y0) * SCALE) / 2 - y0 * SCALE


def to_px(lon, lat):
    x, y = albers(lon, lat)
    return x * SCALE + OX, y * SCALE + OY


def simplify(ring):
    """Projected, point-thinned ring plus its signed pixel area."""
    kept = []
    for lon, lat in ring:
        p = to_px(lon, lat)
        if not kept or abs(p[0] - kept[-1][0]) + abs(p[1] - kept[-1][1]) > TOL:
            kept.append(p)
    if len(kept) < 4:
        return None, 0.0
    a = sum(kept[i][0] * kept[(i + 1) % len(kept)][1]
            - kept[(i + 1) % len(kept)][0] * kept[i][1] for i in range(len(kept)))
    return kept, abs(a) / 2


def draw(kept):
    return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in kept) + "Z"


def path(geom, keep_largest=False):
    """Simplified SVG path.

    At national scale a small county can fall under the sliver threshold and
    vanish. keep_largest guarantees every county still gets a clickable shape,
    so the area filter only ever removes extra rings, never a whole county.
    """
    parts, best = [], (None, 0.0)
    for ring in rings(geom):
        kept, area = simplify(ring)
        if kept is None:
            continue
        if area > best[1]:
            best = (kept, area)
        if area >= MIN_RING_AREA:
            parts.append(draw(kept))
    if not parts and keep_largest and best[0]:
        parts.append(draw(best[0]))
    return "".join(parts) or None


def centroid(geom):
    """Area-weighted centroid of the largest ring, in lon/lat.

    Used as the county's query point for weather and warnings, so it has to sit
    inside the county -- a bounding-box centre would not for coastal shapes.
    """
    best, best_a = None, -1.0
    for ring in rings(geom):
        a = sum(ring[i][0] * ring[(i + 1) % len(ring)][1]
                - ring[(i + 1) % len(ring)][0] * ring[i][1]
                for i in range(len(ring))) / 2
        if abs(a) > best_a:
            best, best_a = ring, abs(a)
    if best is None:
        return None
    cx = cy = 0.0
    for i in range(len(best)):
        x0_, y0_ = best[i]
        x1_, y1_ = best[(i + 1) % len(best)]
        cr = x0_ * y1_ - x1_ * y0_
        cx += (x0_ + x1_) * cr
        cy += (y0_ + y1_) * cr
    a = sum(best[i][0] * best[(i + 1) % len(best)][1]
            - best[(i + 1) % len(best)][0] * best[i][1]
            for i in range(len(best))) / 2
    if abs(a) < 1e-12:  # degenerate ring; vertex mean is close enough
        return (sum(p[0] for p in best) / len(best),
                sum(p[1] for p in best) / len(best))
    return cx / (6 * a), cy / (6 * a)


county_paths, names = {}, {}
for f in conus:
    d = path(f["geometry"], keep_largest=True)
    if not d:
        continue
    county_paths[f["id"]] = d
    c = centroid(f["geometry"])
    names[f["id"]] = [
        f["properties"]["NAME"], ABBR.get(f["id"][:2], f["id"][:2]),
        round(c[1], 4), round(c[0], 4),
    ]

# State outlines on top of the counties, so states read as states.
skip = {"Alaska", "Hawaii", "Puerto Rico"}
state_paths, gulf_state_paths = [], []
gulf_names = {"Alabama", "Florida", "Louisiana", "Mississippi", "Texas"}
for f in states["features"]:
    nm = f["properties"]["name"]
    if nm in skip:
        continue
    d = path(f["geometry"])
    if not d:
        continue
    (gulf_state_paths if nm in gulf_names else state_paths).append(d)

payload = {
    "width": W, "height": H,
    "counties": county_paths,
    "names": names,
    "states": state_paths,
    "gulf_states": gulf_state_paths,
}
OUT.write_text(json.dumps(payload, separators=(",", ":")))

print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
print(f"  {len(county_paths)} county shapes, "
      f"{len(state_paths) + len(gulf_state_paths)} state outlines")
print(f"  counties in the modelled Gulf states: "
      f"{sum(1 for k in county_paths if k[:2] in GULF)}")
