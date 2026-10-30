/* Gulf Coast flood & hurricane risk — prototype dashboard.
 *
 * Risk comes precomputed from docs/data (export_proto.py, fetch_live.py).
 * Weather and official warnings are fetched live in the browser from two
 * keyless CORS-enabled APIs, so the whole thing runs on static hosting.
 */

const SVGNS = "http://www.w3.org/2000/svg";
const RAMP = ["#1b5e3f", "#4caf5c", "#ffc147", "#ff8a3d", "#ff3b2f"];
const BANDS = ["Normal", "Slightly raised", "Elevated", "High", "Very high"];
const HAZ_LABEL = { flood: "Flood", storm: "Hurricane / tropical storm" };
// States the risk models and the FEMA event history actually cover.
const MODELLED_STATES = new Set(["AL", "FL", "LA", "MS", "TX"]);

const el = (id) => document.getElementById(id);
const pct = (x, d = 2) => (100 * x).toFixed(d) + "%";
const fmtInt = (n) => n.toLocaleString();

const addDays = (iso, n) => {
  const t = new Date(iso + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
};
const dayName = (iso) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString(undefined, {
    weekday: "short", month: "short", day: "numeric", timeZone: "UTC",
  });

const S = {
  mode: "live", haz: "flood", date: null, fips: null,
  geo: null, meta: null, events: null, hist: null, live: null,
  paths: new Map(), alertReq: 0, wxReq: 0,
};

// ------------------------------------------------------------------ data

/** Sorted list of dates the current mode can show. */
function dates() {
  return Object.keys(S.mode === "live" ? S.live.days : S.hist.days).sort();
}

function dayRows(d) {
  return (S.mode === "live" ? S.live.days : S.hist.days)[d] || [];
}

/* Live rows are [fips, p_flood, p_storm]; history rows carry the rainfall and
 * the ground-truth active flags too. One shape for the rest of the code. */
function rowObj(r) {
  return {
    fips: r[0], flood: r[1], storm: r[2],
    fwd3: r.length > 3 ? r[3] : null,
    ratio: r.length > 4 ? r[4] : null,
    floodActive: r.length > 5 ? !!r[5] : false,
    stormActive: r.length > 6 ? !!r[6] : false,
  };
}

/** Nearest date we actually hold, so the picker never shows a blank map. */
function snap(target) {
  const ds = dates();
  if (ds.includes(target)) return target;
  const t = Date.parse(target);
  return ds.reduce((a, b) =>
    Math.abs(Date.parse(b) - t) < Math.abs(Date.parse(a) - t) ? b : a);
}

const county = (f) => S.meta.counties[f];
const cname = (f) => {
  const c = county(f);
  return c ? `${c[0]}, ${c[1]}` : f;
};

/** 0-4 band index from a raw probability, against that hazard's quantiles. */
function level(p, haz) {
  const b = S.meta.bands[haz];
  if (p >= b.p999) return 4;
  if (p >= b.p99) return 3;
  if (p >= b.p90) return 2;
  if (p >= b.p50) return 1;
  return 0;
}

// ------------------------------------------------------------------- map

function svgEl(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function drawMap() {
  const svg = el("map");
  svg.setAttribute("viewBox", `0 0 ${S.geo.width} ${S.geo.height}`);
  svg.textContent = "";

  const land = svgEl("g", { class: "m-land" });
  for (const [fips, d] of Object.entries(S.geo.counties)) {
    const p = svgEl("path", { d, class: "cty" });
    p.dataset.fips = fips;
    const t = svgEl("title", {});
    t.textContent = cname(fips);
    p.appendChild(t);
    p.addEventListener("click", () => selectCounty(fips));
    S.paths.set(fips, { path: p, title: t });
    land.appendChild(p);
  }
  svg.appendChild(land);

  // State outlines over the counties, with the modelled states emphasised so
  // the validated region reads as a region rather than as a colour accident.
  for (const [key, cls] of [["states", "m-borders"], ["gulf_states", "m-gulf"]]) {
    if (!S.geo[key]) continue;
    const g = svgEl("g", { class: cls });
    for (const d of S.geo[key]) g.appendChild(svgEl("path", { d }));
    svg.appendChild(g);
  }

  el("legend").innerHTML =
    `<span class="lg-t">${HAZ_LABEL[S.haz]} risk</span>` +
    BANDS.map((b, i) =>
      `<span class="lg"><i style="background:${RAMP[i]}"></i>${b}</span>`).join("") +
    `<span class="lg"><i class="nd"></i>outside the modelled region</span>`;
}

function paintMap(byFips) {
  for (const [fips, { path, title }] of S.paths) {
    const r = byFips.get(fips);
    const modelled = county(fips) && county(fips)[4];
    path.classList.toggle("nodata", !r);
    path.classList.toggle("sel", fips === S.fips);
    if (!r) {
      path.style.fill = "";
      title.textContent = modelled
        ? `${cname(fips)} — no risk computed for this date`
        : `${cname(fips)} — outside the modelled region; click for live weather`
          + " and warnings";
      continue;
    }
    const lv = level(r[S.haz], S.haz);
    path.style.fill = RAMP[lv];
    const gt = S.haz === "flood" ? r.floodActive : r.stormActive;
    title.textContent =
      `${cname(fips)} — ${BANDS[lv]} (${pct(r[S.haz])})` +
      (gt ? " · federal disaster active on this date" : "");
  }
}

// ---------------------------------------------------------------- render

function render() {
  const ds = dates();
  const inp = el("d-date");
  inp.min = ds[0];
  inp.max = ds[ds.length - 1];
  inp.value = S.date;

  const rows = dayRows(S.date).map(rowObj);
  const byFips = new Map(rows.map((r) => [r.fips, r]));
  el("legend").firstChild.textContent = `${HAZ_LABEL[S.haz]} risk`;
  paintMap(byFips);

  el("date-note").innerHTML = S.mode === "live"
    ? `Showing <b>${dayName(S.date)}</b> — scored from the live Open-Meteo
       forecast pulled ${hoursAgo(S.live.generated)} h ago, covering
       ${S.live.n_counties} counties.`
    : `Showing <b>${dayName(S.date)}</b> — scored from observed station
       rainfall. Past days are available around every real federal disaster
       declaration and for the last 60 observed days
       (through ${S.meta.observed_through}).`;

  el("map-foot").textContent =
    `${rows.length} counties scored on this date, of ${S.meta.n_counties_mapped} ` +
    `drawn. Shading is relative: "High" means this county-day sits in the top 1% ` +
    `of all county-days, not that a disaster is likely.`;

  const top = rows.slice().sort((a, b) => b[S.haz] - a[S.haz]).slice(0, 12);
  el("top-table").querySelector("tbody").innerHTML = top.map((r, i) => {
    const lv = level(r[S.haz], S.haz);
    return `<tr class="clickrow" data-fips="${r.fips}">
      <td class="num">${i + 1}</td>
      <td>${cname(r.fips)}</td>
      <td class="num"><span class="dot" style="background:${
        RAMP[level(r.flood, "flood")]}"></span>${pct(r.flood)}</td>
      <td class="num"><span class="dot" style="background:${
        RAMP[level(r.storm, "storm")]}"></span>${pct(r.storm)}</td>
      <td class="num">${r.fwd3 == null ? "—" : r.fwd3.toFixed(0) + " mm"}</td>
    </tr>`;
  }).join("");
  el("top-table").querySelectorAll(".clickrow").forEach((tr) =>
    tr.addEventListener("click", () => selectCounty(tr.dataset.fips)));

  if (S.fips) paintDetail(byFips.get(S.fips));
}

const hoursAgo = (iso) => Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 36e5));

// ---------------------------------------------------------------- detail

function selectCounty(fips) {
  if (!county(fips)) return;
  S.fips = fips;
  el("detail-empty").classList.add("hidden");
  el("detail-body").classList.remove("hidden");
  el("county-pick").value = fips;
  render();
  loadAlerts(fips);
  loadWeather(fips);
}

function clearCounty() {
  S.fips = null;
  el("detail-empty").classList.remove("hidden");
  el("detail-body").classList.add("hidden");
  render();
}

function paintDetail(r) {
  const fips = S.fips;
  const c = county(fips);
  el("d-name").textContent = `${c[0]}, ${c[1]}`;
  el("d-sub").textContent = `FIPS ${fips} · ${c[2].toFixed(2)}, ${c[3].toFixed(2)}`;

  if (!c[4]) {
    el("d-outcome").innerHTML = MODELLED_STATES.has(c[1])
      ? `<div class="outcome none">No weather station in this county in the
         source data, so the model produces no risk for it. The live weather and
         official warnings below are still real.</div>`
      : `<div class="outcome none">Risk modelling currently covers the five Gulf
         states; ${c[1]} is outside it. The live weather and official warnings
         below are real and work nationwide.</div>`;
  } else if (!r) {
    el("d-outcome").innerHTML =
      `<div class="outcome none">No model risk exported for this date. The
       prototype ships risk only around real disaster dates and for the recent
       window, to keep the download small.</div>`;
  } else if (S.mode === "history" && (r.floodActive || r.stormActive)) {
    const what = r.floodActive && r.stormActive ? "flood and storm"
      : r.floodActive ? "flood" : "storm";
    el("d-outcome").innerHTML =
      `<div class="outcome hit">A federal ${what} disaster was active here on
       this date. That is recorded ground truth, not a prediction — use it to
       check what the model said going in.</div>`;
  } else {
    el("d-outcome").innerHTML = "";
  }

  for (const haz of ["flood", "storm"]) {
    const p = r ? r[haz] : null;
    const lv = p == null ? null : level(p, haz);
    el(`rc-${haz}-band`).textContent = lv == null ? "no data" : BANDS[lv];
    el(`rc-${haz}-band`).style.color = lv == null ? "" : RAMP[lv];
    el(`rc-${haz}-num`).textContent = p == null ? "—" : pct(p);
    const bar = el(`rc-${haz}-bar`);
    bar.style.width = lv == null ? "0%" : (lv + 1) * 20 + "%";
    bar.style.background = lv == null ? "" : RAMP[lv];
  }

  el("risk-expl").innerHTML = r
    ? `These are probabilities that a federal ${""}flood or storm declaration
       <b>begins</b> in this county within the next ${S.meta.horizon_days} days.
       The base rate is ${pct(S.meta.flood.test_prevalence, 3)} for floods and
       ${pct(S.meta.storm.test_prevalence, 3)} for storms, so a few percent is
       already tens of times a normal day — the band label is the number to read,
       not the raw percentage.`
    : "";

  eventList(fips);
}

function eventList(fips) {
  const evs = S.events[fips] || [];
  if (!evs.length) {
    el("d-events").innerHTML = MODELLED_STATES.has(county(fips)[1])
      ? `<p class="quiet">No federal flood or storm disaster declarations on
         record for this county since 2010.</p>`
      : `<p class="quiet">Disaster history is loaded for the five Gulf states
         only, so none is listed here. This county may still have declarations
         on record.</p>`;
    return;
  }
  el("d-events").innerHTML = evs.map((e) => `
    <button class="evt" data-begin="${e.begin}">
      <span class="evt-d">${dayName(e.begin)}${
        e.end !== e.begin ? ` &rarr; ${dayName(e.end)}` : ""}</span>
      <span class="pill ${e.hazard}">${
        e.hazard === "both" ? "flood + storm" : e.hazard}</span>
      <span class="evt-t">${e.type} — ${e.title}</span>
    </button>`).join("");
  el("d-events").querySelectorAll(".evt").forEach((b) =>
    b.addEventListener("click", () => gotoDate(b.dataset.begin)));
}

function gotoDate(iso) {
  setMode("history");
  S.date = snap(iso);
  render();
  loadWeather(S.fips);
  el("map").scrollIntoView({ behavior: "smooth", block: "center" });
}

// ------------------------------------------------- live warnings (NWS)

async function loadAlerts(fips) {
  const c = county(fips);
  const box = el("d-alerts");
  box.innerHTML = `<p class="quiet">Loading from weather.gov&hellip;</p>`;
  const req = ++S.alertReq;
  try {
    const url = `https://api.weather.gov/alerts/active?point=${
      c[2].toFixed(4)},${c[3].toFixed(4)}`;
    const res = await fetch(url, { headers: { Accept: "application/geo+json" } });
    if (!res.ok) throw new Error("weather.gov " + res.status);
    const js = await res.json();
    if (req !== S.alertReq) return;
    const fs = js.features || [];
    if (!fs.length) {
      box.innerHTML =
        `<p class="ok">No active National Weather Service warnings for this
         location right now. This is a real query against weather.gov, made
         when you clicked the county.</p>`;
      return;
    }
    box.innerHTML = fs.slice(0, 6).map((f) => {
      const p = f.properties;
      const sev = (p.severity || "Unknown").toLowerCase();
      return `<div class="alert sev-${sev}">
        <div class="al-h">${p.event}
          <span class="al-sev">${p.severity || "—"} · ${p.urgency || "—"}</span>
        </div>
        <div class="al-b">${p.headline || ""}</div>
        ${p.ends ? `<div class="al-f">in force until ${
          new Date(p.ends).toLocaleString()}</div>` : ""}
      </div>`;
    }).join("");
  } catch (e) {
    if (req !== S.alertReq) return;
    box.innerHTML = `<p class="warn">Could not reach weather.gov (${e.message}).</p>`;
  }
}

// ------------------------------------------------ weather (Open-Meteo)

function card(day, w) {
  const cls = w.precip >= 25 ? "wet" : w.precip >= 5 ? "damp" : "";
  return `<div class="fcd ${cls}">
    <div class="fcd-d">${dayName(day)}</div>
    <div class="fcd-p">${w.precip == null ? "—" : w.precip.toFixed(1)} mm</div>
    <div class="fcd-t">${w.tmax == null ? "—" : w.tmax.toFixed(0)}&deg; /
      ${w.tmin == null ? "—" : w.tmin.toFixed(0)}&deg;C</div>
    <div class="fcd-w">wind ${w.wind == null ? "—" : w.wind.toFixed(0)}${
      w.gust ? ` · gust ${w.gust.toFixed(0)}` : ""} km/h</div>
  </div>`;
}

async function loadWeather(fips) {
  const c = county(fips);
  const box = el("d-forecast");
  const req = ++S.wxReq;

  const cached = S.mode === "live" && S.live.weather[fips];
  if (cached) {
    const days = Object.keys(cached).sort();
    box.innerHTML = days.map((d) => card(d, cached[d])).join("") +
      `<p class="note">Open-Meteo forecast, pulled ${
        hoursAgo(S.live.generated)} h ago — the same numbers the model was
        scored on.</p>`;
    return;
  }

  box.innerHTML = `<p class="quiet">Loading from Open-Meteo&hellip;</p>`;
  const start = S.date;
  const end = addDays(S.date, 3);
  const past = Date.parse(end) < Date.now() - 6 * 864e5;
  const base = past
    ? "https://archive-api.open-meteo.com/v1/archive"
    : "https://api.open-meteo.com/v1/forecast";
  const q = new URLSearchParams({
    latitude: c[2].toFixed(4), longitude: c[3].toFixed(4),
    daily: "precipitation_sum,temperature_2m_max,temperature_2m_min," +
           "wind_speed_10m_max,wind_gusts_10m_max",
    timezone: "America/Chicago", start_date: start, end_date: end,
  });
  try {
    const res = await fetch(`${base}?${q}`);
    if (!res.ok) throw new Error("Open-Meteo " + res.status);
    const d = (await res.json()).daily;
    if (req !== S.wxReq) return;
    box.innerHTML = d.time.map((t, i) => card(t, {
      precip: d.precipitation_sum[i], tmax: d.temperature_2m_max[i],
      tmin: d.temperature_2m_min[i], wind: d.wind_speed_10m_max[i],
      gust: d.wind_gusts_10m_max[i],
    })).join("") +
      `<p class="note">${past
        ? "Open-Meteo ERA5 reanalysis — what actually happened on these days."
        : "Open-Meteo forecast."} Queried live when you clicked.</p>`;
  } catch (e) {
    if (req !== S.wxReq) return;
    box.innerHTML = `<p class="warn">Could not reach Open-Meteo (${e.message}).</p>`;
  }
}

// ------------------------------------------------- transfer validation

function transferCard() {
  const t = S.meta.transfer;
  const loss = (r) => 1 - r;

  el("mc-transfer").innerHTML = [
    ["Hurricane skill retained", pct(t.storm_retention, 0),
      `in a state held out of training — a ${pct(loss(t.storm_retention), 0)} loss`],
    ["Flood skill retained", pct(t.flood_retention, 0),
      "no measurable loss moving to a new state"],
    ["States tested", `${t.spatial.length}`,
      "each removed from training entirely, then scored"],
  ].map(([k, v, n]) =>
    `<div class="card hero"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  el("transfer-read").innerHTML =
    `<b>Read this as the headline number.</b> Drop a state out of training and
     the hurricane model still delivers ${pct(t.storm_retention, 0)} of the
     average precision it reaches on states it knows — it gives back
     ${pct(loss(t.storm_retention), 0)}. The flood model gives back nothing.
     Both are measured on identical rows, so the base rate cancels and the ratio
     is the whole story.`;

  const rows = t.spatial.filter((r) => r.storm.retention !== null);
  el("mc-states").querySelector("tbody").innerHTML = rows.map((r) => {
    const s = r.storm;
    const cls = s.retention >= 0.85 ? "good" : s.retention >= 0.7 ? "" : "bad";
    return `<tr>
      <td>${r.name}</td>
      <td class="num">${fmtInt(s.n_pos)}</td>
      <td class="num">${s.pooled_ap.toFixed(5)}</td>
      <td class="num">${s.transfer_ap.toFixed(5)}</td>
      <td class="num ${cls}">${pct(s.retention, 0)}</td>
    </tr>`;
  }).join("");

  el("transfer-haz").innerHTML =
    `Hurricane figures shown; flood onsets are too sparse in three of the five
     states to score a state-level average precision (fewer than
     ${t.min_positives} in the test window), so those cells would be noise
     rather than measurement. <b>Why flood retains everything and hurricane does
     not:</b> the flood model is one monotone feature, so its ranking cannot
     change when it is refitted elsewhere — it carries no local thresholds to
     lose. The gradient-boosted hurricane model buys
     ${S.meta.storm.vs_best_baseline.toFixed(2)}× over its best
     baseline at home and pays ${pct(loss(t.storm_retention), 0)} of it back on
     unfamiliar ground. That trade is the design decision this project is
     really about.`;

  el("mc-years").querySelector("tbody").innerHTML = t.temporal.map((y) => `
    <tr>
      <td>${y.year}</td>
      <td class="num">${y.flood.ap === null ? "—" : y.flood.ap.toFixed(5)}</td>
      <td class="num">${fmtInt(y.flood.n_pos)}</td>
      <td class="num">${y.storm.ap === null ? "—" : y.storm.ap.toFixed(5)}</td>
      <td class="num">${fmtInt(y.storm.n_pos)}</td>
    </tr>`).join("");
}

// ------------------------------------------------------------ model card

function modelCard() {
  const m = S.meta;
  el("mc-cards").innerHTML = [
    ["Question asked", `${m.horizon_days}-day lead`, m.target],
    ["Counties modelled", `${m.n_counties_modeled} / ${m.n_counties_mapped}`,
      "only counties with a weather station in the source data"],
    ["Train / test", `${m.splits.train[0]}-${m.splits.train[1]} / ` +
      `${m.splits.test[0]}-${m.splits.test[1]}`, "split by year, never shuffled"],
    ["Flood onsets", `${fmtInt(m.flood.n_train_pos)} train`,
      `${fmtInt(m.flood.n_test_pos)} in test — the binding constraint on complexity`],
    ["Storm lift", `${m.storm.lift_over_prevalence.toFixed(0)}×`,
      "average precision over the base rate"],
    ["Flood lift", `${m.flood.lift_over_prevalence.toFixed(0)}×`,
      "average precision over the base rate"],
  ].map(([k, v, n]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  el("mc-table").querySelector("tbody").innerHTML = ["flood", "storm"]
    .map((haz) => {
      const h = m[haz];
      const bad = h.vs_best_baseline < 1.0;
      return `<tr>
        <td>${HAZ_LABEL[haz]}</td>
        <td><code>${h.model}</code></td>
        <td class="num">${h.test_ap.toFixed(5)}</td>
        <td class="num">${pct(h.test_prevalence, 3)}</td>
        <td class="num">${h.lift_over_prevalence.toFixed(1)}×</td>
        <td class="num ${bad ? "bad" : ""}">${h.vs_best_baseline.toFixed(2)}×
          <span class="dim">(${h.best_baseline})</span></td>
      </tr>`;
    }).join("");

  transferCard();

  el("mc-limits").innerHTML = [
    `<b>The label is a bureaucratic event, not a hazard.</b> The target is a FEMA
     disaster declaration, which depends on damage estimates and state requests.
     A severe flood with no declaration counts as a negative.`,
    `<b>Training rainfall is observed, not forecast.</b> Features use station
     rainfall over the prediction window, which a real forecaster would not have.
     The fix is to retrain on archived forecasts (GEFSv12 reforecast); until then
     the lead-time claim is not operationally honest.`,
    `<b>Risk covers ${m.n_counties_modeled} of the ${m.n_counties_mapped}
     counties on the map.</b> Training is restricted to the five Gulf states,
     and within them to counties with a rainfall station in the source data.
     Every other county is drawn unshaded and serves live weather and official
     warnings only. The held-out-state result below is the evidence for what
     extending coverage would cost.`,
    `<b>No soil moisture, no river gauges.</b> Antecedent wetness is proxied by
     30- and 90-day rainfall, which is a weak substitute for catchment state.`,
    `<b>No hurricane wind field.</b> Storm risk is driven by rainfall and season;
     track, intensity and storm surge are absent entirely.`,
    `<b>Only ${fmtInt(m.flood.n_test_pos)} flood and
     ${fmtInt(m.storm.n_test_pos)} storm onsets in the test period.</b> Confidence
     intervals on every metric here are correspondingly wide, and three of the
     five states cannot support a state-level flood score at all.`,
  ].map((s) => `<li>${s}</li>`).join("");

  el("proto-banner").innerHTML =
    `Risk is produced by models trained on ${m.splits.train[0]}-${m.splits.train[1]}
     station rainfall and federal disaster declarations across
     ${m.n_counties_modeled} Gulf Coast counties, then validated on
     ${m.splits.test[0]}-${m.splits.test[1]} — years the models never saw. Weather
     and official warnings are fetched live from Open-Meteo and weather.gov when
     you click a county. <b>Held-out-state skill retention:
     ${pct(m.transfer.storm_retention, 0)} hurricane,
     ${pct(m.transfer.flood_retention, 0)} flood.</b>`;

  el("ft-prov").textContent =
    `Risk: ${m.flood.model} (flood) and ${m.storm.model} (storm), trained on ` +
    `station rainfall + FEMA declarations, observed through ${m.observed_through}. ` +
    `Exported ${m.generated}. Warnings: api.weather.gov. Weather: Open-Meteo.`;
}

// ------------------------------------------------------------------ wire

function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll("#mode-seg button").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === mode));
  S.date = snap(S.date || dates()[0]);
}

function wire() {
  document.querySelectorAll("#mode-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      setMode(b.dataset.mode);
      render();
      if (S.fips) loadWeather(S.fips);
    }));

  document.querySelectorAll("#haz-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      S.haz = b.dataset.haz;
      document.querySelectorAll("#haz-seg button").forEach((x) =>
        x.classList.toggle("on", x === b));
      render();
    }));

  el("d-date").addEventListener("change", (e) => {
    S.date = snap(e.target.value);
    render();
    if (S.fips) loadWeather(S.fips);
  });

  const step = (n) => {
    const ds = dates();
    const i = ds.indexOf(S.date);
    S.date = ds[Math.max(0, Math.min(ds.length - 1, (i < 0 ? 0 : i) + n))];
    render();
    if (S.fips) loadWeather(S.fips);
  };
  el("d-prev").addEventListener("click", () => step(-1));
  el("d-next").addEventListener("click", () => step(1));

  el("d-close").addEventListener("click", clearCounty);
  el("county-pick").addEventListener("change", (e) => selectCounty(e.target.value));
}

function fillPicker() {
  const all = Object.entries(S.meta.counties)
    .sort((a, b) => (a[1][1] + a[1][0]).localeCompare(b[1][1] + b[1][0]));
  const opt = ([f, c]) => `<option value="${f}">${c[0]}, ${c[1]}</option>`;
  const group = (label, rows) =>
    `<optgroup label="${label}">${rows.map(opt).join("")}</optgroup>`;
  el("county-pick").innerHTML =
    `<option value="">— choose —</option>` +
    group("Risk modelled", all.filter(([, c]) => c[4])) +
    group("Weather & warnings only", all.filter(([, c]) => !c[4]));
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* private mode */ }
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  const apply = (name) => {
    document.documentElement.setAttribute("data-theme", name);
    try { localStorage.setItem("theme", name); } catch (e) { /* private mode */ }
    document.querySelectorAll("[data-set-theme]").forEach((b) =>
      b.classList.toggle("on", b.dataset.setTheme === name));
  };
  apply(saved || (prefersLight ? "light" : "dark"));
  document.querySelectorAll("[data-set-theme]").forEach((b) =>
    b.addEventListener("click", () => apply(b.dataset.setTheme)));
}

async function boot() {
  const get = async (f) => {
    const r = await fetch(f, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${f}: ${r.status}`);
    return r.json();
  };
  [S.geo, S.meta, S.events, S.hist, S.live] = await Promise.all([
    get("data/geo.json"), get("data/meta.json"), get("data/events.json"),
    get("data/history.json"), get("data/live.json"),
  ]);

  initTheme();
  drawMap();
  fillPicker();
  modelCard();
  setMode("live");
  S.date = dates()[0];
  wire();
  render();
}

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<p class="warn">Failed to load dashboard data: ${e.message}</p>`);
  throw e;
});
