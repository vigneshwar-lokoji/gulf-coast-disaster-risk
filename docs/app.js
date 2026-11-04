/* Gulf Coast flood & hurricane risk.
 *
 * Risk comes precomputed from docs/data (export_v2.py, fetch_live.py), scored
 * by the XGBoost models from one sealed training run. Weather and official
 * warnings are fetched live in the browser from two keyless CORS-enabled APIs,
 * so the whole thing runs on static hosting.
 */

const SVGNS = "http://www.w3.org/2000/svg";
const RAMP = ["#1b5e3f", "#4caf5c", "#ffc147", "#ff8a3d", "#ff3b2f"];
const BANDS = ["Normal", "Slightly raised", "Elevated", "High", "Very high"];
const HAZ_LABEL = { flood: "Flood", storm: "Hurricane / tropical storm" };
// States the risk models and the hazard event history actually cover.
const MODELLED_STATES = new Set(["AL", "FL", "LA", "MS", "TX"]);
const LEADS = [1, 2, 3];

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
  // Official warnings in force now, kept in state because two panels read them:
  // the raw list, and the live comparison against the model.
  alerts: { state: "idle", feats: [], at: null, err: null },
};

// ------------------------------------------------------------------ data

/** Sorted list of dates the current mode can show. */
function dates() {
  return Object.keys(S.mode === "live" ? S.live.days : S.hist.days).sort();
}

function dayRows(d) {
  return (S.mode === "live" ? S.live.days : S.hist.days)[d] || [];
}

/* Live rows are [fips, p_flood, p_storm]; history rows carry the rainfall, the
 * ground-truth active flags and the six exact-lead probabilities too. One shape
 * for the rest of the code, with absent fields null rather than zero so the UI
 * can say "not exported" instead of showing a confident zero. */
function rowObj(r) {
  return {
    fips: r[0], flood: r[1], storm: r[2],
    fwd3: r.length > 3 ? r[3] : null,
    ratio: r.length > 4 ? r[4] : null,
    floodActive: r.length > 5 ? !!r[5] : false,
    stormActive: r.length > 6 ? !!r[6] : false,
    leads: r.length > 12
      ? { flood: [r[7], r[8], r[9]], storm: [r[10], r[11], r[12]] }
      : null,
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
      (gt ? ` · recorded ${S.haz === "flood" ? "flood episode" : "cyclone pass"}`
            + " in progress on this date" : "");
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
    ? `<b>${dayName(S.date)}</b> — live forecast, pulled ${
        hoursAgo(S.live.generated)} h ago.`
    : `<b>${dayName(S.date)}</b> — observed rainfall, through ${
        S.meta.observed_through}.`;

  el("map-foot").textContent =
    `${rows.length} of ${S.meta.n_counties_mapped} counties scored. ` +
    `"High" means the top 1% of all county-days, not that a disaster is likely.`;

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
  S.alerts = { state: "loading", feats: [], at: null, err: null };
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
      ? `<div class="outcome none">No weather station here in the source data, so
         no risk is modelled. Live weather and warnings below are real.</div>`
      : `<div class="outcome none">Risk modelling covers the five Gulf states;
         ${c[1]} is outside it. Live weather and warnings work nationwide.</div>`;
  } else if (!r) {
    el("d-outcome").innerHTML =
      `<div class="outcome none">No risk exported for this date.</div>`;
  } else if (S.mode === "history" && (r.floodActive || r.stormActive)) {
    const both = r.floodActive && r.stormActive;
    const what = both
      ? "A recorded flood episode and a cyclone pass were"
      : r.floodActive ? "A recorded flood episode was"
                      : "A recorded cyclone pass was";
    el("d-outcome").innerHTML =
      `<div class="outcome hit">${what} in progress here on this date —
       observed, not predicted.</div>`;
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
    ? `Chance of onset within ${S.meta.horizon_days} days. A normal day is
       ${pct(S.meta.flood.test_prevalence, 3)} for floods and
       ${pct(S.meta.storm.test_prevalence, 3)} for cyclones, so read the band,
       not the percentage.`
    : "";

  liveCheck();
  leadBlock(r);
  eventList(fips);
}

/** The three exact-lead models for this county-day, each in its own band.
 *
 * A +3 day model sees roughly a third of the pooled prevalence, so its
 * probabilities are smaller for reasons that have nothing to do with this
 * county. Each lead is therefore banded against its own quantiles, which is
 * what makes the three columns comparable to each other.
 */
function leadBlock(r) {
  const box = el("d-leads");
  if (!r || !r.leads) {
    box.innerHTML = `<p class="quiet">No exact-lead risk exported for this
      county-day.</p>`;
    return;
  }
  box.innerHTML = ["flood", "storm"].map((haz) => {
    const cells = LEADS.map((k, i) => {
      const v = r.leads[haz][i];
      if (v == null) return `<span class="lead-d"><b>—</b><i>+${k}d</i></span>`;
      const lv = level(v, `${haz}_l${k}`);
      return `<span class="lead-d" title="${BANDS[lv]} — onset starting exactly
        ${k} day${k > 1 ? "s" : ""} from now">
        <b style="color:${RAMP[lv]}">${pct(v, 2)}</b><i>+${k}d</i></span>`;
    }).join("");
    const known = r.leads[haz].filter((v) => v != null);
    const worst = known.length
      ? Math.max(...LEADS.map((k, i) =>
          r.leads[haz][i] == null ? -1 : level(r.leads[haz][i], `${haz}_l${k}`)))
      : null;
    return `<div class="lead-row">
      <span class="lead-h">${HAZ_LABEL[haz]}</span>
      ${cells}
      <span class="lead-pk" style="color:${worst == null ? "" : RAMP[worst]}">${
        worst == null ? "—" : BANDS[worst]}</span>
    </div>`;
  }).join("");
}

function eventList(fips) {
  const evs = S.events[fips] || [];
  if (!evs.length) {
    el("d-events").innerHTML = MODELLED_STATES.has(county(fips)[1])
      ? `<p class="quiet">Nothing on record for this county in
         ${S.meta.splits.train[0]}&ndash;${S.meta.observed_through.slice(0, 4)}.</p>`
      : `<p class="quiet">Event history is loaded for the five Gulf states
         only.</p>`;
    return;
  }
  el("d-events").innerHTML = evs.map((e) => `
    <div class="evt-wrap">
      <button class="evt" data-begin="${e.begin}">
        <span class="evt-d">${dayName(e.begin)}${
          e.end !== e.begin ? ` &rarr; ${dayName(e.end)}` : ""}</span>
        <span class="pill ${e.hazard}">${
          e.hazard === "both" ? "flood + storm" : e.hazard}</span>
        <span class="evt-t">${e.type} — ${e.title}</span>
      </button>
      ${verdict(e)}
    </div>`).join("");
  el("d-events").querySelectorAll(".evt").forEach((b) =>
    b.addEventListener("click", () => gotoDate(b.dataset.begin)));
}

// The model predicts onset within 3 days, so the days that decide whether it
// saw this event coming are begin-3 .. begin-1. Anything on or after the onset
// day is hindsight and is deliberately not shown here.
const SPLIT_NOTE = {
  train: ["in-sample", "This event is inside the training years, so the model "
    + "had already seen it. Not evidence of forecasting skill."],
  val: ["validation", "This event is in the validation years, used while "
    + "choosing the model."],
  test: ["out-of-sample", "This event is after the training cutoff. The model "
    + "had never seen it, so this is a genuine forecast."],
};

/** Band key for a reading taken `i` steps into a d-3, d-2, d-1 series.
 *
 * The pooled model is one model everywhere, so it bands against the hazard's
 * own quantiles. The exact-lead series is three different models -- the +3 model
 * on d-3, the +1 model on d-1 -- so each day needs the quantiles of the model
 * that produced it, or a +3 reading would be judged against a scale it can
 * never reach.
 */
const bandKey = (haz, kind, i) => kind === "pre" ? haz : `${haz}_l${3 - i}`;

/** Peak band index over a d-3..d-1 series, or null if the whole series is gaps. */
function peakBand(haz, kind, series) {
  const lv = series
    .map((v, i) => v == null ? null : level(v, bandKey(haz, kind, i)))
    .filter((x) => x != null);
  return lv.length ? Math.max(...lv) : null;
}

function leadRow(haz, kind, label, series) {
  const cells = series.map((v, i) => {
    const d = 3 - i;
    if (v == null) return `<span class="lead-d"><b>—</b><i>${d}d</i></span>`;
    const lv = level(v, bandKey(haz, kind, i));
    const what = kind === "pre"
      ? `pooled 3-day model, ${d} day${d > 1 ? "s" : ""} before onset`
      : `the +${d} day model, run ${d} day${d > 1 ? "s" : ""} before onset`;
    return `<span class="lead-d" title="${pct(v)} — ${BANDS[lv]} — ${what}">
      <b style="color:${RAMP[lv]}">${pct(v, 2)}</b><i>${d}d</i></span>`;
  }).join("");
  const lv = peakBand(haz, kind, series);
  return `<div class="lead-row">
    <span class="lead-h">${label}</span>
    ${cells}
    <span class="lead-pk" style="color:${lv == null ? "" : RAMP[lv]}">${
      lv == null ? "—" : BANDS[lv]}</span>
  </div>`;
}

function verdict(e) {
  if (!e.pre) {
    return `<div class="vs vs-miss"><div class="vs-head">
      <span class="vs-flag">No model reading</span>
      ${SPLIT_NOTE[e.split]
        ? `<span class="vs-split" title="${SPLIT_NOTE[e.split][1]}">${
            SPLIT_NOTE[e.split][0]}</span>` : ""}
    </div><div class="vs-col"><div class="vs-real">None of the three days before
      onset are in this county's rainfall record, so the models were never
      asked.</div></div></div>`;
  }
  const hazards = Object.keys(e.pre);
  const rows = hazards.map((haz) =>
    leadRow(haz, "pre", `${HAZ_LABEL[haz]} · pooled`, e.pre[haz]) +
    (e.lead && e.lead[haz]
      ? leadRow(haz, "lead", `${HAZ_LABEL[haz]} · exact lead`, e.lead[haz])
      : "")).join("");

  // The flag goes on the exact-lead reading, not the pooled one. The pooled
  // model answers the easier "sometime in the next three days", so flagging on
  // it would credit the system with warnings a real morning run never gave.
  const which = hazards.map((h) =>
    (e.lead && e.lead[h]) ? peakBand(h, "lead", e.lead[h])
                          : peakBand(h, "pre", e.pre[h]));
  const peakLv = Math.max(...which.map((x) => x == null ? -1 : x));
  const caught = peakLv >= 3;
  const [tag, note] = SPLIT_NOTE[e.split] || ["", ""];
  const real = e.end !== e.begin
    ? `began ${dayName(e.begin)}, ran through ${dayName(e.end)}`
    : `began ${dayName(e.begin)}`;

  return `<div class="vs ${caught ? "vs-hit" : "vs-miss"}">
    <div class="vs-head">
      <span class="vs-flag">${caught
        ? "Flagged by the model that would really have run"
        : "Not flagged at the exact lead"}</span>
      ${tag ? `<span class="vs-split" title="${note}">${tag}</span>` : ""}
    </div>
    <div class="vs-col">
      <div class="vs-lbl">What the models said in the 3 days before</div>
      ${rows}
      ${e.rain == null ? "" : `<div class="vs-rain">Rain the model could see
         coming: <b>${e.rain < 10 ? e.rain.toFixed(1) : e.rain.toFixed(0)} mm</b>
         over the following 3 days</div>`}
    </div>
    <div class="vs-col">
      <div class="vs-lbl">What actually happened</div>
      <div class="vs-real">${e.type} — ${e.title}, ${real}.</div>
    </div>
  </div>`;
}

function gotoDate(iso) {
  setMode("history");
  S.date = snap(iso);
  render();
  loadWeather(S.fips);
  el("map").scrollIntoView({ behavior: "smooth", block: "center" });
}

// ------------------------------------------------- live warnings (NWS)

/* Active alerts are pulled per state and filtered on the county's SAME code.
 * A point query against the county centroid was the obvious approach and is
 * wrong: warning polygons are drawn around rivers and coastlines, so a product
 * issued for the whole county routinely misses its centroid. SAME is an exact
 * county match, which is the unit this dashboard reports in. */
const ALERT_TTL = 3e5;
const alertCache = new Map();

async function stateAlerts(st, force) {
  if (force) alertCache.delete(st);
  const hit = alertCache.get(st);
  if (hit && Date.now() - hit.at < ALERT_TTL) return hit.feats;
  const res = await fetch(`https://api.weather.gov/alerts/active?area=${st}`,
    { headers: { Accept: "application/geo+json" } });
  if (!res.ok) throw new Error("weather.gov " + res.status);
  const feats = (await res.json()).features || [];
  alertCache.set(st, { at: Date.now(), feats });
  return feats;
}

async function loadAlerts(fips, force) {
  const c = county(fips);
  const box = el("d-alerts");
  box.innerHTML = `<p class="quiet">Loading from weather.gov&hellip;</p>`;
  const req = ++S.alertReq;
  S.alerts = { state: "loading", feats: [], at: null, err: null };
  liveCheck();
  try {
    const all = await stateAlerts(c[1], force);
    if (req !== S.alertReq) return;
    const same = "0" + fips;
    const fs = all.filter((f) => {
      const g = (f.properties && f.properties.geocode) || {};
      return (g.SAME || []).includes(same);
    });
    S.alerts = { state: "ok", feats: fs, at: Date.now(), err: null };
    liveCheck();
    if (!fs.length) {
      box.innerHTML =
        `<p class="ok">No active National Weather Service warnings for this
         location right now.</p>`;
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
    S.alerts = { state: "err", feats: [], at: Date.now(), err: e.message };
    liveCheck();
    box.innerHTML = `<p class="warn">Could not reach weather.gov (${e.message}).</p>`;
  }
}

// ------------------------------------------- live check: official vs model

/* The "actual" side of this comparison is the National Weather Service product
 * in force right now — the only live hazard call for a county that is
 * authoritative. A probability and a warning are not the same kind of object,
 * so both sides are mapped onto one 0-3 action ladder; that is what makes them
 * comparable at all, and the raw values stay visible underneath. */
const OFFICIAL = ["none", "Advisory", "Watch", "Warning"];
const HAZ_RE = {
  flood: /flood|hydrolog/i,
  storm: /hurricane|tropical|storm surge|typhoon/i,
};
// Band 0-4 -> ladder 0-3. "Normal" and "slightly raised" are both no-call, and
// the model's flag threshold stays the top-1% band used to score past events.
const MODEL_LADDER = [0, 0, 1, 2, 3];

/** Strongest active NWS product for this hazard, on the same 0-3 ladder. */
function officialFor(haz) {
  let lv = 0, name = null;
  for (const f of S.alerts.feats) {
    const ev = (f.properties && f.properties.event) || "";
    if (!HAZ_RE[haz].test(ev)) continue;
    const r = /warning/i.test(ev) ? 3 : /watch/i.test(ev) ? 2 : 1;
    if (r > lv) { lv = r; name = ev; }
  }
  return { lv, name };
}

/* One rung apart is not a disagreement worth shouting about: "elevated with no
 * advisory out" is the normal state of a rare-event model. Only a gap of two or
 * more rungs, or a side at zero against a watch or warning, is called out. */
function compare(m, o) {
  if (m === o) return ["agree", m ? "Agree" : "Agree — neither flags it"];
  if (o === 0) return m >= 2 ? ["model", "Model only — nothing official out"]
                             : ["part", "Model slightly ahead"];
  if (m === 0) return o >= 2 ? ["nws", "Official only — model does not flag it"]
                             : ["part", "Official slightly ahead"];
  return m > o ? ["part", "Model hotter"] : ["part", "Official hotter"];
}

/** First live day: the one the model is actually being asked about right now. */
const liveToday = () => Object.keys(S.live.days).sort()[0];

function lcCell(label, value, sub, colour) {
  return `<span class="lc-cell"><i>${label}</i>
    <b${colour ? ` style="color:${colour}"` : ""}>${value}</b>
    <u>${sub}</u></span>`;
}

/* Rendered on selection and again when the warning feed lands, so the model
 * side appears immediately and the official side fills in. */
function liveCheck() {
  const box = el("d-live");
  const fips = S.fips;
  if (!box || !fips) return;
  const c = county(fips);
  const day = liveToday();
  const row = c[4]
    ? (S.live.days[day] || []).map(rowObj).find((r) => r.fips === fips)
    : null;
  const pending = S.alerts.state === "loading" || S.alerts.state === "idle";

  const rows = ["flood", "storm"].map((haz) => {
    const p = row ? row[haz] : null;
    const band = p == null ? null : level(p, haz);
    const mLv = band == null ? null : MODEL_LADDER[band];
    const model = band == null
      ? lcCell("this model", "no risk", "county not modelled", null)
      : lcCell("this model", BANDS[band], pct(p), RAMP[band]);

    let official, verdict;
    if (pending) {
      official = lcCell("NWS now", "&hellip;", "checking weather.gov", null);
      verdict = `<span class="lc-v">&nbsp;</span>`;
    } else if (S.alerts.state === "err") {
      official = lcCell("NWS now", "—", "feed unreachable", null);
      verdict = `<span class="lc-v">no comparison</span>`;
    } else {
      const o = officialFor(haz);
      official = lcCell("NWS now", o.lv ? OFFICIAL[o.lv] : "none",
        o.name || "nothing in force", o.lv ? RAMP[o.lv + 1] : null);
      if (mLv == null) {
        verdict = `<span class="lc-v">no model side</span>`;
      } else {
        const [cls, text] = compare(mLv, o.lv);
        verdict = `<span class="lc-v ${cls}">${text}</span>`;
      }
    }
    return `<div class="lc-row">
      <span class="lc-h">${HAZ_LABEL[haz]}</span>${model}${official}${verdict}
    </div>`;
  }).join("");

  const checked = S.alerts.at
    ? new Date(S.alerts.at).toLocaleTimeString() : "—";
  box.innerHTML = rows + `<div class="lc-foot">
    <span>Model: <b>${dayName(day)}</b>, scored ${
      hoursAgo(S.live.generated)} h ago. Official: checked ${checked}, re-checked
      every 5 minutes.</span>
    <button id="lc-refresh" class="ghost sm">Refresh</button>
  </div>`;
  el("lc-refresh").addEventListener("click", () => loadAlerts(fips, true));
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
        hoursAgo(S.live.generated)} h ago — what the model was scored on.</p>`;
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
        ? "Open-Meteo ERA5 reanalysis — what actually happened."
        : "Open-Meteo forecast."}</p>`;
  } catch (e) {
    if (req !== S.wxReq) return;
    box.innerHTML = `<p class="warn">Could not reach Open-Meteo (${e.message}).</p>`;
  }
}

// ------------------------------------------------------------ model card

const hours = (s) => (s / 3600).toFixed(1);
const leadRows = () => S.meta.lead_table;
const leadOf = (haz, k) => leadRows().find((r) => r.hazard === haz && r.lead === k);

/** The lead-time sweep: one model per lead, each against its own baseline. */
function leadCards() {
  const cards = [];
  for (const haz of ["flood", "storm"]) {
    const a = leadOf(haz, 1);
    const c = leadOf(haz, 3);
    const decay = 1 - c.xgb_ap / a.xgb_ap;
    cards.push([
      `${HAZ_LABEL[haz]}, 1 day out`,
      `${a.xgb_vs_base.toFixed(1)}×`,
      `average precision ${a.xgb_ap.toFixed(4)}, against ${a.baseline_ap.toFixed(4)}
       for the best single feature`,
    ], [
      `${HAZ_LABEL[haz]}, 3 days out`,
      `${c.xgb_vs_base.toFixed(1)}×`,
      `skill itself falls ${pct(decay, 0)} from the 1-day model, but the baseline
       falls further, so the margin holds`,
    ]);
  }
  el("mc-lead-cards").innerHTML = cards.map(([k, v, n]) =>
    `<div class="card hero"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  el("mc-leads").querySelector("tbody").innerHTML = leadRows().map((r) => {
    const win = r.xgb_ap >= r.tft_ap;
    const bad = (x) => x < 1.0 ? "bad" : "";
    return `<tr>
      <td>${HAZ_LABEL[r.hazard]}</td>
      <td class="num">+${r.lead} day</td>
      <td class="num">${pct(r.prevalence, 3)}</td>
      <td class="num">${r.baseline_ap.toFixed(4)}
        <span class="dim">(${r.baseline.replace("single_", "")})</span></td>
      <td class="num${win ? " good" : ""}">${r.xgb_ap.toFixed(4)}</td>
      <td class="num ${bad(r.xgb_vs_base)}">${r.xgb_vs_base.toFixed(2)}×</td>
      <td class="num${win ? "" : " good"}">${r.tft_ap.toFixed(4)}</td>
      <td class="num ${bad(r.tft_vs_base)}">${r.tft_vs_base.toFixed(2)}×</td>
    </tr>`;
  }).join("");

  const f1 = leadOf("flood", 1), f3 = leadOf("flood", 3);
  const s1 = leadOf("storm", 1), s3 = leadOf("storm", 3);
  const f2 = leadOf("flood", 2), s2 = leadOf("storm", 2);
  el("lead-read").innerHTML =
    `<b>Boosted skill decays monotonically with lead, in both hazards.</b>
     Flood average precision runs ${f1.xgb_ap.toFixed(4)} →
     ${f2.xgb_ap.toFixed(4)} → ${f3.xgb_ap.toFixed(4)} across +1, +2 and +3
     days; hurricane runs ${s1.xgb_ap.toFixed(4)} → ${s2.xgb_ap.toFixed(4)} →
     ${s3.xgb_ap.toFixed(4)}. The transformer is monotone on hurricanes but not
     on floods, where its +2 fit (${f2.tft_ap.toFixed(4)}) lands above its +1
     fit (${f1.tft_ap.toFixed(4)}); that inversion is left visible rather than
     smoothed. The ratio to baseline does not decay neatly, because the baseline
     is losing signal at the same time — read average precision for forecast
     quality and the ratio only for whether the complexity is earned.`;

  el("lead-foot").innerHTML =
    `Base rate is the share of test county-days with onset on exactly that day.
     Every cell is average precision on ${S.meta.splits.test[0]}&ndash;${
       S.meta.splits.test[1]}, unseen in training and selection. Green marks the
     better architecture at that lead: XGBoost takes ${
       leadRows().filter((r) => r.xgb_ap >= r.tft_ap).length} of
     ${leadRows().length}.`;
}

function modelCard() {
  const m = S.meta;
  el("mc-cards").innerHTML = [
    ["Question asked", `${m.horizon_days}-day lead`, m.target],
    ["Counties modelled", `${m.n_counties_modeled} / ${m.n_counties_mapped}`,
      "only counties with a weather station in the source data"],
    ["Train / val / test", `${m.splits.train[0]}–${m.splits.train[1]} / ` +
      `${m.splits.val[0]}–${m.splits.val[1]} / ` +
      `${m.splits.test[0]}–${m.splits.test[1]}`,
      "split by year, never shuffled — no row from a test year informs training"],
    ["Flood onsets", `${fmtInt(m.flood.n_train_pos)} train`,
      `${fmtInt(m.flood.n_test_pos)} in test — the binding constraint on complexity`],
    ["Flood lift", `${m.flood.lift_over_prevalence.toFixed(0)}×`,
      `average precision ${m.flood.test_ap.toFixed(4)} over a
       ${pct(m.flood.test_prevalence, 3)} base rate`],
    ["Hurricane lift", `${m.storm.lift_over_prevalence.toFixed(0)}×`,
      `average precision ${m.storm.test_ap.toFixed(4)} over a
       ${pct(m.storm.test_prevalence, 3)} base rate`],
  ].map(([k, v, n]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  leadCards();

  // Four rows, not two: the pooled result for each hazard under each
  // architecture. Both were fitted on the same panel and the same splits, so
  // the only thing differing down a hazard's pair is the model.
  const rows = [];
  for (const haz of ["flood", "storm"]) {
    const h = m[haz];
    rows.push([haz, "xgboost_monotone", h.test_ap, h.lift_over_prevalence,
      h.vs_best_baseline, h.best_baseline, h.test_ap >= h.tft.test_ap]);
    rows.push([haz, "temporal_fusion_transformer", h.tft.test_ap,
      h.tft.lift_over_prevalence, h.tft.vs_best_baseline, h.best_baseline,
      h.tft.test_ap > h.test_ap]);
  }
  el("mc-table").querySelector("tbody").innerHTML = rows.map(
    ([haz, name, ap, lift, vs, base, win]) => `<tr>
      <td>${HAZ_LABEL[haz]}</td>
      <td><code>${name}</code></td>
      <td class="num${win ? " good" : ""}">${ap.toFixed(5)}</td>
      <td class="num">${pct(m[haz].test_prevalence, 3)}</td>
      <td class="num">${lift.toFixed(1)}×</td>
      <td class="num ${vs < 1.0 ? "bad" : ""}">${vs.toFixed(2)}×
        <span class="dim">(${base.replace("single_", "")})</span></td>
    </tr>`).join("");

  const fr = m.flood.test_ap / m.flood.tft.test_ap;
  const sr = m.storm.test_ap / m.storm.tft.test_ap;
  el("mc-verdict").innerHTML =
    `<b>Gradient boosting wins both hazards, so the dashboard serves XGBoost.</b>
     Floods are close — ${m.flood.test_ap.toFixed(4)} against
     ${m.flood.tft.test_ap.toFixed(4)}, a ${fr.toFixed(2)}× edge. Hurricanes are
     not: ${m.storm.test_ap.toFixed(4)} against
     ${m.storm.tft.test_ap.toFixed(4)} (${sr.toFixed(2)}×), clearing baseline by
     only ${m.storm.tft.vs_best_baseline.toFixed(2)}×. Likely sample size rather
     than architecture — hurricane onset is
     ${pct(m.storm.test_prevalence, 3)} of county-days, so most 30-day encoder
     windows contain no event. The transformer cost ${m.flood.tft.epochs_run} and
     ${m.storm.tft.epochs_run} epochs on an ${m.run.gpu} against seconds of
     boosting. Both transformer columns are post-calibration
     (${m.flood.tft.calibration.replace(/_/g, " ")}), undoing the
     positive-class weight of 50 used in training.`;

  const bl = [];
  for (const haz of ["flood", "storm"]) {
    for (const [name, ap] of Object.entries(m[haz].baselines)) {
      bl.push([haz, name, ap, m[haz].test_ap / ap]);
    }
  }
  el("mc-baselines").querySelector("tbody").innerHTML = bl.map(
    ([haz, name, ap, ratio]) => `<tr>
      <td>${HAZ_LABEL[haz]}</td>
      <td><code>${name.replace("single_", "")}</code>${
        name === m[haz].best_baseline
          ? ` <span class="dim">strongest</span>` : ""}</td>
      <td class="num">${ap.toFixed(5)}</td>
      <td class="num ${ratio < 1.0 ? "bad" : ""}">${ratio.toFixed(2)}×</td>
    </tr>`).join("");

  const cv = m.coverage;
  el("mc-limits").innerHTML = [
    `<b>The label is an observational record, not hazard physics.</b> Flood
     positives come from ${m.label_source.flood}, so a real flood nobody wrote
     up counts as a negative. Hurricane positives (${m.label_source.storm}) are
     purely geometric — a graze and a major landfall are the same positive.`,
    `<b>Training rainfall is observed, not forecast.</b> Features use station
     rainfall over the prediction window, which a forecaster would not have on
     the morning the model runs. The largest overstatement in the project; treat
     every lead-time number as an upper bound until retrained on GEFSv12.`,
    `<b>Risk covers ${m.n_counties_modeled} of the ${m.n_counties_mapped}
     counties on the map.</b> Five Gulf states only, and within them only
     counties with a rainfall station in the source data.`,
    `<b>No held-out-state validation in this run.</b> Nothing here supports a
     claim about counties outside the Gulf Coast.`,
    `<b>The rainfall record has holes.</b> The median county has
     ${fmtInt(cv.median_days_per_county)} of ${fmtInt(cv.calendar_days)}
     calendar days, the worst ${fmtInt(cv.min_days_per_county)}, and
     ${cv.counties_under_80pct} counties fall below 80%.
     ${fmtInt(cv.events_listed - cv.events_listed_with_leadup)} of
     ${fmtInt(cv.events_listed)} listed events have no lead-up in the record.`,
    `<b>Hurricane labels stop at ${m.label_end.storm}.</b> HURDAT2 is finalised a
     season behind, so hurricane metrics score
     ${m.splits.test[0]}–${m.label_end.storm} while floods use the full test
     window. The two test sets are not the same size.`,
    `<b>No soil moisture, no river gauges.</b> Antecedent wetness is proxied by
     30-, 60- and 90-day rainfall, a weak substitute for catchment state.`,
    `<b>No hurricane wind field.</b> Track forecast, intensity and storm surge
     are absent entirely.`,
    `<b>Only ${fmtInt(m.flood.n_test_pos)} flood and
     ${fmtInt(m.storm.n_test_pos)} hurricane onsets in test.</b> Confidence
     intervals are correspondingly wide, and none are reported.`,
  ].map((s) => `<li>${s}</li>`).join("");

  const r = m.run;
  el("mc-prov").innerHTML =
    `Every number on this page comes from one sealed training run,
     <code>${r.run_id}</code>, finished ${r.finished_utc} after
     ${hours(r.wall_seconds)} hours on an ${r.gpu}
     (xgboost ${r.xgboost}, torch ${r.torch}).
     ${r.self_contained
       ? "The run is self-contained: no file outside its own directory "
         + "contributed to a reported number."
       : "Warning: the run reports external parents, so some numbers depend on "
         + "files outside it."}
     Panel sha256 <code>${r.panel_sha256.slice(0, 16)}…</code>; the copy that
     produced this dashboard hashes identically, and all eight boosters were
     reloaded and confirmed to reproduce their recorded test average precision
     to five decimals before export. Payload exported ${m.generated}.`;

  el("proto-banner").innerHTML =
    `XGBoost on ${m.n_counties_modeled} Gulf Coast counties · test average
     precision <b>${m.flood.test_ap.toFixed(3)}</b> flood
     (${m.flood.lift_over_prevalence.toFixed(0)}× base rate),
     <b>${m.storm.test_ap.toFixed(3)}</b> hurricane
     (${m.storm.lift_over_prevalence.toFixed(0)}×) on
     ${m.splits.test[0]}–${m.splits.test[1]}, unseen in training.`;

  el("ft-prov").textContent =
    `Risk: ${m.flood.model} at ${m.horizon_days}-day lead, run ${m.run.run_id}, ` +
    `trained on station rainfall against NOAA StormEvents and HURDAT2, ` +
    `observed through ${m.observed_through}. Exported ${m.generated}. ` +
    `Warnings: api.weather.gov. Weather: Open-Meteo.`;
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

  // Warnings are issued and cancelled on no schedule, so the official side of
  // the comparison is re-pulled while a county stays open.
  setInterval(() => { if (S.fips) loadAlerts(S.fips, true); }, ALERT_TTL);
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
