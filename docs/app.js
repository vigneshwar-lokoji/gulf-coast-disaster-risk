const FEATURE_LABEL = {
  ratio: "72h rain ÷ county threshold",
  log_fwd3: "72h rainfall total (log)",
  max1: "largest single-day total",
  prev7: "rainfall, prior 7 days",
  prev30: "rainfall, prior 30 days",
  month_sin: "season (sin)",
  month_cos: "season (cos)",
  other_extent: "counties exceeding elsewhere, same day",
  gulf_extent: "Gulf counties exceeding, same day",
};

// One ramp per theme. The dark ramp starts deep so it reads against a dark
// panel; the light ramp starts pale so it reads against white. Both run
// low -> high in the same hue order, so the legend means the same thing.
const RAMPS = {
  dark: [[45, 74, 107], [77, 163, 255], [255, 209, 102], [255, 112, 67]],
  light: [[168, 201, 232], [63, 134, 204], [239, 154, 36], [199, 48, 27]],
};
let STOPS = RAMPS.dark;

const fmtPct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const el = (id) => document.getElementById(id);

let M, R, G, DAYS, DECL_DAYS, PREV, N_YEARS, ALL_P, TOTAL_DECL;

function colour(p, lo, hi) {
  const t = Math.max(0, Math.min(1, (p - lo) / (hi - lo))) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const c = STOPS[i].map((a, k) => Math.round(a + f * (STOPS[i + 1][k] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ------------------------------------------------------------------- map
// Real county boundaries, pre-projected to SVG paths by stage4_geo.py (Albers
// equal-area). The base layers are built once; each day only repaints the fill
// of the counties that are actually in play.

const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

/** Builds the base layers once and returns a draw() that only repaints fills.
 *  Items are {fips, p, outline, title}. */
function createMap(svgId, legendId, outlineLabel) {
  const svg = el(svgId);
  svg.setAttribute("viewBox", `0 0 ${G.width} ${G.height}`);
  svg.textContent = "";

  // Lower-48 outlines, so the Gulf reads as part of the country.
  const backdrop = svgEl("g", { class: "m-backdrop" });
  for (const d of G.states) backdrop.appendChild(svgEl("path", { d }));
  svg.appendChild(backdrop);

  const reg = new Map();
  const land = svgEl("g", { class: "m-land" });
  for (const [fips, d] of Object.entries(G.counties)) {
    const c = M.counties[fips];
    const p = svgEl("path", { d, class: c ? "in" : "out" });
    const t = svgEl("title", {});
    t.textContent = c
      ? `${c.name}, ${c.state} — in scope, threshold ${c.threshold_mm.toFixed(0)} mm`
      : "outside the modelled set";
    p.appendChild(t);
    if (c) reg.set(fips, { path: p, title: t });
    land.appendChild(p);
  }
  svg.appendChild(land);

  const borders = svgEl("g", { class: "m-borders" });
  for (const d of G.gulf_states) borders.appendChild(svgEl("path", { d }));
  svg.appendChild(borders);

  const outlines = svgEl("g", { class: "m-outlines" });
  svg.appendChild(outlines);

  el(legendId).innerHTML =
    `<span>${fmtPct(0.02)}</span><span class="bar"></span>` +
    `<span>${fmtPct(M.max_observed_probability)}</span>` +
    `<span class="key"><i class="in"></i>in scope, below threshold</span>` +
    `<span class="key"><i class="out"></i>not modelled</span>` +
    `<span class="key"><i class="ring"></i>${outlineLabel}</span>`;

  let last = [];
  function draw(items) {
    last = items;
    const lo = 0.02, hi = M.max_observed_probability;
    for (const [fips, { path, title }] of reg) {
      path.style.fill = "";           // falls back to the themed CSS class
      const c = M.counties[fips];
      title.textContent = `${c.name}, ${c.state} — below threshold`;
    }
    outlines.textContent = "";
    for (const it of items) {
      const hit = reg.get(it.fips);
      if (!hit) continue;
      // Near-miss counties get a flat colour, never a point on the probability
      // ramp — they have no probability, only a distance to the threshold.
      hit.path.style.fill = it.near
        ? getComputedStyle(document.documentElement).getPropertyValue("--map-near")
        : colour(it.p, lo, hi);
      hit.title.textContent = it.title;
      if (it.outline) outlines.appendChild(svgEl("path", { d: G.counties[it.fips] }));
    }
  }
  draw.repaint = () => draw(last);     // after a theme change
  return draw;
}

// ----------------------------------------------------------------- theme

function applyTheme(name) {
  document.documentElement.setAttribute("data-theme", name);
  STOPS = RAMPS[name];
  try { localStorage.setItem("theme", name); } catch (e) { /* private mode */ }
  document.querySelectorAll("[data-set-theme]").forEach((b) =>
    b.classList.toggle("on", b.dataset.setTheme === name));
  // Probability fills are inline styles, so they do not follow the stylesheet.
  if (drawReplay) drawReplay.repaint();
  if (drawLive) drawLive.repaint();
}

function wireTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* private mode */ }
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(saved || (prefersLight ? "light" : "dark"));
  document.querySelectorAll("[data-set-theme]").forEach((b) =>
    b.addEventListener("click", () => applyTheme(b.dataset.setTheme)));
}

let drawReplay, drawLive;

// ---------------------------------------------------------------- replay

function showDay(i) {
  const day = DAYS[i];
  const rows = R.days[day];
  el("day-slider").value = i;
  el("day-label").textContent = day;
  const nd = rows.filter((r) => r[2]).length;
  el("day-meta").textContent =
    `${rows.length} count${rows.length === 1 ? "y" : "ies"} over threshold` +
    ` · ${nd} declared within 14 days` +
    ` · day ${i + 1} of ${DAYS.length}`;

  const tb = el("day-table").querySelector("tbody");
  tb.innerHTML = rows
    .map((r, k) => {
      const c = M.counties[r[0]] || { name: r[0], state: "?" };
      return `<tr>
        <td class="num">${k + 1}</td>
        <td>${c.name}, ${c.state}</td>
        <td class="num">${fmtPct(r[1])}</td>
        <td class="num">${r[3].toFixed(2)}&times;</td>
        <td class="num">${r[4].toFixed(0)} mm</td>
        <td><span class="pill ${r[2] ? "yes" : "no"}">${
          r[2] ? "declared" : "no declaration"
        }</span></td>
      </tr>`;
    })
    .join("");
  drawReplay(rows.map((r) => {
    const c = M.counties[r[0]] || { name: r[0], state: "?" };
    return {
      fips: r[0], p: r[1], outline: !!r[2],
      title: `${c.name}, ${c.state} — ${fmtPct(r[1])} · ${r[4].toFixed(0)} mm, ` +
             `${r[3].toFixed(2)}× threshold · ` +
             (r[2] ? "DECLARED" : "no declaration"),
    };
  }));
}

function wireReplay() {
  const s = el("day-slider");
  s.max = DAYS.length - 1;
  s.addEventListener("input", () => showDay(+s.value));
  el("prev-day").addEventListener("click", () =>
    showDay(Math.max(0, +s.value - 1)));
  el("next-day").addEventListener("click", () =>
    showDay(Math.min(DAYS.length - 1, +s.value + 1)));
  el("next-decl").addEventListener("click", () => {
    const cur = +s.value;
    const nxt = DECL_DAYS.find((i) => i > cur);
    showDay(nxt === undefined ? DECL_DAYS[0] : nxt);
  });
  el("rp-period").textContent = R.period;
  showDay(DECL_DAYS[0]);
}

// ------------------------------------------------------------------ live

let LIVE = null;

let LV_DAYS = [];

function showLiveDay(day) {
  const rows = LIVE.days[day] || [];
  const near = LIVE.near[day] || [];
  el("lv-label").textContent = day;
  el("lv-date").value = day;

  const flagged = rows.filter((r) => r[1] >= LIVE.threshold).length;
  el("lv-meta").textContent = rows.length
    ? `${rows.length} of ${LIVE.n_counties} counties over threshold · ` +
      `${flagged} clear the ${fmtPct(LIVE.threshold, 0)} operating point`
    : `0 of ${LIVE.n_counties} counties over threshold`;

  el("lv-table").hidden = !rows.length;
  el("lv-table").querySelector("tbody").innerHTML = rows.map((r, k) => {
    const c = M.counties[r[0]];
    return `<tr>
      <td class="num">${k + 1}</td>
      <td>${c.name}, ${c.state}</td>
      <td class="num">${fmtPct(r[1])}</td>
      <td class="num">${r[2].toFixed(0)} mm</td>
      <td class="num">${r[3].toFixed(2)}&times;</td>
      <td class="num">${r[4].toFixed(0)} mm</td>
      <td class="num">${fmtPct(r[5], 0)}</td>
    </tr>`;
  }).join("");

  // A quiet day is a result, not a blank screen. Show how far the nearest
  // counties actually were, so silence is legible as a measurement.
  el("lv-near").innerHTML = rows.length || !near.length ? "" :
    `<p class="quiet"><b>No county reached its rainfall threshold.</b>
       This is not a forecast of no rain — the model is conditional, and it has
       nothing to say about a county until that county exceeds its own
       1-in-100-day three-day total. Below that line there is no probability to
       report, only a distance.</p>
     <table class="nearlist">
       <thead><tr><th>Closest counties</th><th class="num">72h rain</th>
         <th class="num">Its threshold</th><th class="num">Reached</th></tr></thead>
       <tbody>${near.map((n) => {
         const c = M.counties[n[0]];
         return `<tr><td>${c.name}, ${c.state}</td>
           <td class="num">${n[1].toFixed(0)} mm</td>
           <td class="num">${c.threshold_mm.toFixed(0)} mm</td>
           <td class="num">${fmtPct(n[2], 0)}</td></tr>`;
       }).join("")}</tbody>
     </table>`;

  el("lv-note").innerHTML = rows.length
    ? `<b>Observed</b> is the share of the 30-day antecedent window that came from
       published station data rather than forecast.
       ${rows.every((r) => r[6])
         ? "Forward rainfall on this day is observed, not forecast."
         : "Forward rainfall is forecast by construction."}`
    : "";

  const items = rows.length
    ? rows.map((r) => {
        const c = M.counties[r[0]];
        return {
          fips: r[0], p: r[1], outline: r[1] >= LIVE.threshold,
          title: `${c.name}, ${c.state} — ${fmtPct(r[1])} · ${r[2].toFixed(0)} mm, ` +
                 `${r[3].toFixed(2)}× threshold · ${fmtPct(r[5], 0)} observed`,
        };
      })
    : near.map((n) => {
        const c = M.counties[n[0]];
        return {
          fips: n[0], near: true, outline: false,
          title: `${c.name}, ${c.state} — ${n[1].toFixed(0)} mm, ` +
                 `${fmtPct(n[2], 0)} of its ${c.threshold_mm.toFixed(0)} mm ` +
                 `threshold · below threshold, no probability`,
        };
      });
  drawLive(items);
}

function step(delta) {
  const i = LV_DAYS.indexOf(el("lv-date").value);
  const j = Math.max(0, Math.min(LV_DAYS.length - 1, (i < 0 ? 0 : i) + delta));
  showLiveDay(LV_DAYS[j]);
}

function wireLive() {
  LV_DAYS = Object.keys(LIVE.days).sort();
  const input = el("lv-date");
  input.min = LV_DAYS[0];
  input.max = LV_DAYS[LV_DAYS.length - 1];

  input.addEventListener("change", () => {
    if (LIVE.days[input.value]) return showLiveDay(input.value);
    // Snap to the nearest day we actually hold rather than showing nothing.
    const t = Date.parse(input.value);
    const near = LV_DAYS.reduce((a, b) =>
      Math.abs(Date.parse(b) - t) < Math.abs(Date.parse(a) - t) ? b : a);
    showLiveDay(near);
  });
  el("lv-prev").addEventListener("click", () => step(-1));
  el("lv-next").addEventListener("click", () => step(1));
  el("lv-active").addEventListener("click", () => {
    const i = LV_DAYS.indexOf(el("lv-date").value);
    for (let k = 1; k <= LV_DAYS.length; k++) {
      const d = LV_DAYS[(i + k) % LV_DAYS.length];
      if ((LIVE.days[d] || []).length) return showLiveDay(d);
    }
  });

  const active = LV_DAYS.filter((d) => LIVE.days[d].length).length;
  el("lv-range").innerHTML =
    `Selectable range <b>${LV_DAYS[0]}</b> to <b>${LV_DAYS[LV_DAYS.length - 1]}</b>
     — ${LV_DAYS.length} days, ${active} with at least one county over threshold.
     The forward end stops at ${LIVE.forecast_through}, where the NWS forecast
     runs out; there is no rainfall input past it, so there is no prediction.`;

  const stale = Math.round((Date.now() - Date.parse(LIVE.generated)) / 36e5);
  el("lv-provenance").textContent =
    `Observed rainfall published through ${LIVE.observed_through}; everything ` +
    `after that is forecast. Generated ${stale} h ago by stage4_live.py — this ` +
    `is a snapshot, not a live call from your browser.`;

  // Open on something worth looking at rather than on day one of the window.
  const best = LV_DAYS.reduce((a, b) =>
    (LIVE.days[b].length ? LIVE.days[b][0][1] : 0) >
    (LIVE.days[a].length ? LIVE.days[a][0][1] : 0) ? b : a);
  showLiveDay(best);
}

function liveUnavailable() {
  el("live").innerHTML =
    `<div class="panel-head"><h2>Beyond the training data</h2></div>
     <p class="warn">No live snapshot found. Generate one with
     <code>python stage4_live.py</code>.</p>`;
}

// -------------------------------------------------------------- explorer

function stats(thr) {
  let flagged = 0, hits = 0;
  for (const p of ALL_P) if (p >= thr) flagged++;
  for (let i = 0; i < ALL_P.length; i++) if (ALL_P[i] >= thr && ALL_Y[i]) hits++;
  const precision = flagged ? hits / flagged : NaN;
  return {
    thr, flagged, hits,
    perYear: flagged / N_YEARS,
    precision,
    recall: hits / TOTAL_DECL,
    lift: precision / PREV,
  };
}

let ALL_Y;

function renderExplorer() {
  const thr = +el("thr-slider").value / 100;
  el("thr-out").textContent = fmtPct(thr, 1);
  const s = stats(thr);

  el("ex-cards").innerHTML = [
    ["Counties flagged / year", s.perYear.toFixed(0),
      `${s.flagged} rows over ${N_YEARS} years`],
    ["Precision", isNaN(s.precision) ? "—" : fmtPct(s.precision),
      `${s.hits} of ${s.flagged} flagged were declared`],
    ["Recall", fmtPct(s.recall),
      `${s.hits} of ${TOTAL_DECL} declarations caught`],
    ["Lift vs base rate", isNaN(s.lift) ? "—" : s.lift.toFixed(2) + "×",
      `base rate ${fmtPct(PREV, 2)}`],
  ].map(([k, v, n]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  const grid = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14];
  const near = grid.reduce((a, b) =>
    Math.abs(b - thr) < Math.abs(a - thr) ? b : a);
  el("ex-table").querySelector("tbody").innerHTML = grid.map((t) => {
    const r = stats(t);
    return `<tr class="${t === near ? "here" : ""}">
      <td class="num">${fmtPct(t, 0)}</td>
      <td class="num">${r.perYear.toFixed(0)}</td>
      <td class="num">${isNaN(r.precision) ? "—" : fmtPct(r.precision)}</td>
      <td class="num">${fmtPct(r.recall)}</td>
      <td class="num">${isNaN(r.lift) ? "—" : r.lift.toFixed(2) + "×"}</td>
    </tr>`;
  }).join("");

  el("ex-rows").textContent = R.n_rows.toLocaleString();
  el("ex-prev").textContent = fmtPct(PREV, 2);
}

// ------------------------------------------------------------ model card

function renderModelCard(bytes) {
  const op = M.operating_points.find((o) => o.threshold === 0.10)
    || M.operating_points[0];
  el("mc-cards").innerHTML = [
    ["Average precision", M.test_ap.toFixed(4),
      `${(M.test_ap / M.test_prevalence).toFixed(2)}× the base rate`],
    ["Highest probability it can output", fmtPct(M.max_observed_probability),
      "over the whole test period"],
    ["Trained on", M.trained_on, "ranking model"],
    ["Calibrated on", M.calibrated_on, "Platt scaling, held out"],
    ["Evaluated on", M.evaluated_on, "never seen in training"],
    ["Counties in scope", Object.keys(M.counties).length,
      "AL, FL, LA, MS, TX only"],
  ].map(([k, v, n]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${n}</div></div>`).join("");

  const pooled = M.test_ap / M.test_prevalence;
  el("mc-states").querySelector("tbody").innerHTML =
    Object.entries(M.state_transfer_lift)
      .sort((a, b) => b[1] - a[1])
      .map(([st, lift]) => {
        const read = lift >= 1.5 ? "transfers" :
          lift >= 1.2 ? "weaker than pooled" : "does not transfer";
        return `<tr><td>${st}</td><td class="num">${lift.toFixed(2)}×</td>
          <td><span class="pill ${lift >= 1.5 ? "no" : "yes"}">${read}</span></td></tr>`;
      }).join("");

  const order = M.coef
    .map((c, i) => [M.features[i], c])
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  el("mc-coefs").querySelector("tbody").innerHTML = order.map(([f, c]) =>
    `<tr><td>${FEATURE_LABEL[f] || f}</td>
      <td class="num">${c >= 0 ? "+" : ""}${c.toFixed(3)}</td>
      <td style="color:${c >= 0 ? "var(--hot)" : "var(--accent)"}">${
        c >= 0 ? "raises risk" : "lowers risk"}</td></tr>`).join("");

  el("mc-limits").innerHTML = [
    `<b>This is a ranked priority list, not a warning system.</b> The highest
     probability the model has ever produced is ${fmtPct(M.max_observed_probability)}.
     Most flagged counties will not be declared.`,
    `<b>It answers a conditional question.</b> Rows exist only for county-days that
     already exceeded the county's own 99th-percentile 72-hour rainfall. It does not
     forecast rainfall, and it says nothing about counties below threshold.`,
    `<b>Texas is the weak spot.</b> Held out of training, Texas lift falls to
     ${M.state_transfer_lift.TX.toFixed(2)}× — close to no skill. Treat
     Texas output as unvalidated.`,
    `<b>Every within-state lift is below the pooled ${pooled.toFixed(2)}×.</b>
     Part of the pooled figure is difference in base rates between states, not
     discrimination within a state.`,
    `<b>Social vulnerability indices added nothing.</b> Neither did population or
     county area. Only rainfall severity and how widespread the storm was carried
     signal.`,
    `<b>Recommended operating point: ${fmtPct(op.threshold, 0)}</b> —
     about ${op.flagged_per_year.toFixed(0)} county-alerts a year,
     ${fmtPct(op.precision)} of which end in a declaration, catching
     ${fmtPct(op.recall)} of all declarations.`,
  ].map((s) => `<li>${s}</li>`).join("");

  el("ft-prov").textContent =
    `Ranker trained ${M.trained_on}, calibrated ${M.calibrated_on}, ` +
    `evaluated ${M.evaluated_on}.`;
  el("ft-size").textContent = `${Math.round(bytes / 1024)} KB`;
}

// ----------------------------------------------------------------- boot

function wireTabs() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) =>
        x.classList.toggle("active", x === t));
      document.querySelectorAll(".panel").forEach((p) =>
        p.classList.toggle("hidden", p.id !== t.dataset.panel));
    });
  });
}

async function main() {
  // live.json is rewritten daily by the refresh job, so a cached copy would
  // silently show yesterday's forecast. Revalidate rather than trust the cache.
  const get = (f) => fetch(f, { cache: "no-cache" });
  const [mRes, rRes, gRes, lRes] = await Promise.all([
    get("data/model.json"), get("data/replay.json"), get("data/geo.json"),
    get("data/live.json").catch(() => null),
  ]);
  const mText = await mRes.text();
  M = JSON.parse(mText);
  R = await rRes.json();
  G = await gRes.json();
  LIVE = lRes && lRes.ok ? await lRes.json() : null;

  DAYS = Object.keys(R.days).sort();
  DECL_DAYS = DAYS.map((d, i) => (R.days[d].some((r) => r[2]) ? i : -1))
    .filter((i) => i >= 0);
  N_YEARS = new Set(DAYS.map((d) => d.slice(0, 4))).size;
  ALL_P = [];
  ALL_Y = [];
  for (const d of DAYS) for (const r of R.days[d]) { ALL_P.push(r[1]); ALL_Y.push(r[2]); }
  TOTAL_DECL = R.n_declared;
  PREV = TOTAL_DECL / R.n_rows;

  el("headline-caveat").textContent =
    `The model tops out at ${fmtPct(M.max_observed_probability)}. It is a ranked ` +
    `priority list for ${Object.keys(M.counties).length} Gulf Coast counties, not a ` +
    `warning system — and Texas output is unvalidated.`;

  drawReplay = createMap("map", "legend", "declared");
  wireTabs();
  wireReplay();
  if (LIVE) {
    drawLive = createMap("lv-map", "lv-legend",
      `flagged at ${fmtPct(LIVE.threshold, 0)}`);
    wireLive();
  } else {
    liveUnavailable();
  }
  el("thr-slider").addEventListener("input", renderExplorer);
  renderExplorer();
  renderModelCard(mText.length);
  wireTheme();
}

main().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<p class="warn">Failed to load model data: ${e.message}</p>`);
  throw e;
});
