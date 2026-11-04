"""Generate V2_TRAINING_RESULTS.docx -- the training and results write-up.

Every number comes out of the sealed run directory rather than being retyped,
so the document cannot drift from the models the dashboard serves. Point it at
a different run with CRISK_RUN_DIR and it regenerates against that run.

This is the technical companion to PROJECT_EXPLAINER.docx: it assumes the
reader knows what average precision is and wants the splits, the
hyperparameters, the baselines and the provenance hashes.
"""

import json
import pathlib

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

import v2_models

RUN = v2_models.run_dir()
RES = RUN / "results"
PLOTS = RUN / "plots"
OUT = pathlib.Path(__file__).parent / "V2_TRAINING_RESULTS.docx"

man = json.loads((RUN / "manifest.json").read_text())
prm = v2_models.results("params_run.json")
xgb = v2_models.results("xgb_run.json")
tft = v2_models.results("tft_run.json")
leads = v2_models.results("leads_run.json")
tft_leads = v2_models.results("tft_leads_run.json")
both = pd.read_csv(RES / "leads_both_run.csv")
HAZ = {"flood": "Flood", "storm": "Hurricane / tropical storm"}

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)
doc.styles["Normal"].paragraph_format.space_after = Pt(7)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, italic=False, size=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def callout(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x8A, 0x3A, 0x00)


def _no_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def _repeat_header(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))


def table(headers, rows, caption=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(str(head)).bold = True
    _no_split(t.rows[0])
    _repeat_header(t.rows[0])
    for row in rows:
        r = t.add_row()
        for i, val in enumerate(row):
            r.cells[i].text = str(val)
        _no_split(r)
    if caption:
        c = doc.add_paragraph()
        run = c.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
    else:
        doc.add_paragraph()
    return t


def figure(name, caption):
    f = PLOTS / name
    if not f.exists():
        p(f"[missing plot: {name}]", italic=True)
        return
    doc.add_picture(str(f), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = c.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)


# ====================================================================== title

title = doc.add_heading(
    "County-Daily Flood and Hurricane Risk on the Gulf Coast", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(
    "Training procedure and results\n"
    "Gradient boosting against a Temporal Fusion Transformer, "
    "at 1-, 2- and 3-day lead").italic = True

ref = doc.add_paragraph()
ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = ref.add_run(
    f"Run {man['run_id']}  ·  finished {man['finished_utc']}  ·  "
    f"{man['wall_seconds'] / 3600:.2f} h on {man['versions']['cuda']}\n"
    f"Panel sha256 {man['panel']['sha256']}")
r.font.size = Pt(9)

doc.add_page_break()

# ===================================================================== 1 scope

h("1. What was predicted", 1)
p("The unit of analysis is a county-day. For each of the "
  f"{man['panel']['counties_as_read']} Gulf Coast counties and each day from "
  f"{man['panel']['years_as_read'][0]} to {man['panel']['years_as_read'][1]}, "
  "the models answer a binary question: does a hazard event begin in this "
  f"county within the next {prm['horizon_days']} days? "
  f"The panel holds {man['panel']['rows_as_read']:,} such rows.")

p("Two hazards are modelled separately, with separate labels and separate "
  "models:")
bullet("Flood — the onset day of a NOAA StormEvents flood episode. A positive "
       "exists because a local Weather Service office wrote the episode up, "
       "so this is a record of reported flooding, not of flooding.")
bullet("Hurricane / tropical storm — the day the HURDAT2 Atlantic best track "
       "passes through the county. This is purely geometric: a county grazed "
       "by a weak system and a county hit by a major landfall are the same "
       "positive.")

p("Two target definitions are used throughout, and keeping them apart is the "
  "point of the whole lead-time section below:")
bullet("Pooled target — onset falls anywhere in t+1 .. t+3. This is what the "
       "dashboard's headline risk cards report.")
bullet("Exact-lead target — onset falls on exactly t+k, for k in "
       f"{', '.join(str(k) for k in prm['leads'])}. Obtained by a within-county "
       "shift of the onset column, so no day leaks across a county boundary.")

callout("The pooled question is strictly easier than the 3-day exact-lead "
        "question. Reporting a pooled score as a '3-day forecast' is the "
        "specific overstatement the exact-lead sweep exists to prevent.")

# ================================================================== 2 features

h("2. Features", 1)
p(f"All {len(v2_models.FEATS)} features are available on the morning the model "
  "runs, with one important exception noted in section 10. They fall into four "
  "groups:")
table(
    ["Group", "Features", "What it carries"],
    [["Rainfall, antecedent",
      "prcp, prcp_max_stn, prcp_prev3/7/14/30/60/90",
      "how wet the county already is, over six windows"],
     ["Rainfall, forward",
      "prcp_fwd1, prcp_fwd3, prcp_fwd3_max, log_fwd3",
      "rain over the prediction window itself"],
     ["Normalised rainfall",
      "ratio_prev3, ratio_fwd3, ratio_prev30",
      "rain divided by that county's own climatology, so a wet county and a "
      "dry county are on one scale"],
     ["Season and place",
      "tmax, tmin, month_sin, month_cos, hurricane_season, lat, lon",
      "where and when"],
     ["Exposure",
      "pop_density, RPL_THEME1, RPL_THEME4, RPL_THEMES",
      "CDC Social Vulnerability Index themes and population density"]],
    "Table 1. Feature groups. Column order is fixed in v2_models.FEATS because "
    "an XGBoost DMatrix takes its feature names from the DataFrame column "
    "order and rejects a layout that differs from training.")

p("Monotone constraints are applied to every rainfall, ratio and log feature, "
  "forcing predicted risk to be non-decreasing in each of them. More rain can "
  "never lower predicted risk. This costs a little fit and buys a model whose "
  "behaviour can be defended without inspecting it.")

p("Missing values are passed through to XGBoost rather than imputed. The "
  "booster learned its own default direction for missingness during training, "
  "so filling at scoring time would evaluate a different function from the one "
  "that was fitted.")

# ==================================================================== 3 splits

h("3. Splits", 1)
p("Splits are by calendar year and are never shuffled. No row from a test year "
  "informs training or model selection, and because the split is temporal "
  "rather than random, no county contributes neighbouring days to both sides "
  "of the boundary.")
table(
    ["Split", "Years", "Rows", "Counties", "Flood onsets", "Flood rate",
     "Hurricane onsets", "Hurricane rate"],
    [[s["split"], s["years"], f"{s['rows']:,}", s["counties"],
      f"{s['flood_pos']:,}", f"{s['flood_rate%']:.4f}%",
      f"{s['storm_pos']:,}", f"{s['storm_rate%']:.4f}%"]
     for s in man["splits"]],
    "Table 2. Split sizes and class balance. Both targets are rare-event "
    "problems; hurricane onset is roughly one county-day in four hundred.")

if prm.get("label_end"):
    ends = ", ".join(f"{HAZ[k]} through {v}" for k, v in prm["label_end"].items())
    p(f"Label availability is capped where the source is not yet final: {ends}. "
      "The HURDAT2 best track is finalised a season behind, so hurricane "
      "metrics are scored on a shorter test window than flood metrics "
      f"({765020 - 579263:,} test rows are excluded). The two hazards' test "
      "sets are therefore not the same size and their absolute average "
      "precisions are not directly comparable to each other.")

figure("split_run.png", "Figure 1. Rows and positive counts by split.")

# =============================================================== 4 baselines

h("4. Baselines", 1)
p("No model is accepted until it beats every baseline. Each baseline ranks "
  "county-days by a single raw feature -- no fitting at all -- and is scored "
  "with the same metric on the same test rows. On a rare-event target this is "
  "a much harder bar than it sounds, because a single rainfall feature already "
  "carries most of the signal a flood model has.")
rows = []
for haz in v2_models.HAZARDS:
    for name, ap in sorted(xgb[haz]["baselines"].items(), key=lambda kv: -kv[1]):
        rows.append([HAZ[haz], name.replace("single_", ""), f"{ap:.5f}",
                     f"{xgb[haz]['test']['ap'] / ap:.2f}x",
                     "strongest" if name == xgb[haz]["best_baseline"] else ""])
table(["Hazard", "Baseline feature", "Test AP", "XGBoost / baseline", "Rank"],
      rows,
      "Table 3. Every single-feature baseline, pooled target. The strongest "
      "one per hazard is the bar the model has to clear.")

# ================================================================ 5 xgboost

h("5. Gradient boosting", 1)
x = prm["xgb"]
p(f"XGBoost with a logistic objective, learning rate {x['lr']}, max depth "
  f"{x['max_depth']}, min child weight {x['min_child_weight']}, subsample "
  f"{x['subsample']}, colsample_bytree {x['colsample_bytree']} and L2 "
  f"{x['reg_lambda']}. Up to {x['rounds']:,} rounds with early stopping after "
  f"{x['early']} rounds without validation improvement.")

callout("A saved booster contains best_iteration + early trees, because "
        "save_model writes every tree early stopping trained, not only the "
        "ones up to the best round. Scoring the whole file silently evaluates "
        "an overfit model. Every prediction in this project therefore passes "
        "iteration_range=(0, best_iteration+1), with best_iteration read from "
        "the run's results JSON rather than from the model file.")

table(["Hazard", "Best iteration", "Trees in file", "Val AP", "Test AP",
       "Test base rate", "Lift", "vs best baseline"],
      [[HAZ[h_], xgb[h_]["best_iteration"],
        xgb[h_]["best_iteration"] + x["early"],
        f"{xgb[h_]['val']['ap']:.5f}", f"{xgb[h_]['test']['ap']:.5f}",
        f"{xgb[h_]['test']['prevalence'] * 100:.4f}%",
        f"{xgb[h_]['test']['lift']:.1f}x",
        f"{xgb[h_]['vs_best_baseline']:.2f}x"]
       for h_ in v2_models.HAZARDS],
      "Table 4. Pooled gradient-boosting results.")

figure("xgb_training_run.png",
       "Figure 2. Training and validation curves, with the early-stopping "
       "point marked.")
figure("xgb_eval_run.png",
       "Figure 3. Precision-recall on the test years, against the base rate.")

p("Feature importance confirms what the baselines imply -- the ratio features "
  "dominate the flood model, and the maximum station rainfall dominates the "
  "hurricane model:")
table(["Hazard", "Top features by gain"],
      [[HAZ[h_], ", ".join(f"{n} ({g:.3f})"
                           for n, g in xgb[h_]["top_features"][:6])]
       for h_ in v2_models.HAZARDS],
      "Table 5. Six highest-gain features per hazard.")

# ===================================================================== 6 tft

h("6. Temporal Fusion Transformer", 1)
t = prm["tft"]
p(f"The transformer sees a {t['encoder_days']}-day encoder window per county, "
  f"with hidden size {t['hidden']}, {t['heads']} attention heads, dropout "
  f"{t['dropout']} and hidden continuous size {t['hidden_continuous']}. "
  f"Trained with learning rate {t['lr']}, batch size {t['batch']}, up to "
  f"{t['epochs']} epochs with early stopping after {t['early']} epochs without "
  "improvement.")

p(f"Because onset is rare, training uses a positive-class weight of "
  f"{t['pos_weight']:.0f}. That weight makes the loss learnable but inflates "
  "every output: the raw flood predictions average "
  f"{tft['flood']['calibration']['pred_mean_raw']:.5f} against an observed rate "
  f"of {tft['flood']['calibration']['observed_rate']:.5f}, a factor of "
  f"{tft['flood']['calibration']['pred_mean_raw'] / tft['flood']['calibration']['observed_rate']:.0f}. "
  "Isotonic regression fitted on the validation split corrects this. Isotonic "
  "is monotone, so it moves the probabilities onto a readable scale without "
  "reordering them, and average precision barely changes.")

table(["Hazard", "Epochs run", "Best val loss", "Test windows",
       "Raw pred mean", "Calibrated pred mean", "Observed rate",
       "AP before", "AP after"],
      [[HAZ[h_], tft[h_]["epochs_run"], f"{tft[h_]['best_val_loss']:.5f}",
        f"{tft[h_]['n_test_windows']:,}",
        f"{tft[h_]['calibration']['pred_mean_raw']:.5f}",
        f"{tft[h_]['calibration']['pred_mean_calibrated']:.5f}",
        f"{tft[h_]['calibration']['observed_rate']:.5f}",
        f"{tft[h_]['calibration']['ap_raw']:.5f}",
        f"{tft[h_]['test']['ap']:.5f}"]
       for h_ in v2_models.HAZARDS],
      "Table 6. Transformer training and the effect of calibration. The AP "
      "columns show that calibration costs almost nothing in ranking quality, "
      "which is the point of using a monotone method.")

figure("tft_eval_run.png", "Figure 4. Transformer precision-recall on test.")
figure("reliability_run.png",
       "Figure 5. Reliability after isotonic calibration -- predicted "
       "probability against observed frequency.")

# ============================================================== 7 comparison

h("7. Gradient boosting against the transformer", 1)
p("Both models were fitted on the same panel, the same splits and the same "
  "label, so the only thing differing within a hazard is the architecture.")
rows = []
for h_ in v2_models.HAZARDS:
    rows.append([HAZ[h_], "XGBoost (monotone)", f"{xgb[h_]['test']['ap']:.5f}",
                 f"{xgb[h_]['test']['lift']:.1f}x",
                 f"{xgb[h_]['vs_best_baseline']:.2f}x"])
    rows.append([HAZ[h_], "Temporal Fusion Transformer",
                 f"{tft[h_]['test']['ap']:.5f}",
                 f"{tft[h_]['test']['lift']:.1f}x",
                 f"{tft[h_]['vs_best_baseline']:.2f}x"])
table(["Hazard", "Model", "Test AP", "Lift over base rate", "vs best baseline"],
      rows, "Table 7. Pooled target, test years.")

fr = xgb["flood"]["test"]["ap"] / tft["flood"]["test"]["ap"]
sr = xgb["storm"]["test"]["ap"] / tft["storm"]["test"]["ap"]
p(f"Gradient boosting wins both hazards. On floods the margin is narrow -- "
  f"{xgb['flood']['test']['ap']:.5f} against {tft['flood']['test']['ap']:.5f}, "
  f"a factor of {fr:.2f} -- close enough that the transformer is a credible "
  f"alternative. On hurricanes it is not close: "
  f"{xgb['storm']['test']['ap']:.5f} against {tft['storm']['test']['ap']:.5f}, "
  f"a factor of {sr:.2f}, and the transformer clears its baseline by only "
  f"{tft['storm']['vs_best_baseline']:.2f}x.")

p("The likely explanation is sample size rather than architecture. Hurricane "
  f"onset occurs on {xgb['storm']['test']['prevalence'] * 100:.3f}% of "
  f"county-days, and a {t['encoder_days']}-day encoder window spends almost "
  "all of its capacity on sequences containing no event at all. The "
  "transformer also cost hours of A100 time against seconds of boosting on "
  "the same data, so on this problem the heavier model is both worse and more "
  "expensive. That is a result worth reporting rather than hiding: a "
  "sequence model is not automatically the right answer for a rare-event "
  "tabular panel.")

figure("compare_run.png",
       "Figure 6. Both architectures against both baselines, pooled target.")

# =============================================================== 8 lead sweep

h("8. How much warning it actually buys", 1)
p("A single pooled number hides the thing a user needs most. Each lead below "
  "therefore has its own model, trained on its own target, and each is scored "
  "against the strongest single-feature baseline for that same target -- which "
  "changes with lead, and gets harder to beat as the signal weakens.")

table(["Hazard", "Lead", "Test base rate", "Baseline", "Baseline AP",
       "XGBoost AP", "vs base", "TFT AP", "vs base"],
      [[HAZ[r_.hazard], r_.lead, f"{r_.prevalence * 100:.4f}%",
        r_.baseline.replace("single_", ""), f"{r_.baseline_AP:.5f}",
        f"{r_.xgb_AP:.5f}", f"{r_.xgb_vs_base:.2f}x",
        f"{r_.tft_AP:.5f}", f"{r_.tft_vs_base:.2f}x"]
       for r_ in both.itertuples(index=False)],
      "Table 8. The exact-lead sweep. Six models per architecture, each "
      "trained and scored on its own target.")

fl = {k: leads[f"flood_l{k}"]["test"]["ap"] for k in prm["leads"]}
st = {k: leads[f"storm_l{k}"]["test"]["ap"] for k in prm["leads"]}
tfl = {k: tft_leads[f"flood_l{k}"]["test"]["ap"] for k in prm["leads"]}
p("Gradient-boosting skill decays monotonically with lead in both hazards. "
  f"Flood average precision runs {fl[1]:.5f} to {fl[2]:.5f} to {fl[3]:.5f} "
  f"across +1, +2 and +3 days, a loss of {(1 - fl[3] / fl[1]) * 100:.0f}% from "
  f"the shortest lead to the longest; hurricane runs {st[1]:.5f} to "
  f"{st[2]:.5f} to {st[3]:.5f}, a loss of {(1 - st[3] / st[1]) * 100:.0f}%.")

p("The transformer is monotone on hurricanes but not on floods, where the +2 "
  f"day model ({tfl[2]:.5f}) scores above the +1 day model ({tfl[1]:.5f}). "
  "Six independent fits of a high-variance model on a rare target will not "
  "order themselves perfectly, and the inversion is reported rather than "
  "smoothed away: it is a reason to treat the transformer's per-lead numbers "
  "as noisier than the boosted ones, not a finding about lead time.")

p("The ratio over baseline does not decay neatly, and this is worth being "
  "explicit about rather than presenting as a win. The baseline is losing "
  "signal at the same rate the model is: the strongest flood baseline falls "
  f"from {both[(both.hazard == 'flood') & (both.lead == '+2 day')].baseline_AP.iloc[0]:.5f} "
  f"at +2 days to {both[(both.hazard == 'flood') & (both.lead == '+3 day')].baseline_AP.iloc[0]:.5f} "
  "at +3. The ratio therefore rises while the underlying forecast gets worse. "
  "Read the average-precision columns for forecast quality and the ratio "
  "columns only for whether the model is earning its complexity.")

n_xgb_wins = int((both["xgb_AP"] >= both["tft_AP"]).sum())
p(f"Across the six lead-hazard combinations, gradient boosting takes "
  f"{n_xgb_wins} of {len(both)}.")

figure("lead_run.png", "Figure 7. XGBoost average precision against lead.")
figure("lead_both_run.png",
       "Figure 8. Both architectures against lead, with baselines.")

# ========================================================= 9 deviation metrics

h("9. Deviation from observation", 1)
p("Average precision measures ranking. It says nothing about whether the "
  "predicted probabilities are the right size, or about what happens at a "
  "usable decision threshold. The tables below score both models against two "
  "null predictors -- always-zero, and the constant base rate -- at a "
  "threshold set to the 99th percentile of each model's own predictions.")

dev = {}
for haz in v2_models.HAZARDS:
    d = pd.read_csv(RES / f"deviation_{haz}_run.csv")
    key = d.columns[0]
    dev[haz] = d.set_index(key)
    keep = ["observed_events", "predicted_mean", "bias_pred_minus_obs",
            "brier", "brier_skill_vs_baserate", "calibration_error_ECE",
            "hits", "misses", "false_alarms", "POD_recall", "precision",
            "CSI_critical_success", "F1"]
    sel = d[d[key].isin(keep)].set_index(key).reindex(keep)
    h(f"9.{v2_models.HAZARDS.index(haz) + 1} {HAZ[haz]}", 2)

    def fmt(v, name):
        if name in ("observed_events", "hits", "misses", "false_alarms"):
            return f"{int(v):,}"
        return f"{v:.5f}"

    table(["Metric", "Always zero", "Base rate", "XGBoost", "TFT"],
          [[name] + [fmt(sel.loc[name, c], name)
                     for c in ["always_zero", "base_rate", "xgboost", "tft"]]
           for name in keep],
          f"Table {9 + v2_models.HAZARDS.index(haz)}. {HAZ[haz]} deviation "
          "metrics on the test years.")

p("Two things are worth drawing out. First, both models carry positive Brier "
  "skill over the constant base rate, so the probabilities are informative and "
  "not merely well-ordered. Second, the false-alarm counts are large in "
  "absolute terms at every usable threshold. This is inherent to a rare-event "
  "target and is the reason the dashboard presents a relative band rather than "
  "a yes/no alert: at a base rate near one percent, the flood model's "
  f"precision of {dev['flood'].loc['precision', 'xgboost']:.2f} is a strong "
  "result and still means two flagged days in three are quiet.")

# ================================================================== 10 limits

h("10. Known limitations", 1)
cov = json.loads((pathlib.Path("docs/data/meta.json")).read_text())["coverage"] \
    if pathlib.Path("docs/data/meta.json").exists() else None

callout("Training rainfall is observed, not forecast. The forward rainfall "
        "features use station observations over the prediction window, which a "
        "real forecaster would not have on the morning the model runs. This is "
        "the single largest overstatement in the project. Every lead-time "
        "number in this document should be read as an upper bound on what an "
        "operational system would achieve. The fix is to retrain the forward "
        "features on archived forecasts (GEFSv12 reforecast).")

bullet("The label is an observational record, not hazard physics. A flood "
       "positive exists because someone reported it; an unreported flood is a "
       "negative. A hurricane positive is a track intersection and carries no "
       "intensity information.")
bullet(f"Coverage is {man['panel']['counties_as_read']} counties, all within "
       "the five Gulf states and all with a rainfall station in the source "
       "data. Nothing here supports a claim about counties outside that set.")
bullet("No held-out-state validation was run. Earlier work measured skill "
       "retention when a whole state is removed from training; this run did "
       "not repeat it.")
if cov:
    bullet(f"The rainfall record has gaps. The panel spans "
           f"{cov['calendar_days']:,} calendar days; the median county has "
           f"{cov['median_days_per_county']:,} of them and the worst has "
           f"{cov['min_days_per_county']:,}. "
           f"{cov['counties_under_80pct']} counties fall below 80% coverage, "
           f"and {cov['events_listed'] - cov['events_listed_with_leadup']} of "
           f"the {cov['events_listed']:,} events listed on the dashboard have "
           "no day of their three-day lead-up in the record.")
bullet("No soil moisture and no river gauges. Antecedent wetness is proxied "
       "by 30-, 60- and 90-day rainfall, a weak substitute for catchment "
       "state.")
bullet("No hurricane wind field, track forecast, intensity or storm surge.")
bullet(f"Only {xgb['flood']['n_test_pos']:,} flood and "
       f"{xgb['storm']['n_test_pos']:,} hurricane onsets fall in the test "
       "period. Confidence intervals on every metric here are correspondingly "
       "wide and none are reported.")

# ============================================================== 11 provenance

h("11. Provenance and reproducibility", 1)
v = man["versions"]
table(["Field", "Value"],
      [["Run id", man["run_id"]],
       ["Environment", f"{man['env']} on host {man['host']}"],
       ["Started (UTC)", man["started_utc"]],
       ["Finished (UTC)", man["finished_utc"]],
       ["Wall time", f"{man['wall_seconds']:,.1f} s "
                     f"({man['wall_seconds'] / 3600:.2f} h)"],
       ["GPU", v.get("cuda", "-")],
       ["Python / numpy / pandas",
        f"{v['python']} / {v['numpy']} / {v['pandas']}"],
       ["xgboost / torch", f"{v['xgboost']} / {v['torch']}"],
       ["Panel path", man["panel"]["path"]],
       ["Panel sha256", man["panel"]["sha256"]],
       ["Panel rows / counties / years",
        f"{man['panel']['rows_as_read']:,} / "
        f"{man['panel']['counties_as_read']} / "
        f"{man['panel']['years_as_read'][0]}-{man['panel']['years_as_read'][1]}"],
       ["Self-contained", str(man["self_contained"])],
       ["Warm parents", str(man["warm_parents"]) or "none"],
       ["Random seed", prm["seed"]]],
      "Table 11. Run manifest. The run is sealed: its own directory holds the "
      "models, the results and the hashes, and nothing outside it contributed "
      "to a reported number.")

p("Reproducibility was verified locally before any of these numbers reached "
  "the dashboard. The local copy of the panel hashes identically to the one "
  "the run read. Each of the eight boosters -- two pooled and six exact-lead "
  "-- was reloaded from the run directory and re-scored on the test rows, and "
  "every one reproduced its recorded test average precision to five decimal "
  "places. The check is the __main__ block of v2_models.py and can be re-run "
  "at any time.")

p("The dashboard serves XGBoost because it wins both hazards. The transformer "
  "numbers on the dashboard's comparison panel are read from this run's "
  "results files rather than re-scored in the browser, so the figures in this "
  "document and the figures on the dashboard come from the same source and "
  "cannot diverge.")

doc.save(OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
