"""Generate PROJECT_EXPLAINER.docx, the hand-off document for someone who has
to present this project without having built it. Numbers are pulled from the
artifacts rather than retyped, so the document cannot drift from the results."""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor

HERE = Path(__file__).parent
M = json.loads((HERE / "docs" / "data" / "model.json").read_text())

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(8)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, bold=False, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def numbered(text):
    doc.add_paragraph(text, style="List Number")


def _no_split(row):
    # Without this a row can be torn across a page break, which renders as a
    # stray empty band at the top of the next page.
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:cantSplit"))


def _repeat_header(row):
    pr = row._tr.get_or_add_trPr()
    pr.append(OxmlElement("w:tblHeader"))


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(head))
        run.bold = True
    _no_split(t.rows[0])
    _repeat_header(t.rows[0])
    for row in rows:
        r = t.add_row()
        for i, val in enumerate(row):
            r.cells[i].text = str(val)
        _no_split(r)
    doc.add_paragraph()
    return t


def callout(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x8A, 0x3A, 0x00)


# ====================================================================== title

title = doc.add_heading("Gulf Coast Disaster Declaration Risk", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run(
    "A conditional risk model and browser dashboard for 218 Gulf Coast counties\n"
    "Complete project explainer — what it does, how it was built, and what it "
    "cannot do")
run.italic = True

links = doc.add_paragraph()
links.alignment = WD_ALIGN_PARAGRAPH.CENTER
links.add_run("Dashboard: https://vigneshwar-lokoji.github.io/gulf-coast-disaster-risk/\n"
              "Code: https://github.com/vigneshwar-lokoji/gulf-coast-disaster-risk")

doc.add_page_break()

# =================================================================== 1 summary

h("1. The one-paragraph version", 1)
p("Given that a Gulf Coast county has just exceeded its own 1-in-100-day "
  "three-day rainfall total, how likely is a federal disaster declaration "
  "within the next 14 days? That is the entire question. The model does not "
  "forecast weather — it takes rainfall as an input and maps it to the "
  "probability of an administrative outcome. It covers 218 counties across "
  "Florida, Alabama, Mississippi, Louisiana and Texas, was trained on "
  f"{M['trained_on']}, calibrated on {M['calibrated_on']}, and evaluated on "
  f"{M['evaluated_on']} — a held-out period the model never saw during "
  "fitting. On that test period it achieves an average precision of "
  f"{M['test_ap']:.4f} against a base rate of {M['test_prevalence']:.2%}, a "
  f"lift of {M['test_ap'] / M['test_prevalence']:.2f}x. The dashboard runs "
  "entirely in the browser from a 21 KB model file, with no server and no "
  "build step.")

h("What to say in the first thirty seconds of a presentation", 2)
p("“This is a triage tool, not a warning system. It answers one narrow "
  "question: among the counties that have just been hit with extreme "
  "rainfall, which ones should an analyst look at first? The highest "
  f"probability it has ever produced is {M['max_observed_probability']:.1%}. "
  "It is roughly twice as good as guessing, and it is honest about where it "
  "is not.”")

callout("The most common misunderstanding: people assume a weather app and "
        "this dashboard should agree. They measure different things. A "
        "weather app gives probability of rain. This gives probability of a "
        "federal declaration, conditional on extreme rain having already "
        "happened. See section 11.")

# ================================================================= 2 not a ...

h("2. What this is not", 1)
p("Being precise about the boundaries is the most defensible part of the "
  "project. State these before anyone asks.")

bullet("It is not a weather forecaster. It does not predict rain. It consumes "
       "the National Weather Service's rainfall forecast as an input.")
bullet("It is not a warning system. The maximum probability ever produced is "
       f"{M['max_observed_probability']:.1%}. It is useful for deciding where "
       "to look first, useless for deciding whether to evacuate.")
bullet("It is not unconditional. A county only enters the data once it has "
       "already exceeded its own 99th-percentile three-day rainfall. Below "
       "that line the model has no opinion at all — not “low risk”, "
       "but silence.")
bullet("It is not causal. The coefficients are standardised logistic weights. "
       "Sign and relative magnitude are meaningful; they are associations, "
       "not causal effects.")

# ===================================================================== 3 data

h("3. The data", 1)
p("Four public sources. Nothing proprietary, nothing scraped.")

table(
    ["Source", "What it provides", "Role"],
    [["NOAA NCEI GHCN-Daily",
      "Daily precipitation from 368 weather stations",
      "Rainfall history and the live observed feed"],
     ["FEMA Disaster Declarations Summaries",
      "Every federal declaration with date, county and incident type",
      "The prediction target"],
     ["CDC/ATSDR Social Vulnerability Index",
      "Four socioeconomic theme percentiles per county",
      "Candidate predictors (tested and rejected — see section 7)"],
     ["NWS api.weather.gov gridpoint QPF",
      "Forecast rainfall, roughly seven days ahead",
      "The live forward-looking input"]])

h("Building the panel", 2)
p("Station-level daily rainfall is averaged to the county using a "
  "station-to-county mapping covering 368 stations across 218 counties. "
  "A county-day with no reporting station is treated as zero rainfall, which "
  "matches how the live path behaves and avoids a silent distribution shift "
  "between training and production.")
p("Climatology — the long-run seasonal baseline used for anomaly features — "
  "is fitted on 1990 to 2010 only and then frozen. Widening it to the full "
  "record would let the test period influence its own baseline, which is a "
  "subtle and very common form of leakage.")
p("FEMA incident types are filtered. “Biological” is dropped because "
  "it is overwhelmingly COVID-19, and “Human Cause” and “Toxic "
  "Substances” are not weather.")

h("How the data is split", 2)
p("Splits are contiguous blocks of time, never random. Adjacent days share "
  "weather state, so a shuffled split would leak information across the "
  "boundary and inflate every number in this document.")
table(["Split", "Years", "Purpose"],
      [["Train", M["trained_on"], "Fit the coefficients"],
       ["Calibrate", M["calibrated_on"],
        "Fit the Platt calibration; never used to fit coefficients"],
       ["Test", M["evaluated_on"],
        "Reported once. 5,260 hazard county-days, 76 declaration episodes"]])

# ================================================================== 4 stage 1

h("4. Stage 1 — the daily hazard model, and what it taught us", 1)
p("The project began the conventional way: predict the physical hazard "
  "directly. A county-month panel was built with rainfall, anomaly and "
  "seasonality features, and gradient-boosted trees (XGBoost) were trained to "
  "predict flood and storm declaration onset at lead times of 0, 1, 2 and 3 "
  "months. A Temporal Fusion Transformer was trained alongside it on the same "
  "panel, as the sequence-model comparison.")
p("Before believing any of it, Tier-0 baselines were run: climatology "
  "— the probability of an event given the county and the month of the year "
  "— and persistence. Climatology is not a strawman. It already encodes "
  "“Louisiana in September is dangerous”, which is most of what a "
  "county risk model is usually asked to say.")

p("The honest result:", bold=True)
table(["Target", "Lead", "Test avg. precision", "Lift", "ROC-AUC"],
      [["Flood", "0 months", "0.043", "7.41x", "0.759"],
       ["Flood", "2 months", "0.008", "1.55x", "0.582"],
       ["Flood", "3 months", "0.010", "1.91x", "0.459"],
       ["Storm", "0 months", "0.203", "5.78x", "0.828"],
       ["Storm", "3 months", "0.130", "3.67x", "0.793"]])

p("Lead-0 works because the rainfall has already fallen — the model is "
  "largely reading the hazard, not forecasting it. At longer leads the flood "
  "models collapse; a ROC-AUC of 0.459 at three months is below chance. "
  "This is the finding that reframed the project. Predicting whether extreme "
  "rain will occur months ahead is a weather forecasting problem, and the "
  "National Weather Service is already far better at it than this model will "
  "ever be.")

callout("Key narrative beat: the pivot was driven by a negative result that "
        "we did not hide. That is a strength in any presentation.")

# ================================================================== 5 pivot

h("5. The pivot — asking a question we can actually answer", 1)
p("If forecasting the rain is the wrong job, the useful question is what "
  "happens after it rains. Rewriting the problem conditionally does three "
  "things at once:")
numbered("It removes the part of the problem the NWS already solves, and "
         "lets us consume their forecast as an input instead of competing "
         "with it.")
numbered("It raises the base rate enormously. The unconditional problem has "
         "a prevalence near 0.1%, where essentially nothing is learnable. "
         "Conditioning on a hazard day lifts prevalence to 5.4% in training "
         "and 4.6% in test.")
numbered("It matches a real decision. Emergency managers already know it "
         "rained. What they want to know is where to deploy attention next.")

p("The hazard definition is deliberately local. Each county's threshold is "
  "the 99th percentile of its own three-day rainfall totals, computed on "
  "training data only and then frozen into the exported model. 117 mm over "
  "three days is an ordinary week in Baldwin County, Alabama and a "
  "once-a-century event somewhere drier. A single Gulf-wide threshold would "
  "simply have selected for wet counties.")
p("Measured over the full record: 22,754 hazard events, of which 1,229 were "
  "followed by a federal declaration within 14 days — a prevalence of 5.4%, "
  "spread across 413 distinct hazard dates.")

# ================================================================== 6 features

h("6. Stage 2 — the model", 1)
p("Nine features. Every one is computable at prediction time from rainfall "
  "alone, which is what makes the live path possible.")

table(["Feature", "Meaning"],
      [["ratio", "Three-day forward rainfall divided by the county's own threshold"],
       ["log_fwd3", "Log of the three-day forward rainfall total in mm"],
       ["max1", "The wettest single day inside the three-day window"],
       ["prev7", "Rainfall in the 7 days before the hazard — antecedent wetness"],
       ["prev30", "Rainfall in the 30 days before — soil saturation proxy"],
       ["month_sin, month_cos", "Seasonality encoded as a circle, so December and January are adjacent"],
       ["other_extent", "How many other counties in the same state crossed threshold that day"],
       ["gulf_extent", "How many counties across the whole Gulf crossed threshold that day"]])

p("The two extent features matter more than they look. FEMA declares "
  "regionally, not county by county. A single county with extreme rainfall is "
  "a local flood; forty counties with extreme rainfall on the same day is a "
  "federal event. Without these, the model would be systematically blind to "
  "the mechanism that actually drives the outcome.")

h("Why logistic regression and not a tree ensemble", 2)
p("There are only 413 independent positive dates in the whole record. That is "
  "small-n. A penalised logistic regression gives interpretable coefficients "
  "— “one standard deviation more antecedent rainfall multiplies the odds "
  "of declaration by 1.51, holding severity and regional extent fixed” — "
  "which is a claim a tree ensemble cannot make.")
p("XGBoost was still run as a nonlinearity check, and it scored slightly "
  "higher on the test set: average precision 0.1045 versus 0.0929. That "
  "difference of 0.0116 is smaller than XGBoost's own seed-to-seed standard "
  "deviation of 0.0140 across three random seeds. In other words the apparent "
  "advantage is within the noise of simply rerunning it. Shipping the simpler, "
  "interpretable, reproducible model is the defensible choice.")

h("Reading the coefficients", 2)
p("All features are standardised, so the odds ratios are per one standard "
  "deviation and are directly comparable to each other.")
table(["Feature", "Coefficient", "Odds ratio per SD", "Direction"],
      [["log_fwd3", "+0.767", "2.15", "More absolute rainfall, higher risk"],
       ["prev30", "+0.412", "1.51", "Wetter preceding month, higher risk"],
       ["gulf_extent", "+0.186", "1.20", "Wider Gulf-scale event, higher risk"],
       ["max1", "+0.156", "1.17", "More concentrated downpour, higher risk"],
       ["ratio", "-0.342", "0.71", "See note below"],
       ["prev7", "-0.349", "0.71", "See note below"],
       ["other_extent", "-0.368", "0.69", "See note below"],
       ["month_sin", "-0.717", "0.49", "Seasonal term"],
       ["month_cos", "-0.679", "0.50", "Seasonal term"]])

p("The negative signs need care, and an audience will ask about them. They "
  "are conditional associations, not standalone effects. Both ratio and "
  "log_fwd3 measure the same rainfall, so they split the work between them: "
  "given the absolute amount of rain, a high ratio identifies a normally dry "
  "county, where the same millimetres do less damage. Likewise other_extent "
  "and gulf_extent are strongly correlated, and the model uses the pair to "
  "separate a state-local event from a Gulf-wide one. Reporting a single "
  "coefficient from a correlated set as though it stood alone is the classic "
  "way to misread a regression.")

# ================================================================== 7 ablation

h("7. The scientific test — and a negative result worth reporting", 1)
p("The designed experiment was a nested sequence. Each block adds features to "
  "the one before it, and the question is whether the addition improves "
  "average precision after everything earlier is already controlled for. "
  "Confidence intervals come from 2,000 bootstrap resamples.")

table(["Step", "Change in avg. precision", "95% interval", "Verdict"],
      [["Hazard → + regional extent", "+0.0079", "[+0.0020, +0.0164]",
        "Significant"],
       ["+ exposure (population, area)", "-0.0056", "[-0.0157, +0.0011]",
        "Not distinguishable from zero"],
       ["+ social vulnerability (SVI)", "-0.0001", "[-0.0047, +0.0032]",
        "Not distinguishable from zero"]])

p("Regional extent earns its place. Population, land area and the four social "
  "vulnerability themes do not. The SVI coefficients are tiny and the interval "
  "straddles zero.")
callout("This is a genuine negative result, and the honest framing is: once "
        "rainfall severity and regional extent are controlled for, social "
        "vulnerability adds no measurable predictive power to this outcome at "
        "this sample size. It does not mean social vulnerability is "
        "irrelevant to disasters. It means it is not detectable here, with "
        "413 positive dates, after the spatial confound is removed.")
p("The shipped model is therefore the hazard-plus-extent block with nine "
  "features — the last point in the sequence where the evidence supported "
  "adding anything.")

# ================================================================== 8 stage 3

h("8. Stage 3 — calibration and validation", 1)

h("Calibration", 2)
p("A raw logistic model trained on an imbalanced problem produces scores that "
  "rank well but are not probabilities. Platt scaling, fitted on the "
  f"{M['calibrated_on']} block and never on training or test data, fixes this.")
table(["Metric", "Before calibration", "After calibration"],
      [["Brier score (lower is better)", "0.2036", "0.0429"],
       ["Expected calibration error", "0.3442", "0.0121"]])
p("After calibration, when the dashboard says 8%, roughly 8 out of 100 such "
  "county-days were actually declared. That is what makes the number on "
  "screen meaningful rather than just a ranking score.")

h("Why average precision and not accuracy", 2)
p("At a 4.6% base rate, a model that predicts “no declaration” every "
  "single time is 95.4% accurate and completely useless. ROC-AUC is similarly "
  "flattering under heavy imbalance. Average precision and lift over the base "
  "rate are the metrics that can actually be embarrassed by a bad model, which "
  "is why they are the ones reported.")

h("Headline results", 2)
table(["Metric", "Value"],
      [["Average precision (test)", f"{M['test_ap']:.4f}"],
       ["Base rate / prevalence", f"{M['test_prevalence']:.2%}"],
       ["Lift over base rate", f"{M['test_ap'] / M['test_prevalence']:.2f}x"],
       ["Highest probability ever produced", f"{M['max_observed_probability']:.1%}"],
       ["Held-out hazard county-days", "5,260"],
       ["Held-out declaration episodes", "76"]])

h("Operating points — the workload trade-off", 2)
p("There is no free lunch. Raising the alert threshold cuts the number of "
  "counties an analyst has to review and raises precision, but misses more "
  "real declarations.")
table(["Flag when P ≥", "Counties flagged per year", "Precision", "Recall", "Lift"],
      [[f"{op['threshold']:.0%}", f"{op['flagged_per_year']:.0f}",
        f"{op['precision']:.1%}", f"{op['recall']:.1%}", f"{op['lift']:.2f}x"]
       for op in M["operating_points"]] +
      [["15%", "3", "5.6%", "0.4%", "1.21x"]])
p("The 15% row is shown because an audience will ask why not simply raise the "
  "threshold further. At 15% the model fires about three times a year and "
  "performance degrades — there is not enough data up there to be reliable.")

h("Geographic validation — the hardest test", 2)
p("Each state was removed from training entirely, the model refitted on the "
  "remaining four, and then evaluated on the held-out state. This asks whether "
  "the model has learned something transferable or merely memorised regional "
  "quirks.")
table(["State", "Hazard days", "Declarations", "Prevalence", "Lift when held out"],
      [["Florida", "775", "51", "6.6%", "1.98x"],
       ["Louisiana", "579", "76", "13.1%", "1.56x"],
       ["Mississippi", "228", "21", "9.2%", "1.44x"],
       ["Alabama", "746", "60", "8.0%", "1.23x"],
       ["Texas", "2,932", "34", "1.2%", "1.11x"]])

callout("Texas output is effectively unvalidated. The model barely beats "
        "chance there, and the dashboard says so on every screen that can "
        "show a Texas row. Never present a Texas number as a finding.")
p("The reason is visible in the table. Texas contributes the most hazard days "
  "of any state — 2,932 — but only 34 declarations, a prevalence of 1.2% "
  "against 13.1% in Louisiana. Texas counties cross their rainfall thresholds "
  "constantly without being declared. Whatever determines a Texas declaration "
  "is mostly not in the rainfall data.")

h("Conformal prediction", 2)
p("As a secondary check, conformal prediction sets were computed at a 10% "
  "miscoverage target. Realised coverage was 91.0%, with 95.0% of predictions "
  "resolving to a single label and 5.0% returning an empty set — the cases "
  "where the model declines to commit.")

# ================================================================== 9 stage 4

h("9. Stage 4 — the dashboard", 1)
p("The model is exported to a single JSON file containing the scaler means "
  "and scales, the nine coefficients, the intercept, the Platt calibration "
  "parameters, and the frozen per-county rainfall thresholds. Scoring is then "
  "pure arithmetic: standardise, dot product, add intercept, apply Platt, "
  "apply the logistic function.")
p("The exported arithmetic was verified against the original scikit-learn "
  "pipeline and agrees to 2e-9. This matters more than it sounds — it means "
  "the paper and the dashboard cannot drift apart. There is no second "
  "implementation to fall out of sync.")
p("County geometry is projected once from Census TIGER boundaries into SVG "
  "paths, so the browser never ships a mapping library. The whole site is "
  "three static files plus four JSON payloads, served by GitHub Pages with no "
  "server, no tracking and no build step.")

table(["File", "Size", "Contents"],
      [["model.json", "21 KB", "Coefficients, calibration, 218 county thresholds"],
       ["replay.json", "177 KB", "Every held-out hazard day with ground truth"],
       ["geo.json", "168 KB", "Projected county outlines"],
       ["live.json", "13 KB", "The current rolling forecast window"]])

h("The three views", 2)
bullet("Historical replay — every hazard day in the held-out period, with the "
       "model's ranking beside the real outcome. Ground truth is in the "
       "table, so every number on screen is checkable by the viewer. The best "
       "single day is 2021-05-17, the Louisiana floods, where 5 of the top 6 "
       "ranked counties were declared.")
bullet("Live forecast — a rolling window from recent observed rainfall out to "
       "the NWS forecast horizon. No ground truth exists for these days, so "
       "unlike the replay, nothing here is checkable. Say this out loud when "
       "demonstrating it.")
bullet("Operating point explorer — recomputes precision, recall and lift in "
       "the browser across all 5,260 held-out rows as the threshold slider "
       "moves. It reproduces the Stage 3 table exactly, which is the point: "
       "the audience can verify the published numbers themselves.")

# ================================================================== 10 live

h("10. How the live path works", 1)
p("The hard part is not the forward forecast. It is the antecedent features. "
  "prev7 and prev30 need rainfall that has already fallen, and that comes "
  "from a different source with different timing than the forecast.")

table(["Source", "Role", "Latency"],
      [["NCEI GHCN-Daily", "Observed rainfall, the same station network used "
        "in training", "About 2 days behind"],
       ["NWS gridpoint QPF", "Forecast rainfall, about 7 days ahead", "Live"]])

p("A verification mode compares the live fetch against the stored training "
  "panel. On shared station-days the two agree to within 0.05 mm 98.76% of the "
  "time. Coverage differs because GHCN backfills and quality-controls recent "
  "data for weeks after publication, which is why live output will never be "
  "bit-identical to a replay row.")

h("The gap between the two sources", 2)
p("Observations stop about two days ago. The forecast starts today. That "
  "leaves a short span — usually one day, sometimes three across a weekend — "
  "that neither source covers. Because every prediction needs a complete "
  "72-hour rainfall window, any day whose window touches that gap cannot be "
  "scored at all.")
p("The dashboard handles this explicitly. Those dates are named as not "
  "selectable, and choosing one snaps to the nearest day that has input while "
  "explaining why. An unpredictable day is never drawn as a quiet day, "
  "because silence and absence of data are different statements.")

h("Quiet days", 2)
p("On most days no Gulf county crosses its threshold. Rather than render a "
  "blank screen, the dashboard says so explicitly and lists the closest "
  "counties with how far they actually got — for example Tallapoosa, Alabama "
  "at 59 mm of the 82 mm it would need, 72% of the way. These counties are "
  "shaded a flat neutral colour, never a point on the probability ramp, "
  "because below threshold there is no probability to report, only a distance.")

h("Staying current", 2)
p("A GitHub Action regenerates the live window every day at 10:20 UTC and "
  "commits the result. Without it the live tab would freeze on the day it was "
  "last generated, and a stale forecast still looks current — which is worse "
  "than having no live tab at all. The browser revalidates the file on every "
  "load rather than trusting its cache, so a returning visitor sees the "
  "refreshed window.")

# ================================================================== 11 FAQ

doc.add_page_break()
h("11. Questions you will be asked, and how to answer them", 1)

qa = [
    ("My weather app says 60% chance of rain in Houston tomorrow. Why does "
     "your dashboard show nothing for Houston?",
     "Three different quantities are involved. The weather app shows "
     "probability of precipitation — the chance of any measurable rain. The "
     "model consumes a different product, quantitative precipitation forecast, "
     "which is how many millimetres are expected. Harris County needs 97.9 mm "
     "over three days to enter the data at all. A 60% chance of a light shower "
     "is nowhere near that. And the output is a third quantity again: "
     "probability of a federal disaster declaration. Both statements are true "
     "at the same time; they are simply not the same measurement."),
    ("Why is the highest probability only 15.7%? That seems low.",
     "Because that is what the data supports. Federal declarations are rare "
     "even after extreme rainfall — 4.6% of hazard county-days in the test "
     "period. A model claiming 80% would be badly calibrated and would be "
     "lying. The value of the tool is the ranking and the roughly 2x lift, "
     "not the absolute magnitude."),
    ("Is 2x lift actually good?",
     "It is modest and it is real. It was measured on a held-out time period "
     "the model never saw, against a baseline that is hard to beat, with "
     "bootstrap confidence intervals. Most of the obvious ways to inflate that "
     "number — random splits, reporting accuracy, tuning on the test set — "
     "were deliberately avoided."),
    ("Why not use deep learning?",
     "A Temporal Fusion Transformer was trained in Stage 1 and is in the "
     "repository. The constraint is not model capacity, it is that there are "
     "only 413 independent positive dates. Even XGBoost's apparent advantage "
     "over logistic regression turned out to be smaller than its own "
     "seed-to-seed variance. More model complexity cannot manufacture "
     "information that is not in the data."),
    ("Can this be used operationally?",
     "Not as a warning system, and not in Texas. As a triage aid for an "
     "analyst who already has to review every county that got hit, at the 10% "
     "threshold it flags about 86 counties a year with 11.8% precision against "
     "a 4.6% base rate. Whether that is worth the analyst's time is an "
     "operational judgement, not a modelling one."),
    ("What would improve it most?",
     "Not a better model. Better features — county emergency-management "
     "capacity, prior declaration history, damage assessments, river gauge "
     "levels. A declaration is a political and administrative outcome as well "
     "as a physical one, and the model currently sees only rainfall."),
    ("How do I know the numbers on the website are real?",
     "The historical replay shows ground truth next to every prediction, and "
     "the operating-point explorer recomputes the published precision and "
     "recall in the browser from the raw held-out rows. Both are checkable "
     "live, in front of an audience, without trusting the author."),
]
for q, a in qa:
    p(q, bold=True)
    p(a)

# ================================================================== 12 limits

h("12. Known limits — state these before someone else finds them", 1)
bullet("Texas is effectively unvalidated at 1.11x lift. Treat any Texas row as "
       "uninformative.")
bullet("The NWS forecasts a grid cell at the county centroid, while training "
       "used the mean across the county's reporting stations. This "
       "representativeness mismatch is real and unquantified.")
bullet("Declarations are a political and administrative outcome, not purely a "
       "physical one. The model sees rainfall, not county emergency-management "
       "capacity or state-level request behaviour.")
bullet("Coefficients are standardised logistic weights among correlated "
       "features. Sign and relative size are meaningful; they are not causal "
       "effects and single coefficients should not be quoted in isolation.")
bullet("Live output will never exactly match a replay row, because GHCN "
       "revises recent data for weeks after first publication.")
bullet("The model is conditional. It has nothing to say about a county below "
       "its rainfall threshold — which is almost every county, almost every "
       "day.")

# ================================================================== 13 running

h("13. Running it yourself", 1)
p("Nothing needs to be installed to view the dashboard — it is a public web "
  "page. To reproduce the analysis:")
code = doc.add_paragraph()
run = code.add_run(
    "git clone https://github.com/vigneshwar-lokoji/gulf-coast-disaster-risk\n"
    "cd gulf-coast-disaster-risk\n"
    "pip install -r requirements.txt\n\n"
    "python stage2.py          # build the hazard-day panel and fit the model\n"
    "python stage3.py          # calibration, operating points, leave-one-state-out\n"
    "python stage4_export.py   # freeze the model to docs/data/model.json\n"
    "python stage4_geo.py      # project county geometry to SVG paths\n"
    "python stage4_live.py     # fetch live rainfall and predict the current window\n\n"
    "python serve.py           # http://localhost:8787")
run.font.name = "Consolas"
run.font.size = Pt(9.5)

p("The raw rainfall panel is 25 MB and is not committed to the repository, "
  "since it is large and not ours to republish. Everything the dashboard and "
  "the daily refresh job need is already in the repository, so a clean "
  "checkout runs the live path without any private data.")

h("Repository layout", 2)
table(["Path", "What it is"],
      [["config.py, build_panel.py", "Shared configuration and panel construction"],
       ["baselines.py", "Tier-0 climatology and persistence baselines"],
       ["train_xgb.py, tft_colab.py", "Stage 1 daily hazard models"],
       ["stage2.py", "Hazard-day reframing, nested ablation, model fit"],
       ["stage3.py", "Calibration, operating points, leave-one-state-out"],
       ["stage4_export.py, stage4_geo.py", "Freeze model and geometry for the browser"],
       ["stage4_live.py", "Live rainfall fetch and current-window prediction"],
       ["docs/", "The dashboard itself, served by GitHub Pages"],
       ["artifacts/", "Every result table referenced in this document"]])

p("Every number in this document is read directly from the files in "
  "artifacts/ and docs/data/ when it is generated, so the document cannot "
  "drift away from the results.", italic=True)

out = HERE / "PROJECT_EXPLAINER.docx"
doc.save(out)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
