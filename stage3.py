"""Stage 3: turn the Stage 2 score into a number a dashboard can display.

Stage 2 ranks counties well enough to be useful (2.02x over a 4.6% base rate)
but its output is not a probability. `class_weight="balanced"` deliberately
distorts the intercept to help the ranking, so a raw score of 0.4 does not mean
40%. A dashboard that printed it as a percentage would be lying.

Four things happen here:

  1. Platt calibration fitted on 2015-2019, evaluated on 2020-2026, so the
     displayed probability means what it says. Checked with Brier score,
     expected calibration error and a reliability table.
  2. Operating points. Average precision is the right metric for a paper and
     the wrong one for an operator, who needs "if I act on everything above
     this line, how many counties is that per year and how often am I right?"
  3. Split-conformal prediction sets at 90% target coverage, so the dashboard
     can distinguish a confident call from an ambiguous one.
  4. Leave-one-state-out, because a dashboard is useless if it only works in
     states it was trained on.

The model is the HE block (hazard severity + regional extent). Stage 2 showed
population, area and the SVI themes add nothing, and HE is both the best point
estimate and the simplest.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUT_DIR

STAGE2_DIR = OUT_DIR / "stage2"
STAGE3_DIR = OUT_DIR / "stage3"
STAGE3_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = STAGE2_DIR / "stage2_events_broad.parquet"
TRAIN_END, VAL_END = 2014, 2019
EPISODE_GAP_DAYS = 7
ALPHA = 0.10               # conformal target: 90% coverage
N_BOOT = 2000

# Must match Stage 2's HE block. The assertion below fails loudly if it drifts.
F_HAZARD = ["ratio", "log_fwd3", "max1", "prev7", "prev30", "month_sin", "month_cos"]
F_EXTENT = ["other_extent", "gulf_extent"]
FEATS = F_HAZARD + F_EXTENT
STAGE2_HE_TEST_AP = 0.0929

STATE_NAME = {"01": "AL", "12": "FL", "22": "LA", "28": "MS", "48": "TX"}

# ------------------------------------------------------------------ 1. data

ev = pd.read_parquet(EVENTS)
dates = np.sort(ev["DATE"].unique())
gap = np.diff(dates).astype("timedelta64[D]").astype(int)
ev["episode"] = ev["DATE"].map(
    pd.Series(np.concatenate([[0], (gap > EPISODE_GAP_DAYS).cumsum()]), index=dates)
).to_numpy()
ev["state"] = ev["county_fips"].str[:2].map(STATE_NAME).fillna(ev["county_fips"].str[:2])

med = ev.loc[ev.year <= TRAIN_END, FEATS].median()
ev[FEATS] = ev[FEATS].fillna(med)

tr = ev[ev.year <= TRAIN_END]
va = ev[(ev.year > TRAIN_END) & (ev.year <= VAL_END)]
te = ev[ev.year > VAL_END].reset_index(drop=True)
TEST_YEARS = te.year.max() - te.year.min() + 1

print(f"train {len(tr):,} ({tr.y.sum()} pos)  val {len(va):,} ({va.y.sum()} pos)  "
      f"test {len(te):,} ({te.y.sum()} pos, {te.episode.nunique()} episodes, "
      f"{TEST_YEARS} years)")


def fit_ranker(frame):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
    )
    return pipe.fit(frame[FEATS], frame.y)


model = fit_ranker(tr)
s_va = model.predict_proba(va[FEATS])[:, 1]
s_te = model.predict_proba(te[FEATS])[:, 1]
yte = te["y"].to_numpy()

ap = average_precision_score(yte, s_te)
assert abs(ap - STAGE2_HE_TEST_AP) < 5e-4, (
    f"HE block test AP {ap:.4f} does not reproduce Stage 2's {STAGE2_HE_TEST_AP}: "
    "the feature list or splits have drifted between stage2.py and stage3.py"
)
PREV = yte.mean()
print(f"HE ranker reproduces Stage 2: test AP {ap:.4f}  lift {ap / PREV:.2f}x")

# -------------------------------------------------------- 2. calibration

# Platt scaling, fitted on validation years only.
#
# Isotonic was the obvious first choice and it is wrong here: monotone but not
# *strictly* monotone, it collapsed 5,260 test rows onto three distinct
# probabilities, dropping AP from 0.0929 to 0.0649 and leaving the dashboard
# with three usable thresholds. Platt is strictly monotone, so the ranking - and
# therefore AP - survives exactly. The assertion below enforces that.
platt = LogisticRegression(C=1e6, max_iter=2000)
platt.fit(s_va.reshape(-1, 1), va.y)
p_va = platt.predict_proba(s_va.reshape(-1, 1))[:, 1]
p_te = platt.predict_proba(s_te.reshape(-1, 1))[:, 1]

ap_cal = average_precision_score(yte, p_te)
assert abs(ap_cal - ap) < 1e-9, (
    f"calibration changed AP {ap:.4f} -> {ap_cal:.4f}: the calibrator is not "
    "strictly monotone and is destroying ranking information"
)
p_raw_brier = brier_score_loss(yte, s_te)
p_cal_brier = brier_score_loss(yte, p_te)


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    out = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


print(f"\n--- calibration (Platt fitted on {TRAIN_END + 1}-{VAL_END}, "
      f"evaluated on {te.year.min()}-{te.year.max()}) ---")
print(f"  Brier  raw {p_raw_brier:.4f} -> calibrated {p_cal_brier:.4f}")
print(f"  ECE    raw {ece(yte, s_te):.4f} -> calibrated {ece(yte, p_te):.4f}")
print(f"  AP preserved exactly: {ap_cal:.4f}")

print("\n  reliability (equal-count bins):")
print(f"  {'predicted':>10}  {'observed':>9}  {'n':>6}")
qb = pd.qcut(p_te, 10, duplicates="drop")
rel = pd.DataFrame({"p": p_te, "y": yte, "bin": qb}).groupby("bin", observed=True).agg(
    predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
)
for _, r in rel.iterrows():
    print(f"  {r.predicted:>10.3f}  {r.observed:>9.3f}  {int(r.n):>6,}")
rel.to_csv(STAGE3_DIR / "reliability.csv")

# -------------------------------------------------------- 3. operating points

# What an emergency manager actually needs: pick a line, learn the workload and
# the hit rate. Recall is of all declared counties in the test period.
print(f"\n--- operating points ({te.year.min()}-{te.year.max()}, "
      f"{TEST_YEARS} years) ---")
print(f"  {'p >=':>6}  {'flagged/yr':>10}  {'precision':>9}  {'recall':>7}  {'lift':>6}")
ops = []
for thr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    m = p_te >= thr
    if m.sum() == 0:
        continue
    prec, rec = yte[m].mean(), yte[m].sum() / yte.sum()
    ops.append(dict(threshold=thr, flagged=int(m.sum()),
                    flagged_per_year=m.sum() / TEST_YEARS, precision=prec,
                    recall=rec, lift=prec / PREV))
    print(f"  {thr:>6.2f}  {m.sum() / TEST_YEARS:>10.1f}  {prec:>9.3f}  "
          f"{rec:>7.3f}  {prec / PREV:>6.2f}x")
pd.DataFrame(ops).to_csv(STAGE3_DIR / "operating_points.csv", index=False)

# ------------------------------------------------ 4. split-conformal prediction

# Adaptive nonconformity: s = 1 - p_hat(true label). Calibrated on the
# validation block.
#
# Caveat worth stating in the paper: conformal's finite-sample guarantee assumes
# exchangeability. Calibrating on 2015-2019 and testing on 2020-2026 is a
# temporal shift, so the guarantee is approximate and the empirical coverage
# below is the number to trust, not the nominal 90%.
score_va = np.where(va.y.to_numpy() == 1, 1 - p_va, p_va)
n = len(score_va)
q = np.quantile(score_va, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher")

in_1 = (1 - p_te) <= q
in_0 = p_te <= q
covered = np.where(yte == 1, in_1, in_0)
size = in_0.astype(int) + in_1.astype(int)

print(f"\n--- split-conformal prediction sets (target {1 - ALPHA:.0%}) ---")
print(f"  nonconformity quantile q {q:.4f} from {n:,} calibration rows")
print(f"  empirical coverage {covered.mean():.3f}")
for sz, label in ((0, "empty (model refuses)"), (1, "singleton (confident)"),
                  (2, "both labels (ambiguous)")):
    m = size == sz
    line = f"  set size {sz}  {label:<24} {m.mean():>6.1%} of rows"
    if sz == 1 and m.sum():
        line += f"   coverage {covered[m].mean():.3f}"
    print(line)
# Degeneracy check. At 4.6% prevalence a set-valued predictor can hit 90%
# coverage by answering {not declared} every time, which would be perfectly
# valid and completely useless. The split below is what reveals that.
only_0 = (size == 1) & in_0
only_1 = (size == 1) & in_1
print(f"    of which {{not declared}} {only_0.mean():>6.1%}   "
      f"{{declared}} {only_1.mean():>6.1%}")
if only_1.mean() < 0.01:
    print("  DEGENERATE: the predictor almost never issues a confident")
    print("  'declared' call, so coverage is met by the base rate alone and the")
    print("  sets carry no information. Use the calibrated probability and the")
    print("  operating points above instead.")
elif only_1.sum():
    print(f"  confident-declared calls: {only_1.sum():,} rows, "
          f"{yte[only_1].mean():.3f} actually declared")

# --------------------------------------------- 5. leave-one-state-out transfer

print("\n--- leave-one-state-out (train on 4 states, test on the 5th) ---")
print(f"  {'held out':>9}  {'rows':>6}  {'pos':>4}  {'prev':>6}  {'AP':>6}  "
      f"{'lift':>6}  {'in-state lift':>13}")
loso = []
for st in sorted(ev.state.unique()):
    tr_s = ev[(ev.year <= TRAIN_END) & (ev.state != st)]
    te_s = ev[(ev.year > VAL_END) & (ev.state == st)]
    if te_s.y.sum() < 5:
        print(f"  {st:>9}  too few positives to score ({te_s.y.sum()})")
        continue
    p_out = fit_ranker(tr_s).predict_proba(te_s[FEATS])[:, 1]
    ap_out = average_precision_score(te_s.y, p_out)
    prev_s = te_s.y.mean()
    # Same held-out rows, but scored by the all-states model, to separate "this
    # state is intrinsically hard" from "the model failed to transfer".
    ap_in = average_precision_score(
        te_s.y, model.predict_proba(te_s[FEATS])[:, 1]
    )
    loso.append(dict(state=st, rows=len(te_s), pos=int(te_s.y.sum()),
                     prevalence=prev_s, ap_heldout=ap_out,
                     lift_heldout=ap_out / prev_s, lift_allstates=ap_in / prev_s))
    print(f"  {st:>9}  {len(te_s):>6,}  {int(te_s.y.sum()):>4}  {prev_s:>6.3f}  "
          f"{ap_out:>6.4f}  {ap_out / prev_s:>5.2f}x  {ap_in / prev_s:>12.2f}x")
pd.DataFrame(loso).to_csv(STAGE3_DIR / "leave_one_state_out.csv", index=False)

# ---------------------------------------------------------------- 6. outputs

out = te[["county_fips", "state", "DATE", "year", "episode", "y"]].copy()
out["score"] = s_te
out["p_calibrated"] = p_te
out["conformal_includes_0"] = in_0
out["conformal_includes_1"] = in_1
out.to_parquet(STAGE3_DIR / "stage3_test_predictions.parquet", index=False)

pd.DataFrame([dict(
    test_ap=ap, lift=ap / PREV, prevalence=PREV, test_episodes=te.episode.nunique(),
    brier_raw=p_raw_brier, brier_calibrated=p_cal_brier,
    ece_raw=ece(yte, s_te), ece_calibrated=ece(yte, p_te),
    conformal_alpha=ALPHA, conformal_q=q, conformal_coverage=covered.mean(),
    frac_singleton=(size == 1).mean(), frac_ambiguous=(size == 2).mean(),
    frac_empty=(size == 0).mean(),
)]).to_csv(STAGE3_DIR / "stage3_summary.csv", index=False)

print(f"\nwrote {STAGE3_DIR}/ : reliability.csv, operating_points.csv, "
      "leave_one_state_out.csv, stage3_test_predictions.parquet, stage3_summary.csv")
