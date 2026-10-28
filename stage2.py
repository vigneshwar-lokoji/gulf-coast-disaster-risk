"""Stage 2: conditional vulnerability.

Stage 1 predicts the physical hazard (3-day precipitation above a county's
training-period p99). Stage 2 conditions on the hazard having occurred and asks
which counties were federally declared. Measured by stage2_feasibility.py:
22,754 hazard events, 1,229 declared within 14 days, prevalence 5.4%, spread
over 413 distinct hazard dates.

413 independent positive dates is small-n. The primary model is therefore
penalized logistic regression, not a tree ensemble: its coefficients are the
contribution ("one SD more socioeconomic vulnerability multiplies declaration
odds by X, holding rainfall severity and regional extent fixed"), which is a
claim a tree cannot make. XGBoost runs second as a nonlinearity check.

The scientific test is the nested sequence H -> H+E -> H+E+SVI. SVI only earns
a finding if it adds average precision *after* hazard severity and regional
extent are controlled for, because FEMA declares regionally and spatial extent
is the obvious confound.

There are no epochs here. SMOKE reduces bootstrap resamples, CV folds and tree
count to exercise every path cheaply; it writes to separate files so trial
numbers cannot be mistaken for real ones.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import DATA_DIR, FILES, OUT_DIR

# ---------------------------------------------------------------- 1. config

SMOKE = False

N_BOOT = 200 if SMOKE else 2000
N_FOLDS = 2 if SMOKE else 5
XGB_TREES = 50 if SMOKE else 600
SEEDS = (0, 1, 2)

START_YEAR = 2000
MIN_COVERAGE = 0.95
TRAIN_END = 2014
VAL_END = 2019
LAG_WINDOW_DAYS = 14

# Grouping unit for CV and bootstrap. Calendar date is not enough: 81% of
# consecutive hazard dates are literally adjacent, because storms last days.
# Grouping by date let day 1 of a storm train a model tested on day 2, which
# inflated CV AP to 0.16 against an honest 0.088 on the temporal holdout, and
# made every bootstrap CI too narrow. A storm episode is a contiguous run of
# hazard dates separated by gaps shorter than this.
EPISODE_GAP_DAYS = 7

# "broad" sweeps in wind and surge declarations, where rainfall-driven social
# vulnerability logic does not apply. "flood_only" is the fair test of the
# hypothesis. Note that incidentType == "Flood" alone misses about half of real
# floods, because FEMA files many under "Severe Storm" with the flooding named
# only in declarationTitle — so the flood-only filter has to read both.
LABEL_SCOPE = "broad"

BROAD_HINTS = ("flood", "rain", "storm", "hurricane", "tropical")
DROP_TYPES = {"Biological", "Human Cause", "Toxic Substances", "Fire",
              "Drought", "Freezing", "Snowstorm", "Winter Storm", "Earthquake"}

STAGE2_DIR = OUT_DIR / "stage2"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)

# Data prep is deterministic given the label scope, so it is cached per scope
# and shared by both SMOKE and full runs.
EVENTS_CACHE = STAGE2_DIR / f"stage2_events_{LABEL_SCOPE}.parquet"

_sfx = f"_{LABEL_SCOPE}" + ("_SMOKE" if SMOKE else "")
RESULTS_CSV = STAGE2_DIR / f"stage2_results{_sfx}.csv"
COEFS_CSV = STAGE2_DIR / f"stage2_coefficients{_sfx}.csv"
PREDS_PARQUET = STAGE2_DIR / f"stage2_test_predictions{_sfx}.parquet"

REBUILD_EVENTS = False

# ------------------------------------------------------- 2. the event table


def fips(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def build_events():
    w = pd.read_parquet(DATA_DIR / FILES["weather"], columns=["STATION", "DATE", "PRCP"])
    w = w[w["DATE"].dt.year >= START_YEAR]

    mapping = pd.read_csv(DATA_DIR / FILES["mapping"])
    mapping["county_fips"] = fips(mapping["county_fips"])
    w = w.merge(mapping[["STATION", "county_fips"]], on="STATION", how="inner")

    daily = (
        w.groupby(["county_fips", "DATE"])["PRCP"].mean()
        .rename("prcp").reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(daily["county_fips"].unique()),
         pd.date_range(daily["DATE"].min(), daily["DATE"].max(), freq="D")],
        names=["county_fips", "DATE"],
    )
    daily = daily.set_index(["county_fips", "DATE"]).reindex(grid).reset_index()

    cov = daily.groupby("county_fips")["prcp"].apply(lambda s: s.notna().mean())
    daily = daily[daily["county_fips"].isin(cov[cov >= MIN_COVERAGE].index)].copy()
    daily["prcp"] = daily["prcp"].fillna(0.0)
    daily["year"] = daily["DATE"].dt.year
    daily = daily.sort_values(["county_fips", "DATE"]).reset_index(drop=True)

    g = daily.groupby("county_fips", sort=False)["prcp"]
    daily["fwd3"] = g.transform(lambda s: s.rolling(3).sum().shift(-2))
    # Antecedent wetness: how saturated the ground already was. Backward-looking
    # only, so it is a legitimate feature rather than a peek at the answer.
    daily["prev7"] = g.transform(lambda s: s.rolling(7).sum().shift(1))
    daily["prev30"] = g.transform(lambda s: s.rolling(30).sum().shift(1))
    daily["max1"] = g.transform(lambda s: s.rolling(3).max().shift(-2))

    # Frozen on training years, exactly as in Stage 1.
    thr = (
        daily[daily.year <= TRAIN_END]
        .groupby("county_fips")["fwd3"].quantile(0.99).rename("thr")
    )
    daily = daily.join(thr, on="county_fips").dropna(subset=["fwd3", "thr", "prev30"])

    haz = daily[daily["fwd3"] >= daily["thr"]].copy()
    haz["state"] = haz["county_fips"].str[:2]

    # FEMA declares regionally. Spatial extent is the confound that has to be
    # controlled before social vulnerability is credited with anything.
    extent = (
        haz.groupby(["state", "DATE"])["county_fips"].size()
        .rename("state_extent").reset_index()
    )
    haz = haz.merge(extent, on=["state", "DATE"], how="left")
    haz["other_extent"] = haz["state_extent"] - 1
    gulf = haz.groupby("DATE")["county_fips"].size().rename("gulf_extent").reset_index()
    haz = haz.merge(gulf, on="DATE", how="left")

    # Label: any qualifying declaration for this county with onset in
    # [hazard_date, hazard_date + LAG_WINDOW_DAYS].
    f = pd.read_csv(DATA_DIR / FILES["fema"], low_memory=False)
    f["county_fips"] = fips(f["county_fips"])
    f = f[~f["incidentType"].isin(DROP_TYPES)]
    itype = f["incidentType"].fillna("")
    title = f["declarationTitle"].fillna("").str.lower()
    if LABEL_SCOPE == "flood_only":
        keep = (itype == "Flood") | title.str.contains("flood")
    elif LABEL_SCOPE == "broad":
        keep = (itype.str.lower() + " " + title).str.contains("|".join(BROAD_HINTS), regex=True)
    else:
        raise ValueError(f"unknown LABEL_SCOPE: {LABEL_SCOPE}")
    f = f[keep]
    print(f"  label scope '{LABEL_SCOPE}': {f.disasterNumber.nunique()} disasters, "
          f"{len(f):,} county-onset rows before dedup")
    f["begin"] = pd.to_datetime(f["incidentBeginDate"], errors="coerce", utc=True)
    f["begin"] = f["begin"].dt.tz_localize(None).dt.normalize()
    f = f.dropna(subset=["begin"])
    dec = f[["county_fips", "begin", "disasterNumber"]].drop_duplicates().sort_values("begin")

    haz = haz.sort_values("DATE").reset_index(drop=True)
    matched = pd.merge_asof(
        haz[["county_fips", "DATE"]],
        dec.rename(columns={"begin": "DATE"}),
        on="DATE", by="county_fips", direction="forward",
        tolerance=pd.Timedelta(days=LAG_WINDOW_DAYS),
    )
    haz["disasterNumber"] = matched["disasterNumber"].to_numpy()
    haz["y"] = haz["disasterNumber"].notna().astype(int)

    svi = pd.read_csv(DATA_DIR / FILES["svi"])
    svi["county_fips"] = fips(svi["FIPS"])
    svi = svi[["county_fips", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3",
               "RPL_THEME4", "RPL_THEMES", "E_TOTPOP", "AREA_SQMI"]]
    # CDC codes suppressed percentile ranks as -999.
    for c in [c for c in svi.columns if c.startswith("RPL_")]:
        svi[c] = svi[c].where(svi[c].between(0, 1))
    haz = haz.merge(svi, on="county_fips", how="inner")

    haz["log_pop"] = np.log1p(haz["E_TOTPOP"])
    haz["log_area"] = np.log1p(haz["AREA_SQMI"])
    haz["pop_density"] = haz["E_TOTPOP"] / haz["AREA_SQMI"].replace(0, np.nan)
    haz["log_density"] = np.log1p(haz["pop_density"])
    haz["ratio"] = haz["fwd3"] / haz["thr"]
    haz["log_fwd3"] = np.log1p(haz["fwd3"])
    haz["month"] = haz["DATE"].dt.month
    # Cyclical, so December and January are adjacent. Seasonality is physical,
    # unlike `year`, which encodes FEMA policy drift and must stay out.
    haz["month_sin"] = np.sin(2 * np.pi * haz["month"] / 12)
    haz["month_cos"] = np.cos(2 * np.pi * haz["month"] / 12)

    return haz.reset_index(drop=True)


if EVENTS_CACHE.exists() and not REBUILD_EVENTS:
    ev = pd.read_parquet(EVENTS_CACHE)
    print(f"loaded cached events: {EVENTS_CACHE}")
else:
    ev = build_events()
    ev.to_parquet(EVENTS_CACHE, index=False)
    print(f"built and cached events: {EVENTS_CACHE}")

def assign_episodes(frame):
    """Label each row with its storm episode. Derived from DATE alone, so it
    needs no cache rebuild."""
    dates = np.sort(frame["DATE"].unique())
    gap = np.diff(dates).astype("timedelta64[D]").astype(int)
    ep_of_date = pd.Series(np.concatenate([[0], (gap > EPISODE_GAP_DAYS).cumsum()]),
                           index=dates, name="episode")
    return frame["DATE"].map(ep_of_date).to_numpy()


ev["episode"] = assign_episodes(ev)

print(f"\n[stage2 {LABEL_SCOPE}{' SMOKE' if SMOKE else ''}] rows {len(ev):,}  positives {ev.y.sum():,}  "
      f"prevalence {ev.y.mean():.4f}  counties {ev.county_fips.nunique()}  "
      f"dates {ev.DATE.nunique():,}")
print(f"  {ev.episode.nunique()} storm episodes (the independent unit for CV and "
      f"bootstrap) from {ev.DATE.nunique():,} dates, gap threshold {EPISODE_GAP_DAYS}d")
print(f"  positives span {ev.loc[ev.y == 1, 'episode'].nunique()} episodes "
      f"({ev.loc[ev.y == 1, 'DATE'].nunique()} dates), "
      f"{ev.loc[ev.y == 1, 'county_fips'].nunique()} counties")

# -------------------------------------------------- 3. the nested feature sets

F_HAZARD = ["ratio", "log_fwd3", "max1", "prev7", "prev30", "month_sin", "month_cos"]
F_EXTENT = ["other_extent", "gulf_extent"]
# log_density is log_pop - log_area, an exact linear combination of the other
# two. Including all three made the fit unstable (odds ratios of 139 and 0.008
# cancelling each other) and cost the exposure block AP. Density is recoverable
# from the pair, so nothing is lost by dropping it.
F_EXPOSE = ["log_pop", "log_area"]
F_SVI = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]

BLOCKS = {
    "H_hazard":            F_HAZARD,
    "HE_hazard_extent":    F_HAZARD + F_EXTENT,
    "HEX_plus_exposure":   F_HAZARD + F_EXTENT + F_EXPOSE,
    "HEXS_plus_svi":       F_HAZARD + F_EXTENT + F_EXPOSE + F_SVI,
}

ALL_FEAT = sorted({c for v in BLOCKS.values() for c in v})
missing = ev[ALL_FEAT].isna().sum()
if missing.any():
    print("\nNaNs per feature (median-imputed from training years only):")
    print(missing[missing > 0].to_string())

tr = ev[ev.year <= TRAIN_END].copy()
va = ev[(ev.year > TRAIN_END) & (ev.year <= VAL_END)].copy()
te = ev[ev.year > VAL_END].copy()
med = tr[ALL_FEAT].median()
for part in (tr, va, te):
    part[ALL_FEAT] = part[ALL_FEAT].fillna(med)

print(f"  train {len(tr):,} ({tr.y.sum()} pos)  val {len(va):,} ({va.y.sum()} pos)  "
      f"test {len(te):,} ({te.y.sum()} pos)")

PREV = te.y.mean()
print(f"  test prevalence {PREV:.4f} -> AP of a random ranker")

# ------------------------------------------------------- 4. Tier-0 baselines

print("\n--- Tier-0 baselines (test) ---")
base = {}
base["prevalence"] = PREV
base["severity_only"] = average_precision_score(te.y, te["ratio"])
base["extent_only"] = average_precision_score(te.y, te["other_extent"])
base["population_only"] = average_precision_score(te.y, te["log_pop"])
base["svi_overall_only"] = average_precision_score(te.y, te["RPL_THEMES"].fillna(med.get("RPL_THEME1", 0.5)))
for k, v in base.items():
    print(f"  {k:<18} AP {v:.4f}  lift {v / PREV:.2f}x")

# ------------------------------------------- 5. date-grouped bootstrap on test

# Resampling rows would treat 120 counties hit by one storm as 120 independent
# observations; resampling dates would treat each day of a multi-day storm as
# independent. Whole episodes are the unit that actually is.
test_ep = te["episode"].to_numpy()
uniq_ep = np.unique(test_ep)
ep_rows = {e: np.flatnonzero(test_ep == e) for e in uniq_ep}
print(f"  bootstrap resamples {len(uniq_ep)} test episodes "
      f"({te.DATE.nunique():,} dates, {len(te):,} rows)")


def _resample(rng):
    pick = rng.choice(uniq_ep, size=len(uniq_ep), replace=True)
    return np.concatenate([ep_rows[e] for e in pick])


def boot_ap(y, s, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = _resample(rng)
        yb = y[idx]
        out[b] = average_precision_score(yb, s[idx]) if yb.sum() else np.nan
    return out


def boot_delta(y, s_new, s_old, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = _resample(rng)
        yb = y[idx]
        if not yb.sum():
            out[b] = np.nan
            continue
        out[b] = (average_precision_score(yb, s_new[idx])
                  - average_precision_score(yb, s_old[idx]))
    return out


# ------------------------------------- 6. primary model: logistic regression

print(f"\n--- nested logistic regression (grouped CV {N_FOLDS} folds, "
      f"bootstrap {N_BOOT}) ---")

yte = te["y"].to_numpy()
rows, coef_rows, preds = [], [], te[["county_fips", "DATE", "year", "y"]].copy()
lr_scores = {}

for name, feats in BLOCKS.items():
    pipe = make_pipeline(
        StandardScaler(),
        # Convex objective, deterministic solver: this genuinely has no seed
        # dependence, unlike the subsampled XGBoost below.
        LogisticRegression(C=1.0, class_weight="balanced",
                           max_iter=2000, solver="lbfgs"),
    )

    cv_ap = []
    for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(tr[feats], tr.y, groups=tr["episode"]):
        pipe.fit(tr.iloc[tr_i][feats], tr.iloc[tr_i].y)
        p = pipe.predict_proba(tr.iloc[va_i][feats])[:, 1]
        cv_ap.append(average_precision_score(tr.iloc[va_i].y, p))

    pipe.fit(tr[feats], tr.y)
    s = pipe.predict_proba(te[feats])[:, 1]
    lr_scores[name] = s
    preds[f"p_lr_{name}"] = s

    ap = average_precision_score(yte, s)
    ci = np.nanpercentile(boot_ap(yte, s), [2.5, 97.5])
    val_ap = average_precision_score(va.y, pipe.predict_proba(va[feats])[:, 1])

    print(f"  {name:<20} n_feat {len(feats):>2}  cv AP {np.mean(cv_ap):.4f}"
          f"±{np.std(cv_ap):.4f}  val AP {val_ap:.4f}  "
          f"test AP {ap:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  lift {ap / PREV:.2f}x")

    rows.append(dict(model="logreg", block=name, n_feat=len(feats),
                     cv_ap=np.mean(cv_ap), cv_sd=np.std(cv_ap), val_ap=val_ap,
                     test_ap=ap, lift=ap / PREV, ci_lo=ci[0], ci_hi=ci[1],
                     smoke=SMOKE, n_boot=N_BOOT, n_folds=N_FOLDS))

    lr = pipe.named_steps["logisticregression"]
    for f_, c_ in zip(feats, lr.coef_[0]):
        coef_rows.append(dict(block=name, feature=f_, coef=c_,
                              odds_ratio_per_sd=float(np.exp(c_)), smoke=SMOKE))

# The actual scientific test: each increment against the one before it.
print("\n--- incremental value of each block (episode-grouped bootstrap) ---")
order = list(BLOCKS)
for prev_name, name in zip(order, order[1:]):
    d = boot_delta(yte, lr_scores[name], lr_scores[prev_name])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    sig = "SIGNIFICANT" if lo > 0 else ("negative" if hi < 0 else "not distinguishable from zero")
    print(f"  {prev_name} -> {name}: dAP {np.nanmean(d):+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  {sig}")
    rows.append(dict(model="delta_logreg", block=f"{prev_name}->{name}",
                     test_ap=np.nanmean(d), ci_lo=lo, ci_hi=hi, verdict=sig,
                     smoke=SMOKE, n_boot=N_BOOT, n_folds=N_FOLDS))

# --------------------------------- 7. secondary model: XGBoost, seeded 0/1/2

print("\n--- XGBoost on the full block, seeded (nonlinearity check) ---")
feats = BLOCKS["HEXS_plus_svi"]
spw = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
xgb_ap = {}
for seed in SEEDS:
    clf = xgb.XGBClassifier(
        n_estimators=XGB_TREES, learning_rate=0.05, max_depth=3,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2.0, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=40, tree_method="hist", n_jobs=-1,
        random_state=seed,
    )
    clf.fit(tr[feats], tr.y, eval_set=[(va[feats], va.y)], verbose=False)
    s = clf.predict_proba(te[feats])[:, 1]
    preds[f"p_xgb_seed{seed}"] = s
    xgb_ap[seed] = average_precision_score(yte, s)
    print(f"  seed {seed}: {clf.best_iteration + 1} trees  test AP {xgb_ap[seed]:.4f}  "
          f"lift {xgb_ap[seed] / PREV:.2f}x")

xv = np.array([xgb_ap[s] for s in SEEDS])
print(f"  across-seed: mean {xv.mean():.4f}  sd {xv.std(ddof=1):.4f}  "
      f"spread {xv.max() - xv.min():.4f}")
rows.append(dict(model="xgboost", block="HEXS_plus_svi", n_feat=len(feats),
                 test_ap=xv.mean(), lift=xv.mean() / PREV,
                 seed_sd=xv.std(ddof=1), seed_spread=xv.max() - xv.min(),
                 smoke=SMOKE, n_boot=N_BOOT, n_folds=N_FOLDS))

lr_full = average_precision_score(yte, lr_scores["HEXS_plus_svi"])
gap = xv.mean() - lr_full
print(f"\n  XGB - logreg gap {gap:+.4f} vs XGB seed spread "
      f"{xv.max() - xv.min():.4f}: "
      + ("gap exceeds seed noise" if abs(gap) > (xv.max() - xv.min())
         else "gap is WITHIN seed noise, not a finding"))

# --------------------------------------------------------------- 8. outputs

pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
pd.DataFrame(coef_rows).to_csv(COEFS_CSV, index=False)
preds.to_parquet(PREDS_PARQUET, index=False)
print(f"\nwrote {RESULTS_CSV}")
print(f"wrote {COEFS_CSV}")
print(f"wrote {PREDS_PARQUET}")

if SMOKE:
    print("\n" + "=" * 68)
    print("SMOKE RUN. Reduced bootstrap/folds/trees; numbers are for plumbing")
    print("checks only. Set SMOKE = False for the results that go in the paper.")
    print("=" * 68)
