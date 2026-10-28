"""Generates tft_colab.ipynb. Kept as a generator so the notebook JSON never has
to be hand-escaped."""

import json
from pathlib import Path

DRIVE_DIR = "/content/drive/MyDrive/ganesh climate data"

MD_INTRO = f"""# Stage 1 hazard model — Temporal Fusion Transformer

Predicts the **total rainfall over the next 3 days** for each Gulf Coast county,
as a full quantile distribution rather than a single number. The extreme-event
probability is then read off that distribution and compared head-to-head against
the XGBoost result.

**XGBoost scored average precision 0.0214 on the same task (test 2020-2026).**
That is the number to beat.

Data is read from `{DRIVE_DIR}`.

Run cells in order. `SMALL_RUN = True` keeps the first pass to a few minutes so
you can confirm the pipeline works before spending compute units.
"""

C_INSTALL = """%%capture
!pip install -q pytorch-forecasting pytorch-lightning
"""

C_VERSIONS = """import torch, lightning.pytorch as pl, pytorch_forecasting as pf
print("torch               ", torch.__version__)
print("lightning           ", pl.__version__)
print("pytorch-forecasting ", pf.__version__)
print("cuda available      ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu                 ", torch.cuda.get_device_name(0))
"""

C_MOUNT = f'''from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
DRIVE_DIR = Path("{DRIVE_DIR}")

FILES = {{
    "weather": "gulf_coast_historical.parquet",
    "svi": "svi_gulf_coast_counties (1).csv",
    "mapping": "station_county_mapping (1).csv",
    "stations": "gulf_coast_stations_official.csv",
}}

for k, v in FILES.items():
    p = DRIVE_DIR / v
    print(("OK   " if p.exists() else "MISSING "), k, "->", v)
'''

C_BUILD = '''import numpy as np, pandas as pd

START_YEAR = 1990
CLIM_LO, CLIM_HI = 1990, 2010
HORIZON = 3


def fips(s):
    return s.astype("int64").astype(str).str.zfill(5)


def build():
    m = pd.read_csv(DRIVE_DIR / FILES["mapping"])
    m["county_fips"] = fips(m["county_fips"])

    w = pd.read_parquet(DRIVE_DIR / FILES["weather"],
                        columns=["STATION", "DATE", "PRCP", "TMAX", "TMIN"])
    w = w[w["DATE"].dt.year >= START_YEAR].merge(
        m[["STATION", "county_fips"]], on="STATION", how="inner")

    daily = (w.groupby(["county_fips", "DATE"])
               .agg(prcp=("PRCP", "mean"), tmax=("TMAX", "mean"), tmin=("TMIN", "mean"))
               .reset_index())

    # Complete the calendar so rolling windows mean real elapsed days.
    grid = pd.MultiIndex.from_product(
        [sorted(daily.county_fips.unique()),
         pd.date_range(daily.DATE.min(), daily.DATE.max(), freq="D")],
        names=["county_fips", "DATE"])
    daily = daily.set_index(["county_fips", "DATE"]).reindex(grid).reset_index()

    P = daily.pivot(index="DATE", columns="county_fips", values="prcp").sort_index()
    TX = daily.pivot(index="DATE", columns="county_fips", values="tmax").sort_index()
    TN = daily.pivot(index="DATE", columns="county_fips", values="tmin").sort_index()
    P0 = P.fillna(0.0)

    cols = {
        "prcp": P0,
        "tmax": TX.ffill(limit=5),
        "tmin": TN.ffill(limit=5),
        "trail7": P.rolling(7, min_periods=3).sum(),
        "trail30": P.rolling(30, min_periods=10).sum(),
        "api95": P0.ewm(alpha=0.05, adjust=False).mean(),
        "api85": P0.ewm(alpha=0.15, adjust=False).mean(),
    }
    # Strictly future: rolling(h).sum().shift(-h) at t covers days t+1..t+h.
    cols["fut3"] = P0.rolling(HORIZON).sum().shift(-HORIZON)

    df = pd.concat([v.stack().rename(k) for k, v in cols.items()], axis=1).reset_index()
    df.columns = ["DATE", "county_fips"] + list(df.columns[2:])

    # Extreme threshold per county, from the reference period only.
    trail3 = P.rolling(HORIZON, min_periods=2).sum()
    ref = trail3[(trail3.index.year >= CLIM_LO) & (trail3.index.year <= CLIM_HI)]
    thr = ref.quantile(0.99).rename("thr_p99")
    df = df.join(thr, on="county_fips")
    df["y_ext3"] = (df["fut3"] >= df["thr_p99"]).astype(float)
    df.loc[df["fut3"].isna(), "y_ext3"] = np.nan

    svi = pd.read_csv(DRIVE_DIR / FILES["svi"])
    svi["county_fips"] = fips(svi["FIPS"])
    svi["pop_density"] = svi["E_TOTPOP"] / svi["AREA_SQMI"].replace(0, np.nan)
    svi = svi[["county_fips", "ST_ABBR", "RPL_THEMES", "pop_density"]].rename(
        columns={"ST_ABBR": "state"})

    st = pd.read_csv(DRIVE_DIR / FILES["stations"]).merge(
        m[["STATION", "county_fips"]], left_on="ID", right_on="STATION")
    geo = st.groupby("county_fips").agg(
        lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"),
        elevation=("ELEVATION", "mean")).reset_index()

    df = df.merge(svi, on="county_fips").merge(geo, on="county_fips")

    df["year"] = df.DATE.dt.year
    doy = df.DATE.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["time_idx"] = (df.DATE - df.DATE.min()).dt.days
    return df.dropna(subset=["fut3"])


df = build()
print(f"{len(df):,} county-days | {df.county_fips.nunique()} counties")
print(f"{df.DATE.min().date()} -> {df.DATE.max().date()}")
print(f"extreme rate: {df.y_ext3.mean():.4f}")
'''

C_CONFIG = '''SMALL_RUN = True          # True = few counties / few years / 2 epochs
MAX_ENCODER_LENGTH = 60   # days of history the model reads

TRAIN_YEARS = (1990, 2014)
VAL_YEARS   = (2015, 2019)
TEST_YEARS  = (2020, 2026)

if SMALL_RUN:
    keep = sorted(df.county_fips.unique())[:40]
    d = df[df.county_fips.isin(keep) & (df.year >= 2008)].copy()
    MAX_EPOCHS, HIDDEN, BATCH = 2, 16, 256
else:
    d = df.copy()
    MAX_EPOCHS, HIDDEN, BATCH = 30, 64, 512

# time_idx must restart contiguously for the subset.
d["time_idx"] = d.groupby("county_fips").cumcount()
d = d.sort_values(["county_fips", "time_idx"]).reset_index(drop=True)

print(f"{'SMALL' if SMALL_RUN else 'FULL'} run")
print(f"  {len(d):,} rows | {d.county_fips.nunique()} counties | "
      f"{d.year.min()}-{d.year.max()} | epochs={MAX_EPOCHS} hidden={HIDDEN}")
'''

C_DATASET = '''from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

UNKNOWN = ["prcp", "tmax", "tmin", "trail7", "trail30", "api95", "api85"]
KNOWN   = ["time_idx", "doy_sin", "doy_cos"]
STATIC  = ["lat", "lon", "elevation", "RPL_THEMES", "pop_density", "thr_p99"]

d[UNKNOWN + KNOWN + STATIC] = d[UNKNOWN + KNOWN + STATIC].astype("float32")
d[UNKNOWN + STATIC] = d[UNKNOWN + STATIC].fillna(0.0)

train_cut = d.loc[d.year <= TRAIN_YEARS[1], "time_idx"].max()
val_cut   = d.loc[d.year <= VAL_YEARS[1], "time_idx"].max()

training = TimeSeriesDataSet(
    d[d.time_idx <= train_cut],
    time_idx="time_idx",
    target="fut3",
    group_ids=["county_fips"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=1,
    static_reals=STATIC,
    time_varying_known_reals=KNOWN,
    time_varying_unknown_reals=UNKNOWN,
    # Rainfall is non-negative and heavily skewed, so normalise in log space
    # per county rather than globally.
    target_normalizer=GroupNormalizer(groups=["county_fips"], transformation="log1p"),
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, d[d.time_idx <= val_cut], min_prediction_idx=train_cut + 1, predict=False)
testing = TimeSeriesDataSet.from_dataset(
    training, d, min_prediction_idx=val_cut + 1, predict=False)

train_dl = training.to_dataloader(train=True, batch_size=BATCH, num_workers=2)
val_dl   = validation.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=2)
test_dl  = testing.to_dataloader(train=False, batch_size=BATCH * 2, num_workers=2)

print(f"train {len(training):,} | val {len(validation):,} | test {len(testing):,} samples")
'''

C_SANITY = '''# Sanity gate: the model must be able to memorise a tiny batch. If the loss does
# not collapse here, something is wrong with the windowing or the target and no
# amount of epochs will fix it.
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

tiny = training.to_dataloader(train=True, batch_size=64, num_workers=0)
probe = TemporalFusionTransformer.from_dataset(
    training, learning_rate=0.05, hidden_size=16, attention_head_size=1,
    dropout=0.0, hidden_continuous_size=8, loss=QuantileLoss(quantiles=QUANTILES))

t = pl.Trainer(max_steps=60, limit_train_batches=1, enable_checkpointing=False,
               logger=False, enable_progress_bar=False,
               accelerator="auto", devices=1)
t.fit(probe, train_dataloaders=tiny)
print("final train loss on one repeated batch:", float(t.callback_metrics["train_loss"]))
print("-> should be well below the first-step value; if flat, stop and debug.")
'''

C_TRAIN = '''from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=HIDDEN,
    attention_head_size=4,
    dropout=0.15,
    hidden_continuous_size=HIDDEN // 2,
    loss=QuantileLoss(quantiles=QUANTILES),
    log_interval=50,
)
print(f"{sum(p.numel() for p in tft.parameters()):,} parameters")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        # Checkpoint to Drive, not /content — Colab sessions are ephemeral and a
        # dropped session otherwise loses the whole run.
        ModelCheckpoint(dirpath=str(DRIVE_DIR / "checkpoints"),
                        monitor="val_loss", save_top_k=1),
    ],
    logger=False,
)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
print("best val_loss:", float(trainer.checkpoint_callback.best_model_score))
'''

C_EVAL = '''from sklearn.metrics import average_precision_score, roc_auc_score

raw = tft.predict(test_dl, mode="quantiles", return_x=True, return_y=True)
q = np.asarray(raw.output).reshape(-1, len(QUANTILES))   # predicted mm per quantile
y_true_mm = np.asarray(raw.y[0]).reshape(-1)
thr = np.asarray(raw.x["static_reals"][:, STATIC.index("thr_p99")]).reshape(-1)
# Undo the dataset's scaling of the static input to recover mm.
thr = testing.get_parameters()["scalers"]["thr_p99"].inverse_transform(
    thr.reshape(-1, 1)).reshape(-1) if "thr_p99" in testing.get_parameters().get("scalers", {}) else thr


def exceedance_prob(qvals, threshold, quantiles=QUANTILES):
    """P(rain > threshold) read off the predicted quantile curve by locating the
    threshold between adjacent quantile levels."""
    qs = np.asarray(quantiles)
    out = np.empty(len(qvals))
    for i, (row, t) in enumerate(zip(qvals, threshold)):
        if t <= row[0]:
            out[i] = 1.0
        elif t >= row[-1]:
            out[i] = 0.0
        else:
            j = np.searchsorted(row, t)
            lo, hi = row[j - 1], row[j]
            w = 0.0 if hi == lo else (t - lo) / (hi - lo)
            out[i] = 1.0 - (qs[j - 1] + w * (qs[j] - qs[j - 1]))
    return out


p_tft = exceedance_prob(q, thr)
y_ext = (y_true_mm >= thr).astype(int)

prev = y_ext.mean()
ap = average_precision_score(y_ext, p_tft)
print(f"test rows {len(y_ext):,} | positives {y_ext.sum():,} | prevalence {prev:.4f}")
print(f"TFT     average precision {ap:.4f}   lift {ap/prev:.2f}x   ROC {roc_auc_score(y_ext, p_tft):.4f}")
print(f"XGBoost average precision 0.0214   lift 2.73x   ROC 0.7151")
print()
print("BEATS XGBoost" if ap > 0.0214 else "DOES NOT BEAT XGBoost")
print("(SMALL_RUN uses a county/year subset, so prevalence differs — treat this"
      " as a pipeline check, not the real comparison. Set SMALL_RUN = False for"
      " the number that goes in the paper.)")
'''

C_INTERP = '''# TFT's built-in interpretability: which inputs it selected, and where in the
# 60-day history it attended. This is the explainability section of the paper.
interp = tft.interpret_output(raw.output if hasattr(raw, "output") else raw, reduction="sum")
tft.plot_interpretation(interp)
'''

CELLS = [
    ("md", MD_INTRO),
    ("md", "## 1. Environment"),
    ("code", C_INSTALL),
    ("code", C_VERSIONS),
    ("md", "## 2. Mount Drive and check the files are all present"),
    ("code", C_MOUNT),
    ("md", "## 3. Build daily features\n\nSame feature definitions as the local XGBoost run, so the comparison is fair."),
    ("code", C_BUILD),
    ("md", "## 4. Run size\n\nLeave `SMALL_RUN = True` for the first pass."),
    ("code", C_CONFIG),
    ("md", "## 5. Build the TFT dataset"),
    ("code", C_DATASET),
    ("md", "## 6. Sanity gate — can it memorise one batch?"),
    ("code", C_SANITY),
    ("md", "## 7. Train"),
    ("code", C_TRAIN),
    ("md", "## 8. Score against XGBoost"),
    ("code", C_EVAL),
    ("md", "## 9. What the model used"),
    ("code", C_INTERP),
]


def main():
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({
                "cell_type": "code", "metadata": {}, "source": lines,
                "execution_count": None, "outputs": [],
            })

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out = Path(__file__).parent / "tft_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
