"""Unified training pipeline for the county flood / hurricane risk models.

Runs unchanged on Colab (data from Drive) and on CREST (data staged over SFTP).
Cell 2 is the only place that knows the difference; every later cell reads
PATHS and PARAMS and never hardcodes a location.

Both models warm-start from a saved checkpoint when one is present, so a run
continues the previous one instead of starting over -- XGBoost keeps boosting
from an existing booster, the TFT resumes from a Lightning checkpoint. Delete
the contents of models/ to force a cold start.

The file is valid Python as well as a notebook: `python pipeline.py` executes
all eleven cells top to bottom, which is how the Slurm job runs it.

Every run writes an immutable directory under artifacts/runs/<stamp>-<tag>/
holding its manifest, console log, results, plots, per-epoch TFT metrics and a
copy of its checkpoints. Nothing is overwritten between runs. Use CRISK_COLD=1
for any run whose numbers are going to be reported, so the archive does not
depend on checkpoints left behind by an earlier one.
"""

# %% ==========================================================================
# CELL 1 - libraries
# =============================================================================
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    # find_spec raises rather than returning None when the parent package is
    # absent, which is exactly the case off Colab.
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False

# Colab starts bare. On CREST the conda env is built ahead of time by
# hpc/00_setup_env.sbatch, because compute nodes should not be pip-installing.
if IN_COLAB:
    _need = [p for m, p in (
        ("xgboost", "xgboost>=2.1"),
        ("pytorch_forecasting", "pytorch-forecasting>=1.3"),
        ("lightning", "lightning>=2.2"),
    ) if importlib.util.find_spec(m) is None]
    if _need:
        # pytorch-forecasting resolves its own torch pin and will happily swap
        # Colab's CUDA build for a CPU wheel. That costs the GPU silently: the
        # run still completes, just ~30x slower. So record torch beforehand and
        # compare in a fresh interpreter, since this one has already imported it.
        _before = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True).stdout.strip()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need],
                       check=True)
        _after = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.__version__, torch.cuda.is_available())"],
            capture_output=True, text=True).stdout.strip()
        if _before and _after and not _after.startswith(_before):
            raise SystemExit(
                f"pip replaced torch {_before} -> {_after}\n"
                "Runtime > Restart session, then run this cell again. The "
                "packages are installed now, so the restarted run will skip "
                "this branch entirely.")
        if _after.endswith("False"):
            print(f"WARNING: torch reports no CUDA after install ({_after}). "
                  "Check Runtime > Change runtime type > GPU.")

import matplotlib
if not IN_COLAB:
    # Slurm and redirected local runs have no display. Colab keeps its inline
    # backend so the plots appear under the cell as well as landing in plots/.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve)

# Torch is optional: a CPU-only XGBoost run on the normal partition should not
# fail just because the deep-learning stack is absent.
try:
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.metrics import CrossEntropy
    # Python 3.14 switched the default start method to forkserver, whose
    # handshake dies with ECONNRESET under the dataloader here. Colab is on 3.12
    # and still defaults to fork, so this pins both to the one that works. The
    # dataset holds only pandas/numpy, never a CUDA handle, so fork is safe.
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    HAVE_TORCH = True
    TFT_SKIP_REASON = None
except ImportError as e:
    HAVE_TORCH = False
    TFT_SKIP_REASON = f"torch stack unavailable ({e})"
    print(f"{TFT_SKIP_REASON}; cells 6-7 will be skipped")

# The TFT is the only part that needs a GPU, so allow the XGBoost half to run
# on its own. Reuses the HAVE_TORCH gate rather than adding a second one, but
# keeps the reason separate: "skipped because I asked" and "skipped because the
# install is broken" need different responses, and conflating them cost time.
if os.environ.get("CRISK_SKIP_TFT") == "1":
    HAVE_TORCH = False
    TFT_SKIP_REASON = "CRISK_SKIP_TFT=1 (TFT not attempted)"
    print("CRISK_SKIP_TFT=1; running XGBoost only, cells 6-7 skipped")

# Declared here rather than in cell 6 so that cell 7 degrades to "no trained
# TFT" instead of raising NameError when cell 6 has not been run. Notebook
# cells get executed out of order by hand; that is a normal condition.
TFT, TFT_DS, TFT_HIST = {}, {}, {}

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
# Lightning advises on num_workers and on nn.Module hyperparameters that
# pytorch_forecasting sets internally; neither is actionable from here.
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning\..*")

# Lightning prints a hardware banner and three promos per fit(); in a Slurm log
# that buries the numbers the run exists to produce.
import logging
for _n in ("lightning.pytorch", "lightning.pytorch.utilities.rank_zero",
           "lightning.fabric", "pytorch_lightning"):
    logging.getLogger(_n).setLevel(logging.ERROR)

print(f"python     {platform.python_version()}")
print(f"numpy      {np.__version__}")
print(f"pandas     {pd.__version__}")
print(f"xgboost    {xgb.__version__}")
if HAVE_TORCH:
    print(f"torch      {torch.__version__}  cuda={torch.cuda.is_available()}")


# %% ==========================================================================
# CELL 2 - paths and parameters. The only environment-aware cell.
# =============================================================================
def detect_env():
    if IN_COLAB:
        return "colab"
    if os.environ.get("SLURM_JOB_ID") or Path("/work").is_dir():
        return "hpc"
    return "local"


ENV = os.environ.get("CRISK_ENV_KIND") or detect_env()

if ENV == "colab":
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

try:
    _here = Path(__file__).resolve().parent
except NameError:  # running as a notebook, where __file__ is undefined
    _here = Path.cwd()

ROOT = Path(os.environ.get("CRISK_ROOT") or {
    "colab": "/content/drive/MyDrive/crisk",
    "hpc": str(Path.home() / "crisk"),
    "local": str(_here),
}[ENV])

# Re-uploading the notebook drops any `%env CRISK_ROOT` cell added by hand, so
# the default path comes back and the run dies on a panel that was never there.
# The Drive folder holding the project gets renamed and re-nested freely, but
# the crisk/artifacts/daily_panel.parquet tail is stable -- so find it instead
# of demanding the user re-type the root every upload. Bounded to three depths
# because globbing all of Drive over FUSE is slow enough to look like a hang.
if ENV == "colab" and not os.environ.get("CRISK_ROOT"):
    if not (ROOT / "artifacts" / "daily_panel.parquet").exists():
        _md = Path("/content/drive/MyDrive")
        _hit = next((p for pat in ("crisk/artifacts/daily_panel.parquet",
                                   "*/crisk/artifacts/daily_panel.parquet",
                                   "*/*/crisk/artifacts/daily_panel.parquet")
                     for p in sorted(_md.glob(pat))), None)
        if _hit:
            ROOT = _hit.parent.parent
            print(f"CRISK_ROOT unset; found the panel under {ROOT}")
        else:
            print(f"CRISK_ROOT unset and no panel under {_md}/*/crisk/artifacts")

PATHS = {
    "root": ROOT,
    "data": ROOT / "data",            # raw staged inputs (Drive / WinSCP)
    "panel": ROOT / "artifacts" / "daily_panel.parquet",
    "artifacts": ROOT / "artifacts",
    "models": ROOT / "artifacts" / "models",      # checkpoints, warm-start source
    "plots": ROOT / "artifacts" / "plots",
    "results": ROOT / "artifacts" / "results",
    # Lightning rewrites the top-k checkpoint on every validation improvement.
    # On Colab that lands on Drive's FUSE mount, where the delete-then-rename
    # it does to replace a file is the least reliable operation available. Keep
    # the churn on local disk; the best epoch is copied to models/ after fit().
    "scratch": Path("/content/ckpt") if ENV == "colab" else ROOT / "artifacts" / "models",
}
for k in ("artifacts", "models", "plots", "results", "scratch"):
    PATHS[k].mkdir(parents=True, exist_ok=True)

PARAMS = {
    "tag": os.environ.get("CRISK_TAG", "run"),
    "smoke": os.environ.get("CRISK_SMOKE", "0") == "1",
    "seed": 0,
    "hazards": ["flood", "storm"],
    "horizon_days": 3,
    # The headline models answer "onset anywhere in t+1..t+3". A reviewer
    # reading a 1-3 day lead claim wants the three days separately, because
    # skill that does not decay with lead time is skill read off the present.
    "leads": [1, 2, 3],
    # Contiguous blocks, never shuffled: adjacent days share a weather state,
    # so a random split leaks the answer across the boundary.
    "years": {"train": [2000, 2014], "val": [2015, 2019], "test": [2020, 2026]},
    # Last year each hazard actually has labels for. NHC publishes a HURDAT2
    # season only once it is finalised, so the hurricane label trails the rain
    # record. build_daily.py prints the correct value when it builds the panel;
    # raise it with CRISK_STORM_LABEL_END once a new season lands.
    "label_end": {"storm": int(os.environ.get("CRISK_STORM_LABEL_END", 2024))},
    "counties": 0,          # 0 = all; otherwise subsample whole counties
    # A warm-started run continues whatever happened to be in models/, so its
    # numbers depend on files that are not in its own run directory and it
    # cannot be reproduced from the archive alone. Fine while iterating; set
    # CRISK_COLD=1 for any run whose numbers are going to be reported.
    "warm_start": os.environ.get("CRISK_COLD", "0") != "1",
    "xgb": {
        "rounds": 2000, "early": 50, "lr": 0.05, "max_depth": 6,
        "min_child_weight": 50, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
    },
    "tft": {
        "encoder_days": 30, "hidden": 32, "heads": 4, "dropout": 0.2,
        "hidden_continuous": 16, "lr": 0.01, "epochs": 30, "batch": 512,
        "early": 5, "pos_weight": 50.0, "limit_train_batches": 0.25,
        # Window assembly is pure Python and runs per batch, so a single-process
        # loader starves the GPU: the smoke run was slower on a T4 than on a
        # laptop CPU. Validation is capped too, because a full pass over 2015-2019
        # costs more than the epoch it is scoring.
        # Capped at the actual core count: a Colab CPU runtime has 2, and
        # oversubscribing makes the workers fight each other instead of feeding
        # the GPU. CRISK_WORKERS=0 falls back to in-process loading if an
        # environment's multiprocessing is broken; slow, but it finishes.
        "workers": int(os.environ.get("CRISK_WORKERS",
                                      min(4, (os.cpu_count() or 2)))),
        "limit_val_batches": 0.2,
    },
}

if PARAMS["smoke"]:
    PARAMS["tag"] = "smoke"
    PARAMS["years"] = {"train": [2015, 2019], "val": [2020, 2021],
                       "test": [2022, 2026]}
    PARAMS["counties"] = 150
    PARAMS["xgb"].update(rounds=60, early=15)
    PARAMS["tft"].update(epochs=2, limit_train_batches=0.05, hidden=16,
                         limit_val_batches=0.1)
    # The smoke run only reads 2015+ and 150 counties, so a pre-trimmed panel
    # is a fraction of the size. Uploading 100 MB to Drive through a browser
    # fails silently often enough that it is worth being able to prove the GPU
    # path works first, on a file small enough to transfer reliably.
    smoke_panel = PATHS["artifacts"] / "daily_panel_smoke.parquet"
    if smoke_panel.exists():
        PATHS["panel"] = smoke_panel
        print(f"smoke panel: {smoke_panel.name}")

random.seed(PARAMS["seed"])
np.random.seed(PARAMS["seed"])
if HAVE_TORCH:
    torch.manual_seed(PARAMS["seed"])

TAG = PARAMS["tag"]
# Tagged, because smoke and full runs are different architectures: smoke uses
# hidden=16 on a 150-county subsample, full uses hidden=32 on all 356. Sharing
# one filename means a smoke run leaves a checkpoint that the next full run
# tries to warm-start from and dies on a shape mismatch.
TARGETS = PARAMS["hazards"] + [f"{h}_l{k}" for h in PARAMS["hazards"]
                               for k in PARAMS["leads"]]
CKPT = {
    "xgb": {k: PATHS["models"] / f"xgb_{k}_{TAG}.json" for k in TARGETS},
    "tft": {k: PATHS["models"] / f"tft_{k}_{TAG}.ckpt" for k in TARGETS},
}

# Same-tag runs used to overwrite each other, so a directory listing could not
# tell you which numbers belonged together -- artifacts/results once held a TFT
# from 19:18 beside an XGBoost from 22:18 and a summary that predated both.
# Every run now owns a directory, written to directly rather than copied at the
# end, so a run that dies half way still leaves its manifest and whatever it
# had produced. Nothing is ever overwritten.
RUN_ID = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{TAG}"
# On Colab the run directory must NOT live on Drive. The 2026-11-02 GPU run
# wrote results_run.json, xgb_run.json, tft_run.json and split_run.png to the
# Drive mount, the log confirms all four writes, and none of them survived the
# session -- only params_run.json, written in the first seconds, made it. The
# mount buffers small writes and drops them when the session ends. So the run
# is assembled on local disk and sealed to Drive as one archive at the end,
# which is a single sequential write that is then read back and verified.
RUN_HOME = Path("/content/runs") if ENV == "colab" else PATHS["artifacts"] / "runs"
RUN_DIR = RUN_HOME / RUN_ID
# Where the sealed copy goes. None off Colab, where RUN_DIR is already durable.
RUN_SEAL_TO = (PATHS["artifacts"] / "runs") if ENV == "colab" else None
PATHS["results"] = RUN_DIR / "results"
PATHS["plots"] = RUN_DIR / "plots"
for k in ("results", "plots"):
    PATHS[k].mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirrors stdout into the run directory as it is produced.

    Appends, so re-running a cell by hand adds to the log rather than
    truncating the record of what already ran. stderr is left alone; note that
    Lightning's progress bar does not go there but to stdout, so it arrives
    here and is filtered out in write() rather than avoided.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1)

    def write(self, s):
        self.stream.write(s)
        # A live progress bar redraws itself with carriage returns instead of
        # newlines. Mirroring those frames fills the log with thousands of
        # partial redraws that bury the results, so only real lines are kept
        # and the bar stays on screen where it is useful.
        if "\r" not in s:
            self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()


if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(sys.stdout, RUN_DIR / "console.log")


def _git(*args):
    """Empty string off a working tree, which is the normal case on Colab."""
    try:
        r = subprocess.run(["git", *args], cwd=str(_here), capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if limit and fh.tell() > limit:
                break
    return h.hexdigest()


MANIFEST = RUN_DIR / "manifest.json"


def manifest_update(**kw):
    """Read-modify-write, so each cell can add what it alone knows.

    Cell 2 cannot describe the panel without loading it and cell 3 cannot
    report a wall time, so the manifest is built up instead of written once.
    """
    cur = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    cur.update(kw)
    MANIFEST.write_text(json.dumps(cur, indent=2, default=str))
    return cur


RUN_T0 = time.time()
# Populated by xgb_fit and tft_fit when they continue a prior checkpoint, so
# the archive records which parent a warm run descends from instead of leaving
# the reader to guess why two runs on the same panel disagree.
WARM_PARENTS = {}
manifest_update(
    run_id=RUN_ID, tag=TAG, smoke=PARAMS["smoke"],
    started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    env=ENV, host=platform.node(), root=str(ROOT),
    # A result is only reproducible against the code that produced it, and the
    # dirty flag is the honest part: it says the working tree held edits that
    # the recorded commit does not contain.
    git={"commit": _git("rev-parse", "HEAD"),
         "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
         "dirty": bool(_git("status", "--porcelain"))},
    versions={"python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, "xgboost": xgb.__version__,
              "torch": torch.__version__ if HAVE_TORCH else None,
              "cuda": (torch.cuda.get_device_name(0)
                       if HAVE_TORCH and torch.cuda.is_available() else None)},
    tft_skipped=TFT_SKIP_REASON,
    params=PARAMS,
)
(PATHS["results"] / f"params_{TAG}.json").write_text(json.dumps(PARAMS, indent=2))

print(f"run   {RUN_ID}")
print(f"env   {ENV}")
print(f"root  {ROOT}")
print(f"panel {PATHS['panel']}  exists={PATHS['panel'].exists()}")
for h in PARAMS["hazards"]:
    print(f"warm-start {h}: xgb={CKPT['xgb'][h].exists()}  "
          f"tft={CKPT['tft'][h].exists()}")
print(f"smoke {PARAMS['smoke']}  tag {TAG}")


# %% ==========================================================================
# CELL 3 - load the panel and split it into partitions
# =============================================================================
FEATS = [
    "prcp", "prcp_max_stn", "prcp_prev3", "prcp_prev7", "prcp_prev14",
    "prcp_prev30", "prcp_prev60", "prcp_prev90", "prcp_fwd1", "prcp_fwd3",
    "prcp_fwd3_max", "ratio_prev3", "ratio_fwd3", "ratio_prev30", "log_fwd3",
    "tmax", "tmin", "month_sin", "month_cos", "hurricane_season",
    "lat", "lon", "pop_density", "RPL_THEME1", "RPL_THEME4", "RPL_THEMES",
]
# More rain must never lower predicted risk. Without this the trees carve
# non-monotone thresholds out of individual historical events.
MONO_UP = {f for f in FEATS if f.startswith(("prcp", "ratio", "log"))}

if not PATHS["panel"].exists():
    # On Colab the cause is almost always a wrong root rather than a missing
    # panel, so show what is actually on Drive before suggesting a 40-minute
    # rebuild of a file the user already has.
    hint = ""
    if ENV == "colab":
        seen = sorted(Path("/content/drive/MyDrive").glob("*/crisk/artifacts"))
        hint = ("\nartifacts dirs on Drive: "
                + (", ".join(str(s) for s in seen) if seen else "none found")
                + "\nif one of those holds the panel, set it:"
                "\n  %env CRISK_ROOT=<the path above, minus /artifacts>")
    raise SystemExit(
        f"panel not found at {PATHS['panel']}{hint}\n"
        f"build it first:  CLIMATE_DATA_DIR={PATHS['data']} "
        f"CLIMATE_OUT_DIR={PATHS['artifacts']} "
        f"PANEL_START={PARAMS['years']['train'][0]} python build_daily.py"
    )

panel = pd.read_parquet(PATHS["panel"])
panel["year"] = panel["DATE"].dt.year

# A stale panel once produced two confidently wrong conclusions here, because
# nothing in the output said which file the numbers came from. The hash and
# mtime pin the input, and the label columns distinguish the current four-label
# NOAA build from the old two-label FEMA one at a glance.
_pstat = PATHS["panel"].stat()
manifest_update(panel={
    "path": str(PATHS["panel"]), "bytes": _pstat.st_size,
    "mtime_utc": datetime.fromtimestamp(_pstat.st_mtime, timezone.utc)
                         .isoformat(timespec="seconds"),
    "sha256": _sha256(PATHS["panel"]),
    "rows_as_read": int(len(panel)),
    "counties_as_read": int(panel["county_fips"].nunique()),
    "years_as_read": [int(panel["year"].min()), int(panel["year"].max())],
    "label_columns": sorted(c for c in panel.columns if c.startswith("y_")),
    "onset_columns": sorted(c for c in panel.columns if c.endswith("_onset")),
})
print(f"panel {PATHS['panel'].name}  {len(panel):,} rows  "
      f"{panel['year'].min()}-{panel['year'].max()}  "
      f"labels={sorted(c for c in panel.columns if c.startswith('y_'))}")
lo, hi = PARAMS["years"]["train"][0], PARAMS["years"]["test"][1]
panel = panel[panel["year"].between(lo, hi)]

if PARAMS["counties"]:
    # Subsample whole counties, never random rows: a county's days share one
    # climatology, so dropping rows inside one leaves a panel no county resembles.
    rng = np.random.default_rng(PARAMS["seed"])
    fips = np.sort(panel["county_fips"].unique())
    keep = rng.choice(fips, min(PARAMS["counties"], len(fips)), replace=False)
    panel = panel[panel["county_fips"].isin(keep)]

panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
panel["t"] = (panel["DATE"] - panel["DATE"].min()).dt.days.astype(int)
missing = [f for f in FEATS if f not in panel.columns]
if missing:
    raise SystemExit(f"panel is missing feature columns: {missing}")

# Exact-lead targets, derived here rather than in build_daily.py: {haz}_onset
# already marks the event's begin day, so y_{haz}_l{k} is just that column
# pulled k days back within the county. The shift relies on the sort above and
# must land before SPLIT, which takes copies.
for h in PARAMS["hazards"]:
    g = panel.groupby("county_fips", sort=False)[f"{h}_onset"]
    for k in PARAMS["leads"]:
        panel[f"y_{h}_l{k}"] = g.shift(-k).fillna(0).astype("int8")

yr = panel["year"]
SPLIT = {k: panel[yr.between(*v)] for k, v in PARAMS["years"].items()}
for name, part in SPLIT.items():
    if part.empty:
        raise SystemExit(f"{name} split is empty -- check PARAMS['years']")

# HURDAT2 only publishes a season after NHC finalises it, so the hurricane
# label stops short of the panel's last year. Those years hold no positives by
# construction: training on them teaches the model that hurricanes stopped
# happening, and scoring on them hands every negative-class metric a free pass.
LABEL_END = {h: int(v) for h, v in PARAMS.get("label_end", {}).items()}


def base_haz(target):
    """'storm_l2' -> 'storm'. Lead targets inherit the hazard's label cap."""
    return re.sub(r"_l\d+$", "", target)


def S(name, target):
    """The rows of a split that carry a usable label for this target."""
    part = SPLIT[name]
    cap = LABEL_END.get(base_haz(target))
    return part if cap is None else part[part["year"] <= cap]


for h, cap in LABEL_END.items():
    drop = {n: len(SPLIT[n]) - len(S(n, h)) for n in SPLIT}
    if any(drop.values()):
        print(f"y_{h}: labels end {cap}; dropping rows "
              + ", ".join(f"{n} {v:,}" for n, v in drop.items() if v))

rows = []
for name, part in SPLIT.items():
    row = {"split": name, "years": f"{part['year'].min()}-{part['year'].max()}",
           "rows": len(part), "counties": part["county_fips"].nunique()}
    for h in PARAMS["hazards"]:
        # Per hazard, so the printed base rate is the one the model is scored
        # against rather than one diluted by years the label does not cover.
        s = S(name, h)
        row[f"{h}_pos"] = int(s[f"y_{h}"].sum())
        row[f"{h}_rate%"] = round(s[f"y_{h}"].mean() * 100, 4)
    rows.append(row)
split_table = pd.DataFrame(rows)
print(split_table.to_string(index=False))
manifest_update(splits=split_table.to_dict("records"),
                lead_targets=sorted(c for c in panel.columns
                                    if re.search(r"_l\d+$", c)))

fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 3.2), sharex=True)
colour = {"train": "#4878a8", "val": "#d4a017", "test": "#b5443a"}
for ax, h in zip(np.atleast_1d(axes), PARAMS["hazards"]):
    for name, part in SPLIT.items():
        s = part.groupby("year")[f"y_{h}"].sum()
        ax.bar(s.index, s.values, color=colour[name], label=name)
    ax.set_title(f"y_{h} positives per year")
    ax.set_xlabel("year")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"split_{TAG}.png", dpi=130)
plt.close(fig)
print(f"wrote {PATHS['plots'] / f'split_{TAG}.png'}")


# %% ==========================================================================
# CELL 4 - XGBoost training (warm-started from the saved booster if present)
# =============================================================================
def xgb_fit(haz):
    p = PARAMS["xgb"]
    model = xgb.XGBClassifier(
        n_estimators=p["rounds"], learning_rate=p["lr"],
        max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        monotone_constraints=tuple(1 if f in MONO_UP else 0 for f in FEATS),
        eval_metric="aucpr", early_stopping_rounds=p["early"],
        tree_method="hist", n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None,
        random_state=PARAMS["seed"],
    )
    prior = CKPT["xgb"][haz]
    warm = None
    if PARAMS["warm_start"] and prior.exists():
        # A booster can only be continued with the identical feature layout,
        # so a checkpoint from an older feature set is skipped, not forced.
        try:
            b = xgb.Booster()
            b.load_model(str(prior))
            if b.num_features() == len(FEATS):
                warm = b
                WARM_PARENTS[f"xgb:{haz}"] = {
                    "file": prior.name, "sha256": _sha256(prior),
                    "rounds_inherited": int(b.num_boosted_rounds())}
                print(f"  warm start from {prior.name} "
                      f"({b.num_boosted_rounds()} rounds)")
            else:
                print(f"  {prior.name} has {b.num_features()} features, "
                      f"panel has {len(FEATS)} -- cold start")
        except Exception as exc:
            print(f"  could not read {prior.name} ({exc}) -- cold start")

    model.fit(
        S("train", haz)[FEATS], S("train", haz)[f"y_{haz}"].to_numpy(),
        eval_set=[(S("train", haz)[FEATS], S("train", haz)[f"y_{haz}"].to_numpy()),
                  (S("val", haz)[FEATS], S("val", haz)[f"y_{haz}"].to_numpy())],
        xgb_model=warm, verbose=False,
    )
    model.save_model(CKPT["xgb"][haz])
    return model


XGB = {}
for haz in PARAMS["hazards"]:
    print(f"=== xgboost {haz} ===")
    t0 = time.time()
    XGB[haz] = xgb_fit(haz)
    print(f"  best_iteration={XGB[haz].best_iteration}  "
          f"{time.time() - t0:.1f}s  -> {CKPT['xgb'][haz].name}")

fig, axes = plt.subplots(1, len(XGB), figsize=(11, 3.4))
for ax, (haz, m) in zip(np.atleast_1d(axes), XGB.items()):
    h = m.evals_result()
    ax.plot(h["validation_0"]["aucpr"], label="train", color="#4878a8")
    ax.plot(h["validation_1"]["aucpr"], label="val", color="#d4a017")
    ax.axvline(m.best_iteration, ls="--", lw=1, color="#888")
    ax.set_title(f"{haz} - AUCPR by boosting round")
    ax.set_xlabel("round")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"xgb_training_{TAG}.png", dpi=130)
plt.close(fig)


# %% ==========================================================================
# CELL 5 - XGBoost validation and test, against Tier-0 baselines
# =============================================================================
BASELINE_FEATS = ("ratio_fwd3", "prcp_fwd3", "prcp_fwd3_max", "prcp_prev30",
                  "ratio_prev3", "log_fwd3", "pop_density")


def baselines(haz):
    """Anything the model cannot beat here is not a model."""
    tr, te = S("train", haz), S("test", haz)
    y = te[f"y_{haz}"].to_numpy()
    out = {"constant_prevalence": float(average_precision_score(
        y, np.full(len(y), tr[f"y_{haz}"].mean())))}
    for f in BASELINE_FEATS:
        out[f"single_{f}"] = float(average_precision_score(
            y, te[f].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()))
    rate = tr.groupby(tr["DATE"].dt.month)[f"y_{haz}"].mean()
    out["month_climatology"] = float(average_precision_score(
        y, te["DATE"].dt.month.map(rate).fillna(tr[f"y_{haz}"].mean()).to_numpy()))
    return out


def score(y, p):
    prev = float(np.mean(y))
    ap = float(average_precision_score(y, p))
    return {"ap": ap, "prevalence": prev, "lift": ap / prev if prev else None}


XGB_RESULT, XGB_PRED = {}, {}
for haz in PARAMS["hazards"]:
    base = baselines(haz)
    best = max(base, key=base.get)
    sv, st = S("val", haz), S("test", haz)
    pv = XGB[haz].predict_proba(sv[FEATS])[:, 1]
    pt = XGB[haz].predict_proba(st[FEATS])[:, 1]
    yv = sv[f"y_{haz}"].to_numpy()
    yt = st[f"y_{haz}"].to_numpy()
    # Keyed, not positional: cell 8 aligns this against the TFT, which scores a
    # different row subset because every window needs an encoder history.
    XGB_PRED[haz] = pd.DataFrame({
        "county_fips": st["county_fips"].astype(str).to_numpy(),
        "t": st["t"].to_numpy(), "y": yt, "p": pt})

    r = {"model": "xgboost_monotone",
         "val": score(yv, pv), "test": score(yt, pt),
         "best_baseline": best, "best_baseline_ap": base[best],
         "vs_best_baseline": score(yt, pt)["ap"] / base[best] if base[best] else None,
         "best_iteration": int(XGB[haz].best_iteration),
         "n_train_pos": int(S("train", haz)[f"y_{haz}"].sum()),
         "n_test_pos": int(yt.sum()), "baselines": base,
         "top_features": [[f, round(float(v), 5)] for f, v in sorted(
             zip(FEATS, XGB[haz].feature_importances_), key=lambda x: -x[1])[:10]]}
    r["beats_baseline"] = bool(r["test"]["ap"] > base[best])
    XGB_RESULT[haz] = r

    print(f"=== xgboost {haz} ===")
    for k, v in sorted(base.items(), key=lambda x: -x[1])[:3]:
        print(f"  baseline {k:28s} AP={v:.5f}")
    print(f"  val   AP={r['val']['ap']:.5f}  lift={r['val']['lift']:.1f}x")
    print(f"  TEST  AP={r['test']['ap']:.5f}  lift={r['test']['lift']:.1f}x  "
          f"vs_best={r['vs_best_baseline']:.2f}x")
    if not r["beats_baseline"]:
        print(f"  WARNING: loses to {best}. Keep the baseline, not the model.")

fig, axes = plt.subplots(2, len(PARAMS["hazards"]), figsize=(11, 7))
axes = np.atleast_2d(axes).reshape(2, -1)
for j, haz in enumerate(PARAMS["hazards"]):
    yt = XGB_PRED[haz]["y"].to_numpy()
    pt = XGB_PRED[haz]["p"].to_numpy()
    pr, rc, _ = precision_recall_curve(yt, pt)
    axes[0, j].plot(rc, pr, color="#4878a8")
    axes[0, j].axhline(yt.mean(), ls="--", lw=1, color="#b5443a",
                       label=f"prevalence {yt.mean():.5f}")
    axes[0, j].set_title(f"{haz} - test PR curve")
    axes[0, j].set_xlabel("recall")
    axes[0, j].set_ylabel("precision")
    axes[0, j].legend(frameon=False, fontsize=8)
    top = XGB_RESULT[haz]["top_features"][::-1]
    axes[1, j].barh([f for f, _ in top], [v for _, v in top], color="#4878a8")
    axes[1, j].set_title(f"{haz} - feature importance")
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"xgb_eval_{TAG}.png", dpi=130)
plt.close(fig)
(PATHS["results"] / f"xgb_{TAG}.json").write_text(json.dumps(XGB_RESULT, indent=2))
print(f"wrote {PATHS['results'] / f'xgb_{TAG}.json'}")


# %% ==========================================================================
# CELL 6 - TFT training (resumed from the saved checkpoint if present)
# =============================================================================
TFT_KNOWN = ["prcp_fwd1", "prcp_fwd3", "prcp_fwd3_max", "ratio_fwd3",
             "log_fwd3", "month_sin", "month_cos", "hurricane_season"]
TFT_UNKNOWN = ["prcp", "prcp_max_stn", "prcp_prev3", "prcp_prev7",
               "prcp_prev14", "prcp_prev30", "prcp_prev60", "prcp_prev90",
               "ratio_prev3", "ratio_prev30", "tmax", "tmin"]
TFT_STATIC = ["lat", "lon", "pop_density", "RPL_THEME1", "RPL_THEME4",
              "RPL_THEMES"]

# TFT / TFT_DS / TFT_HIST are initialised in cell 1. Re-running this cell
# clears them so a partial previous attempt cannot be mistaken for a result.
TFT.clear()
TFT_DS.clear()
TFT_HIST.clear()


def tft_frame(haz):
    """One frame with an integer time index and a per-row sample weight.

    Positives are reweighted rather than resampled so every county keeps its
    real calendar -- the encoder needs an unbroken 30-day history.
    """
    cols = (["county_fips", "t", "year", "DATE", f"y_{haz}"]
            + TFT_KNOWN + TFT_UNKNOWN + TFT_STATIC)
    d = panel[cols].copy()

    # `county_fips` is a static categorical, so a county with no rows in the
    # training window has no embedding and makes the encoder raise on unknown
    # category at val/test. Counties whose gauges came online late are the
    # usual cause. Drop them rather than mapping to a catch-all: an untrained
    # embedding still produces a score, and that score would be reported as if
    # it meant something.
    seen = set(d.loc[d["year"].between(*PARAMS["years"]["train"]), "county_fips"])
    unseen = sorted(set(d["county_fips"]) - seen)
    if unseen:
        d = d[d["county_fips"].isin(seen)]
        print(f"  dropped {len(unseen)} counties absent from the training "
              f"window: {', '.join(unseen[:6])}{' ...' if len(unseen) > 6 else ''}")

    # Same label-coverage cap the XGBoost side applies, so the two models are
    # not scored over different year ranges for the same hazard.
    cap = LABEL_END.get(base_haz(haz))
    if cap is not None:
        d = d[d["year"] <= cap]
    d[TFT_KNOWN + TFT_UNKNOWN + TFT_STATIC] = (
        d[TFT_KNOWN + TFT_UNKNOWN + TFT_STATIC]
        .replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32"))
    d["target"] = d[f"y_{haz}"].astype(int)
    d["w"] = np.where(d["target"] == 1, PARAMS["tft"]["pos_weight"], 1.0)
    d["county_fips"] = d["county_fips"].astype(str)
    return d


def tft_datasets(d):
    enc = PARAMS["tft"]["encoder_days"]
    bounds = {k: (int(d.loc[d["year"].between(*v), "t"].min()),
                  int(d.loc[d["year"].between(*v), "t"].max()))
              for k, v in PARAMS["years"].items()}

    lo, hi = bounds["train"]
    train = TimeSeriesDataSet(
        d[d["t"] <= hi], time_idx="t", target="target",
        group_ids=["county_fips"],
        max_encoder_length=enc, max_prediction_length=1,
        min_prediction_idx=lo + enc,
        static_categoricals=["county_fips"], static_reals=TFT_STATIC,
        time_varying_known_reals=TFT_KNOWN,
        time_varying_unknown_reals=TFT_UNKNOWN,
        target_normalizer=NaNLabelEncoder(), weight="w",
        allow_missing_timesteps=True, add_relative_time_idx=True,
    )
    # Each later split carries `enc` days of lead-in for context, but
    # min_prediction_idx keeps the predicted days inside the split. Without it
    # the training windows reappear as validation windows and early stopping
    # watches a loss that falls as the model memorises.
    others = {}
    for name in ("val", "test"):
        s, e = bounds[name]
        others[name] = TimeSeriesDataSet.from_dataset(
            train, d[(d["t"] >= s - enc) & (d["t"] <= e)],
            min_prediction_idx=s, predict=False, stop_randomization=True)
    return {"train": train, **others}


class EpochLine(pl.Callback):
    """One durable line per epoch: which epoch, the loss, how close to stopping.

    The progress bar answers "is it alive" but is filtered out of console.log,
    and max_epochs is only a ceiling because early stopping decides the real
    end, so neither the bar nor the archive would otherwise say how far along a
    fit got.

    The countdown is tracked here rather than read off EarlyStopping.wait_count:
    Lightning reorders callbacks, so EarlyStopping has not necessarily run when
    this fires, and reading it printed a stale patience that stayed at 5 through
    an epoch that had already regressed. Only `patience` is read from it, which
    is a constant. Mirrors EarlyStopping's default min_delta=0, i.e. ties count
    as no improvement.
    """

    def __init__(self):
        self.best = None
        self.wait = 0

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        vl = trainer.callback_metrics.get("val_loss")
        if vl is None:
            return
        vl = float(vl)
        if self.best is None or vl < self.best:
            self.best, self.wait = vl, 0
        else:
            self.wait += 1
        es = next((c for c in trainer.callbacks
                   if isinstance(c, EarlyStopping)), None)
        tail = f"  stop_in={es.patience - self.wait}" if es else ""
        print(f"    epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
              f"  val_loss={vl:.5f}  best={self.best:.5f}{tail}")


def tft_fit(haz):
    p = PARAMS["tft"]
    d = tft_frame(haz)
    ds = tft_datasets(d)
    TFT_DS[haz] = (d, ds)

    nw = p["workers"]
    dl_tr = ds["train"].to_dataloader(train=True, batch_size=p["batch"],
                                      num_workers=nw,
                                      persistent_workers=nw > 0)
    dl_va = ds["val"].to_dataloader(train=False, batch_size=p["batch"] * 2,
                                    num_workers=nw,
                                    persistent_workers=nw > 0)

    model = TemporalFusionTransformer.from_dataset(
        ds["train"], learning_rate=p["lr"], hidden_size=p["hidden"],
        attention_head_size=p["heads"], dropout=p["dropout"],
        hidden_continuous_size=p["hidden_continuous"],
        output_size=2, loss=CrossEntropy(), log_interval=-1,
    )

    prior = CKPT["tft"][haz]
    if PARAMS["warm_start"] and prior.exists():
        # strict=False forgives missing and unexpected keys but NOT shape
        # mismatches on keys that are present, so a checkpoint whose county set
        # or hidden size differs raises instead of partially loading. Drop the
        # tensors that cannot be copied and keep the rest.
        sd = torch.load(prior, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        cur = model.state_dict()
        fit = {k: v for k, v in sd.items()
               if k in cur and cur[k].shape == v.shape}
        dropped = len(sd) - len(fit)
        if not fit:
            print(f"  {prior.name} is a different architecture entirely; "
                  f"cold start")
        else:
            got = model.load_state_dict(fit, strict=False)
            WARM_PARENTS[f"tft:{haz}"] = {
                "file": prior.name, "sha256": _sha256(prior),
                "tensors_loaded": len(fit), "tensors_dropped": dropped}
            print(f"  warm start from {prior.name} (loaded={len(fit)} "
                  f"shape-mismatch={dropped} missing={len(got.missing_keys)})")

    ckpt_cb = ModelCheckpoint(dirpath=PATHS["scratch"], filename=f"tft_{haz}_best",
                              monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=p["epochs"], accelerator="auto", devices=1,
        gradient_clip_val=0.1, limit_train_batches=p["limit_train_batches"],
        limit_val_batches=p["limit_val_batches"],
        callbacks=[EarlyStopping(monitor="val_loss", patience=p["early"],
                                 mode="min"), ckpt_cb, EpochLine()],
        # On Colab a fit is 45+ minutes of total silence without this, with no
        # way to tell a live run from a hung one: cells queue, so no second cell
        # can poll while this one holds the kernel. _Tee keeps the redraw frames
        # out of console.log, so the bar costs nothing but screen.
        enable_progress_bar=sys.stdout.isatty() or ENV == "colab",
        # Was logger=False, which threw away the per-epoch loss curve -- the
        # only evidence that early stopping fired on a real plateau rather than
        # on noise. CSVLogger writes metrics.csv into the run directory.
        logger=CSVLogger(str(RUN_DIR / "tft_logs"), name=haz, version=""),
        enable_model_summary=False,
    )
    trainer.fit(model, dl_tr, dl_va)

    best = Path(ckpt_cb.best_model_path or "")
    if best.exists():
        model = TemporalFusionTransformer.load_from_checkpoint(best)
        trainer.save_checkpoint(prior)
    TFT_HIST[haz] = {"val_loss": float(ckpt_cb.best_model_score or float("nan")),
                     "epochs_run": trainer.current_epoch}
    return model


if HAVE_TORCH:
    for haz in PARAMS["hazards"]:
        print(f"=== tft {haz} ===")
        t0 = time.time()
        TFT[haz] = tft_fit(haz)
        print(f"  epochs={TFT_HIST[haz]['epochs_run']}  "
              f"best_val_loss={TFT_HIST[haz]['val_loss']:.5f}  "
              f"{time.time() - t0:.1f}s  -> {CKPT['tft'][haz].name}")
else:
    print(f"skipped: {TFT_SKIP_REASON}")


# %% ==========================================================================
# CELL 7 - TFT validation and test
# =============================================================================
def tft_score(haz, split_name):
    d, ds = TFT_DS[haz]
    dl = ds[split_name].to_dataloader(train=False, batch_size=1024,
                                      num_workers=PARAMS["tft"]["workers"])
    out = TFT[haz].predict(dl, mode="raw", return_index=True)
    logits = out.output.prediction  # (n_windows, 1, 2)
    p = torch.softmax(logits[:, 0, :], dim=-1)[:, 1].cpu().numpy()
    # Join on (county, day) rather than trusting row order: the dataloader
    # emits windows in its own sequence, and a silent misalignment here would
    # look like a model that simply does not work.
    idx = out.index[["county_fips", "t"]].copy()
    idx["p"] = p
    truth = d[["county_fips", "t", "target"]]
    j = idx.merge(truth, on=["county_fips", "t"], how="inner")
    return j.rename(columns={"target": "y"})[["county_fips", "t", "y", "p"]]


TFT_RESULT, TFT_PRED = {}, {}
if HAVE_TORCH and TFT:
    for haz in PARAMS["hazards"]:
        vf = tft_score(haz, "val")
        tf = tft_score(haz, "test")
        yv, pv = vf["y"].to_numpy(), vf["p"].to_numpy()

        # pos_weight makes the network see a base rate ~pos_weight times the
        # real one, so its raw probabilities run that far high. Isotonic on the
        # validation split maps them back. It is monotone, so the ranking --
        # and therefore AP -- is essentially untouched; what changes is whether
        # the number means anything as a probability.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pv, yv)
        tf["p_raw"] = tf["p"]
        tf["p"] = iso.predict(tf["p_raw"].to_numpy())

        yt, pt = tf["y"].to_numpy(), tf["p"].to_numpy()
        TFT_PRED[haz] = tf
        base = XGB_RESULT[haz]["baselines"]
        best = XGB_RESULT[haz]["best_baseline"]
        r = {"model": "temporal_fusion_transformer",
             "val": score(yv, pv), "test": score(yt, pt),
             "best_baseline": best, "best_baseline_ap": base[best],
             "epochs_run": TFT_HIST[haz]["epochs_run"],
             "best_val_loss": TFT_HIST[haz]["val_loss"],
             "n_test_pos": int(yt.sum()), "n_test_windows": int(len(yt)),
             "calibration": {
                 "method": "isotonic_on_val",
                 "pos_weight": PARAMS["tft"]["pos_weight"],
                 "observed_rate": float(yt.mean()),
                 "pred_mean_raw": float(tf["p_raw"].mean()),
                 "pred_mean_calibrated": float(pt.mean()),
                 "ap_raw": float(average_precision_score(
                     yt, tf["p_raw"].to_numpy())) if yt.sum() else None}}
        r["vs_best_baseline"] = (r["test"]["ap"] / base[best]
                                 if base[best] else None)
        r["beats_baseline"] = bool(r["test"]["ap"] > base[best])
        TFT_RESULT[haz] = r

        c = r["calibration"]
        print(f"=== tft {haz} ===")
        print(f"  calibration  mean pred {c['pred_mean_raw']:.5f} -> "
              f"{c['pred_mean_calibrated']:.5f}  (observed {c['observed_rate']:.5f})"
              f"   AP {c['ap_raw']:.5f} -> {r['test']['ap']:.5f}")
        print(f"  val   AP={r['val']['ap']:.5f}  lift={r['val']['lift']:.1f}x")
        print(f"  TEST  AP={r['test']['ap']:.5f}  lift={r['test']['lift']:.1f}x  "
              f"vs_best={r['vs_best_baseline']:.2f}x  "
              f"({r['n_test_pos']:,} pos / {r['n_test_windows']:,} windows)")
        if not r["beats_baseline"]:
            print(f"  WARNING: loses to {best}. Keep the baseline, not the model.")

    fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 3.6))
    for ax, haz in zip(np.atleast_1d(axes), PARAMS["hazards"]):
        yt = TFT_PRED[haz]["y"].to_numpy()
        pt = TFT_PRED[haz]["p"].to_numpy()
        pr, rc, _ = precision_recall_curve(yt, pt)
        ax.plot(rc, pr, color="#6a8f4f")
        ax.axhline(yt.mean(), ls="--", lw=1, color="#b5443a")
        ax.set_title(f"{haz} - TFT test PR curve")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / f"tft_eval_{TAG}.png", dpi=130)
    plt.close(fig)
    (PATHS["results"] / f"tft_{TAG}.json").write_text(
        json.dumps(TFT_RESULT, indent=2))
    print(f"wrote {PATHS['results'] / f'tft_{TAG}.json'}")
else:
    print("skipped: no trained TFT")


# %% ==========================================================================
# CELL 8 - results for both models side by side
# =============================================================================
rows = []
for haz in PARAMS["hazards"]:
    base_ap = XGB_RESULT[haz]["best_baseline_ap"]
    rows.append({"hazard": haz, "model": "baseline: " + XGB_RESULT[haz]["best_baseline"],
                 "test_AP": round(base_ap, 5), "lift": None, "vs_baseline": 1.0,
                 "verdict": "reference"})
    for name, res in (("xgboost", XGB_RESULT.get(haz)),
                      ("tft", TFT_RESULT.get(haz))):
        if not res:
            continue
        rows.append({
            "hazard": haz, "model": name,
            "test_AP": round(res["test"]["ap"], 5),
            "lift": round(res["test"]["lift"], 1) if res["test"]["lift"] else None,
            "vs_baseline": round(res["vs_best_baseline"], 2),
            "verdict": "keep" if res["beats_baseline"] else "LOSES TO BASELINE",
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))


# --- predicted vs observed, one table per hazard ----------------------------
THR_Q = 0.99  # alarm on the top 1% of scores, matching the dashboard's bands


def deviation_metrics(y, p, thr_q=THR_Q):
    """How far the predictions sit from what actually happened.

    RMSE and MAE are reported because they were asked for, but at this
    prevalence they are won by predicting zero everywhere -- which is why the
    always_zero column exists. Brier skill, calibration error and the
    contingency scores are the ones that survive a rare-event setting.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    n = len(y)
    obs = y.mean()
    err = p - y

    m = {
        "n_rows": n,
        "observed_events": int(y.sum()),
        "observed_rate": obs,
        "predicted_mean": p.mean(),
        "bias_pred_minus_obs": p.mean() - obs,
        "predicted_sd": p.std(),
        "observed_sd": y.std(),
        "MAE": np.abs(err).mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "max_abs_error": np.abs(err).max(),
        "brier": float(brier_score_loss(y, p)),
    }
    ref = float(brier_score_loss(y, np.full(n, obs)))
    m["brier_skill_vs_baserate"] = 1 - m["brier"] / ref if ref else np.nan
    m["log_loss"] = float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12),
                                   labels=[0, 1]))

    # Equal-count bins, not equal-width: almost every prediction lands in the
    # bottom width-bin, which would make the calibration error meaningless.
    order = np.argsort(p)
    ece = sum(len(b) / n * abs(p[b].mean() - y[b].mean())
              for b in np.array_split(order, 10) if len(b))
    m["calibration_error_ECE"] = float(ece)

    m["average_precision"] = (float(average_precision_score(y, p))
                              if y.sum() else np.nan)
    m["lift_over_rate"] = m["average_precision"] / obs if obs else np.nan

    # Rank the scores and alarm on the top k, rather than thresholding on a
    # quantile value. A constant predictor has every row at its own quantile,
    # so a value threshold would flag the whole panel and report POD = 1.
    k = max(1, int(round(n * (1 - thr_q))))
    top = np.argsort(-p, kind="stable")[:k]
    flag = np.zeros(n, dtype=bool)
    flag[top] = True
    thr = float(p[top[-1]])
    hits = int(np.sum(flag & (y == 1)))
    false_alarms = int(np.sum(flag & (y == 0)))
    misses = int(np.sum(~flag & (y == 1)))
    pod = hits / (hits + misses) if hits + misses else np.nan
    prec = hits / (hits + false_alarms) if hits + false_alarms else np.nan
    m.update({
        f"threshold_p{int(thr_q * 100)}": thr,
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "POD_recall": pod,
        "FAR_false_alarm_ratio": (false_alarms / (hits + false_alarms)
                                  if hits + false_alarms else np.nan),
        "precision": prec,
        "CSI_critical_success": (hits / (hits + misses + false_alarms)
                                 if hits + misses + false_alarms else np.nan),
        "F1": (2 * prec * pod / (prec + pod)
               if prec and pod and prec + pod else np.nan),
    })
    return m


def aligned(haz):
    """XGBoost and TFT on exactly the same rows, so the columns are comparable.

    The TFT scores fewer rows -- every window needs `encoder_days` of history
    first -- so the comparison is restricted to the intersection.
    """
    f = XGB_PRED[haz].rename(columns={"p": "xgboost"})
    if haz in TFT_PRED:
        t = TFT_PRED[haz].rename(columns={"p": "tft"}).drop(columns=["y"])
        f = f.merge(t, on=["county_fips", "t"], how="inner")
    return f


DEVIATION = {}
for haz in PARAMS["hazards"]:
    f = aligned(haz)
    y = f["y"].to_numpy()
    # Baselines are absent here on purpose: a rain ratio is not a probability,
    # so its RMSE and Brier score would not mean the same thing as a model's.
    cols = {"always_zero": np.zeros(len(y)),
            "base_rate": np.full(len(y), S("train", haz)[f"y_{haz}"].mean())}
    for name in ("xgboost", "tft"):
        if name in f.columns:
            cols[name] = f[name].to_numpy()

    tbl = pd.DataFrame({k: deviation_metrics(y, v) for k, v in cols.items()})
    tbl.index.name = f"{haz}: predicted vs observed"
    DEVIATION[haz] = tbl

    print(f"\n=== {haz} - predicted vs observed, test period "
          f"({len(y):,} county-days, {int(y.sum())} real onsets) ===")
    print(tbl.to_string(float_format=lambda v: f"{v:,.6g}"))
    tbl.to_csv(PATHS["results"] / f"deviation_{haz}_{TAG}.csv")

# --- reliability: does a predicted 1-in-200 day happen 1 in 200 times? ------
fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 4))
for ax, haz in zip(np.atleast_1d(axes), PARAMS["hazards"]):
    f = aligned(haz)
    y = f["y"].to_numpy()
    for name, col in (("xgboost", "#4878a8"), ("tft", "#6a8f4f")):
        if name not in f.columns:
            continue
        p = f[name].to_numpy()
        order = np.argsort(p)
        xs, ys = [], []
        for b in np.array_split(order, 12):
            if len(b):
                xs.append(p[b].mean())
                ys.append(y[b].mean())
        ax.plot(xs, ys, "o-", ms=4, color=col, label=name)
    lim = [1e-6, 1]
    ax.plot(lim, lim, ls="--", lw=1, color="#888", label="perfect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{haz} - reliability")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"reliability_{TAG}.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 3.8))
for ax, haz in zip(np.atleast_1d(axes), PARAMS["hazards"]):
    for name, pred, col in (("xgboost", XGB_PRED.get(haz), "#4878a8"),
                            ("tft", TFT_PRED.get(haz), "#6a8f4f")):
        if pred is None:
            continue
        y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rc, pr, color=col, label=f"{name}  AP={ap:.4f}")
    y = S("test", haz)[f"y_{haz}"].to_numpy()
    ax.axhline(y.mean(), ls="--", lw=1, color="#b5443a",
               label=f"prevalence {y.mean():.5f}")
    ax.set_title(f"{haz} - test, both models")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"compare_{TAG}.png", dpi=130)
plt.close(fig)

final = {"tag": TAG, "env": ENV, "host": platform.node(),
         "params": PARAMS, "splits": split_table.to_dict("records"),
         "summary": summary.to_dict("records"),
         "deviation": {h: t.to_dict() for h, t in DEVIATION.items()},
         "xgboost": XGB_RESULT, "tft": TFT_RESULT}
out = PATHS["results"] / f"results_{TAG}.json"
out.write_text(json.dumps(final, indent=2, default=str))
summary.to_csv(PATHS["results"] / f"summary_{TAG}.csv", index=False)

print(f"\nwrote {out}")
print(f"plots in {PATHS['plots']}")
for f in sorted(PATHS["plots"].glob(f"*_{TAG}.png")):
    print(f"  {f.name}")


# %% ==========================================================================
# CELL 9 - skill vs lead time: one model per exact lead day
# =============================================================================
# The headline models pool t+1..t+3 into one "onset within 3 days" target,
# which cannot tell a reviewer whether day 3 carries any signal at all. One
# booster per exact lead day answers that. XGBoost only: the TFT costs ~50 min
# per hazard, so six of them does not fit a session, and the decay shape being
# asked about is a property of atmospheric predictability, not of the
# architecture. A flat curve here would mean the label leaks the present.
LEAD_RESULT, lead_rows = {}, []


def lead_row(haz, lead, pos, prev, ap, lift, base, base_ap, vs):
    return {"hazard": haz, "lead": lead, "test_pos": pos,
            "prevalence": round(prev, 6), "test_AP": round(ap, 5),
            "lift": round(lift, 1) if lift else None, "baseline": base,
            "baseline_AP": round(base_ap, 5),
            "vs_baseline": round(vs, 2) if vs else None}


for haz in PARAMS["hazards"]:
    rp = XGB_RESULT[haz]
    lead_rows.append(lead_row(
        haz, "1-3 pooled", rp["n_test_pos"], rp["test"]["prevalence"],
        rp["test"]["ap"], rp["test"]["lift"], rp["best_baseline"],
        rp["best_baseline_ap"], rp["vs_best_baseline"]))

    for k in PARAMS["leads"]:
        key = f"{haz}_l{k}"
        print(f"=== xgboost {haz} lead {k} day ===")
        t0 = time.time()
        model = xgb_fit(key)
        st = S("test", key)
        yt = st[f"y_{key}"].to_numpy()
        pt = model.predict_proba(st[FEATS])[:, 1]
        sc = score(yt, pt)
        base = baselines(key)
        best = max(base, key=base.get)
        vs = sc["ap"] / base[best] if base[best] else None

        LEAD_RESULT[key] = {
            "hazard": haz, "lead_days": k, "test": sc,
            "best_baseline": best, "best_baseline_ap": base[best],
            "vs_best_baseline": vs, "baselines": base,
            "best_iteration": int(model.best_iteration),
            "n_train_pos": int(S("train", key)[f"y_{key}"].sum()),
            "n_test_pos": int(yt.sum()), "fit_seconds": round(time.time() - t0, 1),
            "top_features": [[f, round(float(v), 5)] for f, v in sorted(
                zip(FEATS, model.feature_importances_), key=lambda x: -x[1])[:6]]}

        lead_rows.append(lead_row(
            haz, f"+{k} day", int(yt.sum()), sc["prevalence"], sc["ap"],
            sc["lift"], best, base[best], vs))
        print(f"  best_iteration={model.best_iteration}  "
              f"{time.time() - t0:.1f}s")
        print(f"  TEST  AP={sc['ap']:.5f}  lift={sc['lift']:.1f}x  "
              f"vs_best={vs:.2f}x  ({int(yt.sum()):,} pos / {len(yt):,} rows)")
        if vs is not None and vs <= 1:
            print(f"  WARNING: loses to {best} at this lead.")

lead_table = pd.DataFrame(lead_rows)
print("\n=== skill vs lead time (XGBoost, test period) ===")
print(lead_table.to_string(index=False))

fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 3.6))
for ax, haz in zip(np.atleast_1d(axes), PARAMS["hazards"]):
    ks = PARAMS["leads"]
    got = [LEAD_RESULT[f"{haz}_l{k}"] for k in ks]
    ax.plot(ks, [r["test"]["lift"] for r in got], "o-", color="#4878a8",
            label="xgboost")
    # Baselines are plotted as lift too, so the gap on the chart is the gap in
    # the only number that survives a prevalence that changes with lead.
    ax.plot(ks, [r["best_baseline_ap"] / r["test"]["prevalence"] for r in got],
            "s--", color="#b5443a", label="best single feature")
    ax.axhline(1, lw=1, color="#888", label="prevalence")
    ax.set_xticks(ks)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{haz} - lift over prevalence by lead")
    ax.set_xlabel("lead (days ahead)")
    ax.set_ylabel("AP / prevalence")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PATHS["plots"] / f"lead_{TAG}.png", dpi=130)
plt.close(fig)

lead_table.to_csv(PATHS["results"] / f"leads_{TAG}.csv", index=False)
(PATHS["results"] / f"leads_{TAG}.json").write_text(
    json.dumps(LEAD_RESULT, indent=2, default=str))
print(f"\nwrote {PATHS['results'] / f'leads_{TAG}.csv'}")
print(f"wrote {PATHS['plots'] / f'lead_{TAG}.png'}")


# %% ==========================================================================
# CELL 10 - the same lead sweep for the TFT
# =============================================================================
# One TFT per exact lead day, matching the XGBoost table row for row. The
# alternative was a single model with max_prediction_length=3, reading the
# three decoder steps -- cheaper and the more natural use of a sequence
# decoder, but it answers a different question: a joint forecast rather than
# six independently fitted models. Symmetry with the XGBoost table is what
# makes the architecture comparison per lead mean anything, so this trains six.
#
# It costs roughly 6 x the single-hazard TFT time. Set CRISK_TFT_LEADS=0 to
# skip while iterating on anything else in the pipeline.
TFT_LEAD_RESULT = {}
RUN_TFT_LEADS = (HAVE_TORCH and PARAMS["leads"]
                 and os.environ.get("CRISK_TFT_LEADS", "1") == "1")

if not RUN_TFT_LEADS:
    print(f"TFT lead sweep skipped ({TFT_SKIP_REASON or 'CRISK_TFT_LEADS=0'})")
else:
    pooled = [TFT_HIST[h]["epochs_run"] for h in TFT_HIST if h in PARAMS["hazards"]]
    print(f"TFT lead sweep: {len(PARAMS['hazards']) * len(PARAMS['leads'])} "
          f"models to fit"
          + (f"; the pooled models ran {pooled} epochs for reference"
             if pooled else ""))

    # Cell 6 left a frame and a TimeSeriesDataSet per hazard in TFT_DS, each a
    # full copy of the panel's TFT columns. Holding those while fitting a
    # seventh was enough to get the process OOM-killed on a 14 GB box at smoke
    # size; the full panel is four times larger. Cells 7 and 8 already took
    # what they need into TFT_PRED, and the weights are on disk in CKPT, so
    # nothing below this point reads either dict.
    for h in list(TFT_DS):
        TFT_DS.pop(h, None)
        TFT.pop(h, None)
    gc.collect()

    for haz in PARAMS["hazards"]:
        for k in PARAMS["leads"]:
            key = f"{haz}_l{k}"
            print(f"=== tft {haz} lead {k} day ===")
            t0 = time.time()
            TFT[key] = tft_fit(key)
            print(f"  epochs={TFT_HIST[key]['epochs_run']}  "
                  f"best_val_loss={TFT_HIST[key]['val_loss']:.5f}  "
                  f"{time.time() - t0:.1f}s")

            vf, tf = tft_score(key, "val"), tft_score(key, "test")
            yv, pv = vf["y"].to_numpy(), vf["p"].to_numpy()
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv, yv)
            tf["p_raw"] = tf["p"]
            tf["p"] = iso.predict(tf["p_raw"].to_numpy())
            yt, pt = tf["y"].to_numpy(), tf["p"].to_numpy()

            # Same baselines the XGBoost at this lead was judged against, so
            # the two vs_baseline columns are comparable.
            base = LEAD_RESULT[key]["baselines"]
            best = LEAD_RESULT[key]["best_baseline"]
            r = {"hazard": haz, "lead_days": k,
                 "model": "temporal_fusion_transformer",
                 "val": score(yv, pv), "test": score(yt, pt),
                 "best_baseline": best, "best_baseline_ap": base[best],
                 "epochs_run": TFT_HIST[key]["epochs_run"],
                 "best_val_loss": TFT_HIST[key]["val_loss"],
                 "n_test_pos": int(yt.sum()), "n_test_windows": int(len(yt)),
                 "fit_seconds": round(time.time() - t0, 1),
                 "calibration": {
                     "method": "isotonic_on_val",
                     "pos_weight": PARAMS["tft"]["pos_weight"],
                     "observed_rate": float(yt.mean()),
                     "pred_mean_raw": float(tf["p_raw"].mean()),
                     "pred_mean_calibrated": float(pt.mean()),
                     "ap_raw": float(average_precision_score(
                         yt, tf["p_raw"].to_numpy())) if yt.sum() else None}}
            r["vs_best_baseline"] = (r["test"]["ap"] / base[best]
                                     if base[best] else None)
            r["beats_baseline"] = bool(r["test"]["ap"] > base[best])
            TFT_LEAD_RESULT[key] = r
            TFT_PRED[key] = tf

            print(f"  TEST  AP={r['test']['ap']:.5f}  "
                  f"lift={r['test']['lift']:.1f}x  "
                  f"vs_best={r['vs_best_baseline']:.2f}x  "
                  f"({r['n_test_pos']:,} pos / {r['n_test_windows']:,} windows)")
            if not r["beats_baseline"]:
                print(f"  WARNING: loses to {best} at this lead.")

            # Each entry holds a full copy of the panel's TFT columns. Eight
            # targets of that at once is enough to exhaust a Colab runtime, and
            # nothing after this point needs the frame or the weights.
            TFT_DS.pop(key, None)
            TFT.pop(key, None)
            gc.collect()
            if HAVE_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    both_rows = []
    for haz in PARAMS["hazards"]:
        for k in PARAMS["leads"]:
            key = f"{haz}_l{k}"
            x, t = LEAD_RESULT[key], TFT_LEAD_RESULT[key]
            both_rows.append({
                "hazard": haz, "lead": f"+{k} day",
                "prevalence": round(x["test"]["prevalence"], 6),
                "baseline": x["best_baseline"],
                "baseline_AP": round(x["best_baseline_ap"], 5),
                "xgb_AP": round(x["test"]["ap"], 5),
                "xgb_vs_base": round(x["vs_best_baseline"], 2),
                "tft_AP": round(t["test"]["ap"], 5),
                "tft_vs_base": round(t["vs_best_baseline"], 2),
                "tft_epochs": t["epochs_run"],
                # The TFT scores fewer rows: every window needs 30 days of
                # encoder history, so the first month of each split drops out.
                "tft_windows": t["n_test_windows"]})
    lead_both = pd.DataFrame(both_rows)
    print("\n=== skill vs lead time, both models (test period) ===")
    print(lead_both.to_string(index=False))

    fig, axes = plt.subplots(1, len(PARAMS["hazards"]), figsize=(11, 3.6))
    for ax, haz in zip(np.atleast_1d(axes), PARAMS["hazards"]):
        ks = PARAMS["leads"]
        xs = [LEAD_RESULT[f"{haz}_l{k}"] for k in ks]
        ts = [TFT_LEAD_RESULT[f"{haz}_l{k}"] for k in ks]
        ax.plot(ks, [r["test"]["lift"] for r in xs], "o-", color="#4878a8",
                label="xgboost")
        ax.plot(ks, [r["test"]["lift"] for r in ts], "^-", color="#6a8f4f",
                label="tft")
        ax.plot(ks, [r["best_baseline_ap"] / r["test"]["prevalence"] for r in xs],
                "s--", color="#b5443a", label="best single feature")
        ax.axhline(1, lw=1, color="#888", label="prevalence")
        ax.set_xticks(ks)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{haz} - lift over prevalence by lead")
        ax.set_xlabel("lead (days ahead)")
        ax.set_ylabel("AP / prevalence")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / f"lead_both_{TAG}.png", dpi=130)
    plt.close(fig)

    lead_both.to_csv(PATHS["results"] / f"leads_both_{TAG}.csv", index=False)
    (PATHS["results"] / f"tft_leads_{TAG}.json").write_text(
        json.dumps(TFT_LEAD_RESULT, indent=2, default=str))
    print(f"\nwrote {PATHS['results'] / f'leads_both_{TAG}.csv'}")
    print(f"wrote {PATHS['plots'] / f'lead_both_{TAG}.png'}")


# %% ==========================================================================
# CELL 11 - seal the run directory
# =============================================================================
# The checkpoints are copied in rather than referenced, so the directory stays
# a complete record even after the next run overwrites models/. They are a few
# MB; a result that cannot be re-scored is not much of a proof.
_mdir = RUN_DIR / "models"
_mdir.mkdir(parents=True, exist_ok=True)
model_files = {}
for kind, paths in CKPT.items():
    for key, src in paths.items():
        if not src.exists():
            continue
        shutil.copy2(src, _mdir / src.name)
        model_files[src.name] = {"sha256": _sha256(src), "bytes": src.stat().st_size}

inventory = sorted(str(p.relative_to(RUN_DIR))
                   for p in RUN_DIR.rglob("*") if p.is_file())
final_manifest = manifest_update(
    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    wall_seconds=round(time.time() - RUN_T0, 1),
    headline={"xgboost": {h: round(r["test"]["ap"], 5)
                          for h, r in XGB_RESULT.items()},
              "tft": {h: round(r["test"]["ap"], 5)
                      for h, r in TFT_RESULT.items()},
              "xgb_leads": {k: round(r["test"]["ap"], 5)
                            for k, r in LEAD_RESULT.items()},
              "tft_leads": {k: round(r["test"]["ap"], 5)
                            for k, r in TFT_LEAD_RESULT.items()}},
    models=model_files, inventory=inventory,
    warm_parents=WARM_PARENTS,
    self_contained=not WARM_PARENTS)

if RUN_SEAL_TO is not None:
    # One zip, written once, then hashed on both sides. A mismatch or a missing
    # file means the archive did not reach Drive, and the run must say so
    # rather than print a path that will be empty tomorrow.
    RUN_SEAL_TO.mkdir(parents=True, exist_ok=True)
    local_zip = Path(shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR))
    drive_zip = RUN_SEAL_TO / local_zip.name
    shutil.copy2(local_zip, drive_zip)
    want = _sha256(local_zip)
    got = _sha256(drive_zip) if drive_zip.exists() else None
    shutil.copy2(MANIFEST, RUN_SEAL_TO / f"{RUN_ID}.manifest.json")
    if got == want:
        print(f"\nsealed to Drive: {drive_zip}  "
              f"({drive_zip.stat().st_size / 1e6:.1f} MB, sha256 verified)")
        print("DOWNLOAD THAT ZIP. Drive has already lost one run's results.")
    else:
        print(f"\nWARNING: the archive did not land on Drive intact "
              f"(want {want[:12]}, got {got and got[:12]}).")
        print(f"Download it from the Colab file browser instead: {local_zip}")

print(f"\n=== run sealed: {RUN_DIR} ===")
print(f"wall {final_manifest['wall_seconds']}s   "
      f"{len(inventory)} files   git {final_manifest['git']['commit'][:8] or 'n/a'}"
      f"{' (dirty)' if final_manifest['git']['dirty'] else ''}")
if WARM_PARENTS:
    print(f"NOT self-contained: warm-started from {len(WARM_PARENTS)} prior "
          f"checkpoint(s). Re-run with CRISK_COLD=1 to report these numbers.")
else:
    print("cold start: reproducible from this directory and the panel hash alone")
for f in inventory:
    print(f"  {f}")
