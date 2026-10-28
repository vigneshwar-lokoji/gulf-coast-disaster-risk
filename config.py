"""Shared configuration. Set DATA_DIR before running anything else."""

import os
from pathlib import Path

# Colab reads from Drive; local runs fall back to the copy on disk. Mount first:
#     from google.colab import drive; drive.mount('/content/drive')
COLAB_DATA_DIR = Path("/content/drive/MyDrive/ganesh climate data")
LOCAL_DATA_DIR = Path("/home/vigneshwar-lokoji/ganesh project")

DATA_DIR = COLAB_DATA_DIR if COLAB_DATA_DIR.exists() else LOCAL_DATA_DIR

# Env var wins if present, so the path above can stay as-is during local runs.
if os.environ.get("CLIMATE_DATA_DIR"):
    DATA_DIR = Path(os.environ["CLIMATE_DATA_DIR"])

OUT_DIR = Path("./artifacts")

FILES = {
    "weather": "gulf_coast_historical.parquet",
    "fema": "fema_disaster_declarations_gulf_coast (1).csv",
    "svi": "svi_gulf_coast_counties (1).csv",
    "mapping": "station_county_mapping (1).csv",
    "stations": "gulf_coast_stations_official.csv",
}

START_YEAR = 1990

# Climatology baseline is fit on 1990-2010 only and frozen, so that anomaly
# features for later years never see their own distribution. Widening this to
# the full record leaks test-period information into the training features.
CLIMATOLOGY_YEARS = (1990, 2010)

LEADS = [0, 1, 2, 3]

# Temporal splits are contiguous blocks, never random — adjacent months share
# weather state, so a shuffled split leaks across the boundary.
SPLIT_TRAIN = (1990, 2014)
SPLIT_VAL = (2015, 2019)
SPLIT_TEST = (2020, 2026)

STORM_TYPES = {
    "Hurricane",
    "Severe Storm",
    "Tropical Storm",
    "Tornado",
    "Coastal Storm",
}

# Not weather. 'Biological' is overwhelmingly COVID-19 declarations.
DROP_INCIDENT_TYPES = {"Biological", "Human Cause", "Toxic Substances"}

WET_DAY_MM = 1.0
HEAVY_DAY_MM = 25.4
