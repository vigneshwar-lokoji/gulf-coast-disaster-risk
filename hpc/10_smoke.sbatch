#!/bin/bash
#SBATCH --job-name=crisk-smoke
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Exercises the whole chain -- panel build, splits, XGBoost, TFT, comparison --
# on a deliberately small slice. The point is to prove the plumbing; the AP
# numbers it prints are expected to be poor.

export CRISK_ROOT="${CRISK_ROOT:-$HOME/crisk}"
source "$CRISK_ROOT/hpc/env.sh"

export CLIMATE_DATA_DIR="$CRISK_ROOT/data"
export CLIMATE_OUT_DIR="$CRISK_ROOT/artifacts"
export CRISK_SMOKE=1

# Short window keeps the build quick. Climatology is pinned to the training
# years so the smoke config is not quietly leaky.
export PANEL_START=2015
export PANEL_CLIM_END=2019

echo "=== node ==="
hostname; nproc; free -g | head -2
echo

if [ ! -f "$CLIMATE_OUT_DIR/daily_panel.parquet" ]; then
  echo "=== panel build ==="
  time python build_daily.py
  echo
fi

echo "=== pipeline ==="
time python pipeline.py

echo
echo "SMOKE OK"
