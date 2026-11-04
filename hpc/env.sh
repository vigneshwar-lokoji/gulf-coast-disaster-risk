#!/bin/bash
# Sourced by every job script. Keeps the module/conda incantation in one place
# so a change to the cluster's module names is a one-line fix.

set -euo pipefail

export CRISK_ROOT="${CRISK_ROOT:-$HOME/crisk}"
export CRISK_ENV="${CRISK_ENV:-$HOME/.conda/envs/crisk}"

module purge
module load miniconda3/24.1.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CRISK_ENV"

# BLAS threads fight XGBoost's own pool for the same cores and make the job
# slower than single-threaded. Let Slurm's allocation be the only source.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$CRISK_ROOT"
