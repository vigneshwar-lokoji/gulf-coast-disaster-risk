#!/bin/bash
#SBATCH --job-name=crisk-env
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Run once. Builds the conda env on a compute node, because the cluster policy
# forbids running code on the login node and a conda solve is not free.
#
# CPU torch by default: ~200 MB instead of ~2.5 GB, and /home is the only
# storage available until /work/gpinnu exists. Re-run with CRISK_TORCH=cu121
# once GPU QOS is granted.

set -euo pipefail

CRISK_ENV="${CRISK_ENV:-$HOME/.conda/envs/crisk}"
CRISK_TORCH="${CRISK_TORCH:-cpu}"

module purge
module load miniconda3/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "$CRISK_ENV" ]; then
  echo "env already exists at $CRISK_ENV"
else
  conda create -y -p "$CRISK_ENV" python=3.11
fi

conda activate "$CRISK_ENV"

python -m pip install --no-cache-dir \
  'numpy>=1.26,<2.2' 'pandas>=2.2,<3' 'pyarrow>=16' \
  'scikit-learn>=1.5' 'xgboost>=2.1' 'matplotlib>=3.8'

python -m pip install --no-cache-dir \
  --index-url "https://download.pytorch.org/whl/${CRISK_TORCH}" torch

python -m pip install --no-cache-dir \
  'pytorch-forecasting>=1.3' 'lightning>=2.2'

echo "=== versions ==="
python - <<'PY'
import importlib
for name in ("numpy", "pandas", "pyarrow", "sklearn", "xgboost",
             "matplotlib", "torch", "lightning", "pytorch_forecasting"):
    try:
        m = importlib.import_module(name)
        print(f"{name:22s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:22s} MISSING ({exc})")
import torch
print(f"cuda available         {torch.cuda.is_available()}")
PY
du -sh "$CRISK_ENV"
echo "env ready at $CRISK_ENV"
