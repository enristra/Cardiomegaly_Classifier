#!/bin/bash
#SBATCH --job-name=extract_train_224
#SBATCH --partition=l40                 # coda l40; niente --gres -> CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/logs/extract_%j.out
#SBATCH --error=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/logs/extract_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=enrico.strangio@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/src

set -euo pipefail
echo "Node: $(hostname)  PWD: $(pwd)  Date: $(date)"

PROJECT_ROOT="/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier"
VENV_DIR="$PROJECT_ROOT/cardio_env"

# Metti tutte le cache in scratch (non in HOME)
export MEDMNIST_ROOT="/scratch.hpc/enrico.strangio/medmnist_cache"
export XDG_CACHE_HOME="/scratch.hpc/enrico.strangio/.cache"
mkdir -p "$MEDMNIST_ROOT" "$XDG_CACHE_HOME" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/data/images"

# Attiva l'ambiente
source "$VENV_DIR/bin/activate"

# (opzionale) assicurati i pacchetti, senza cache
python - <<'PY'
try:
    import medmnist, numpy, PIL
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "medmnist", "numpy", "pillow"])
PY

# Avvia lo script di estrazione (usa path assoluti già dentro lo script)
python extract_files.py

echo "DONE at $(date)"

