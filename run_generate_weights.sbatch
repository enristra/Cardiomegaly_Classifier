#!/bin/bash
#SBATCH --job-name=gen_weights
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=rtx2080        # oppure l40 se vuoi la GPU L40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --chdir=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/

# Attiva ambiente virtuale
source ~/venv/bin/activate

# (Facoltativo) installa jupyter se non già presente
# pip install --no-cache-dir jupyter

# Converte il notebook in .py (eseguito solo se generate_weights.py non esiste)
if [ ! -f generate_weights.py ]; then
    jupyter nbconvert --to script generate_weights.ipynb
fi

# Esegue lo script Python
python generate_weights.py
