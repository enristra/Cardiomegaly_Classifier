#!/bin/bash -l
#SBATCH --job-name=cardio_infer
#SBATCH --partition=l40,rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/src
#SBATCH --output=../logs/infer_%j.out
#SBATCH --error=../logs/infer_%j.err

set -euo pipefail



echo "Node: $(hostname) | Date: $(date)"
nvidia-smi || true

# Esecuzione vera e propria
srun ./cardiomegaly_cuda \
  --images-dir ../data/images \
  --weights ../data/weights/static_weights_224x224.npy \
  --labels ../data/labels.npy \
  --auto-th \
  --kernel naive \
  --out-csv ../results/gpu_scores.csv \
  --out-metrics ../results/gpu_metrics.txt

echo "Job completed."
