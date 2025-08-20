#!/bin/bash
#SBATCH --job-name=cardio_cuda
#SBATCH --output=log_cuda_%j.out
#SBATCH --error=log_cuda_%j.err
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

echo "Job started on $(hostname)"
echo "CUDA Version:"
/usr/bin/nvcc --version

cd /scratch.hpc/enrico.strangio/Cardiomegaly_Classifier

# Compila includendo cnpy
/usr/bin/nvcc -O2 -std=c++17 -arch=sm_75 -lineinfo src/cardiomegaly_cuda.cu src/cnpy.cpp -I src -lz -o cardiomegaly_cuda 

# Controlla se la compilazione è andata a buon fine
if [ ! -f cardiomegaly_cuda ]; then
  echo "Errore: compilazione fallita"
  exit 1
fi

# Info GPU (sul nodo GPU allocato)
echo "GPU info:"
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Prepara output
mkdir -p results


# Sanity check file input
test -d data/images || { echo "Manca data/images"; exit 1; }
test -f data/weights/static_weights_224x224.npy || { echo "Manca data/weights/static_weights_224x224.npy"; exit 1; }
test -f data/labels.npy || { echo "Manca data/labels.npy"; exit 1; }

# Esegui
echo "[RUN] Inference..."
srun ./cardiomegaly_cuda \
  --images-dir data/images \
  --weights data/weights/static_weights_224x224.npy \
  --labels data/labels.npy \
  --auto-th \
  --out-csv results/test.csv \
  --out-metrics results/test.txt
echo "Job completed."
