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
/usr/bin/nvcc -O2 src/cardiomegaly_cuda.cu src/cnpy.cpp -o cardiomegaly_cuda -lz

# Controlla se la compilazione è andata a buon fine
if [ ! -f cardiomegaly_cuda ]; then
  echo "Errore: compilazione fallita"
  exit 1
fi

echo "GPU Info before execution:"
nvidia-smi

# Esegui il programma
./cardiomegaly_cuda

echo "Job completed."
