#!/bin/bash
#SBATCH --job-name=cardio_cuda
#SBATCH --output=log_cuda_%j.out
#SBATCH --error=log_cuda_%j.err
#SBATCH --partition=rtx2080       # Partizione con RTX 2080 Ti
#SBATCH --gres=gpu:1              # Richiede 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

echo "Job started on $(hostname)"
echo "CUDA Version:"
nvcc --version

module purge
module load cuda/12.2
module load gcc/11.2

cd /scratch.hpc/enrico.strangio/Cardiomegaly_Classifier

# Compila il file CUDA
nvcc -O2 cardiomegaly_cuda.cu -o cardiomegaly_cuda -lcnpy -lz

# Mostra info GPU
echo "GPU Info before execution:"
nvidia-smi

# Esegui il programma
./cardiomegaly_cuda

# Mostra info GPU dopo
echo "GPU Info after execution:"
nvidia-smi

echo "Job completed."
