#!/bin/bash
#SBATCH --job-name=cardio_cuda
#SBATCH --output=log_cuda_%j.out
#SBATCH --error=log_cuda_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

module load cuda/12.2
module load gcc/11.2

cd /scratch.hpc/enrico.strangio/Cardiomegaly_Classifier

# Compila il file
nvcc -O2 -o cardiomegaly_cuda cardiomegaly_cuda.cu -lcnpy -lz

# Esegui
./cardiomegaly_cuda
