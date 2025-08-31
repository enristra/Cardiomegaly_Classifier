#!/bin/bash -l
#SBATCH --job-name=prof_vec4_256
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/src
#SBATCH --output=../logs/profile_vec4_256_%j.out
#SBATCH --error=../logs/profile_vec4_256_%j.err

nvidia-smi || true

# Profilazione Nsight Compute con blockDim=256
ncu --set full --target-processes all \
    ./cardiomegaly_cuda \
    --images-dir ../data/images \
    --weights ../data/weights/static_weights_224x224.npy \
    --labels  ../data/labels.npy \
    --kernel vec4 --block 256 --limit 1000 --auto-th \
    > ../results/vec4_profile_256.txt
