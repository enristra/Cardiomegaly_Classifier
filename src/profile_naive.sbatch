#!/bin/bash -l
#SBATCH --job-name=prof_naive
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/src
#SBATCH --output=../logs/profile_naive_%j.out
#SBATCH --error=../logs/profile_naive_%j.err


nvidia-smi || true

ncu --set full --target-processes all \
    ./cardiomegaly_cuda \
    --images-dir ../data/images \
    --weights ../data/weights/static_weights_224x224.npy \
    --labels  ../data/labels.npy \
    --kernel naive --limit 1000 --auto-th \
    > ../results/naive_profile.txt


# ncu --set full --target-processes all \
#     --export ../results/naive_profile.ncu-rep \
#     ./cardiomegaly_cuda \
#     --images-dir ../data/images \
#     --weights ../data/weights/static_weights_224x224.npy \
#     --labels  ../data/labels.npy \
#     --kernel naive --limit 1000 --auto-th \
#     > ../results/naive_profile.txt
