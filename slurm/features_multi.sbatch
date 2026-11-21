#!/bin/bash
#SBATCH --job-name=dnn_multi
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
export TORCH_HOME=/scratch/arnav.shakya/rsa_work/torch_home
export OMP_NUM_THREADS=4
export SIMCLR_CKPT=$(ls /scratch/arnav.shakya/rsa_work/hf_home/hub/*/snapshots/*/checkpoint.pth)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
/scratch/arnav.shakya/conda/envs/dnn/bin/python extract_features_multi.py "$@"
