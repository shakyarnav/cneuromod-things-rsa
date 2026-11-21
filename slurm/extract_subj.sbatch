#!/bin/bash
#SBATCH --job-name=roi_extract
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
/scratch/arnav.shakya/conda/envs/rsa/bin/python src/extract_roi_betas.py "$@"
