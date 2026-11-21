#!/bin/bash
#SBATCH --job-name=rsa
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
export OMP_NUM_THREADS=4
/scratch/arnav.shakya/conda/envs/rsa/bin/python /scratch/arnav.shakya/rsa_work/rsa.py sub-01
