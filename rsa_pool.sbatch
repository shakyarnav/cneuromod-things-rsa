#!/bin/bash
#SBATCH --job-name=rsa_pool
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
export OMP_NUM_THREADS=4
cd /scratch/arnav.shakya/rsa_work
/scratch/arnav.shakya/conda/envs/rsa/bin/python rsa_pool.py sub-01 sub-02 sub-03 sub-06
