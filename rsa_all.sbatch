#!/bin/bash
#SBATCH --job-name=rsa_all
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
export OMP_NUM_THREADS=4
/scratch/arnav.shakya/conda/envs/rsa/bin/python /scratch/arnav.shakya/rsa_work/brain_rdm.py sub-01 sub-02 sub-03 sub-06
/scratch/arnav.shakya/conda/envs/rsa/bin/python /scratch/arnav.shakya/rsa_work/rsa.py sub-01 sub-02 sub-03 sub-06
