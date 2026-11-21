#!/bin/bash
#SBATCH --job-name=brain_rdm
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/arnav.shakya/rsa_work/logs/%x_%j.out
/scratch/arnav.shakya/conda/envs/rsa/bin/python src/brain_rdm.py sub-01
