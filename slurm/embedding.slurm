#!/bin/bash
#SBATCH -J pepgate-embed-study
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/embedstudy_%j.out
#SBATCH -e slurm/logs/embedstudy_%j.err

cd /ocean/projects/cis260266p/halhosan/PepGate
source activate pepgate
export OMP_NUM_THREADS=8
python analysis/embedding_study.py --encoders rdkit
