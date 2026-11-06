#!/bin/bash
#SBATCH --job-name=pepgate-embed
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/embed_%j.out
#SBATCH --error=slurm/logs/embed_%j.err

set -euo pipefail

PROJECT=/ocean/projects/cis260266p/halhosan/PepGate
ENV=/ocean/projects/cis260266p/halhosan/envs/pepgate

module load anaconda3/2024.10-1
cd "$PROJECT"

export HF_HOME=/ocean/projects/cis260266p/halhosan/.cache/huggingface
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Job $SLURM_JOB_ID on $(hostname), start $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

"$ENV/bin/python" -u src/models/embed.py --encoder peptideclm1
"$ENV/bin/python" -u src/models/embed.py --encoder helmbert

echo "Finished: $(date)"
