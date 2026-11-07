#!/bin/bash
#SBATCH --job-name=pepgate-embed-cpu
#SBATCH --account=cis260266p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/embedcpu_%j.out
#SBATCH --error=slurm/logs/embedcpu_%j.err

set -euo pipefail

PROJECT=/ocean/projects/cis260266p/halhosan/PepGate
ENV=/ocean/projects/cis260266p/halhosan/envs/pepgate

module load anaconda3/2024.10-1
cd "$PROJECT"

export HF_HOME=/ocean/projects/cis260266p/halhosan/.cache/huggingface
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

"$ENV/bin/python" -u src/models/embed.py --encoder peptideclm1
"$ENV/bin/python" -u src/models/embed.py --encoder helmbert

echo "Finished: $(date)"
