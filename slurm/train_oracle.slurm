#!/bin/bash
#SBATCH --job-name=pepgate-clm2
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/peptideclm2_%j.out
#SBATCH --error=slurm/logs/peptideclm2_%j.err

set -euo pipefail

# Project and environment paths
PROJECT=/ocean/projects/cis260266p/halhosan/PepGate
ENV=/ocean/projects/cis260266p/halhosan/envs/pepgate

# Initialize Bridges-2 environment
module load anaconda3/2024.10-1

cd "$PROJECT"

# Environment configuration
export HF_HOME=/ocean/projects/cis260266p/halhosan/.cache/huggingface
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" results/peptideclm2_v1

echo "===================================="
echo "PepGate - PeptideCLM-2 PAMPA Regression"
echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Project: $(pwd)"
echo

# Display allocated GPU
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

# Verify PyTorch, Transformers and CUDA
"$ENV/bin/python" -u - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA GPU is unavailable"

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("CUDA matrix multiplication: PASSED")
PY

echo
echo "Starting PeptideCLM-2 fine-tuning..."
echo

# Fine-tune PeptideCLM-2 Hybrid-small for PAMPA regression
"$ENV/bin/python" -u src/models/train_oracle.py --model peptideclm1 \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-5

echo
echo "Training finished: $(date)"

