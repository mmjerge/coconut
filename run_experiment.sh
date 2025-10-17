#!/bin/bash

#SBATCH --job-name=coconut-noise
#SBATCH --output=logs/coconut_%j.out
#SBATCH --error=logs/coconut_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --account=uvasrg_paid
#SBATCH --constraint=a100_80gb

# Create logs directory
mkdir -p logs

# Load CUDA
module load cuda/11.8.0

# Activate environment (adjust path if needed)
# Option 1: If using conda
if [ -d "$HOME/.conda/envs/coconut_env" ]; then
    source $HOME/.conda/etc/profile.d/conda.sh
    conda activate coconut_env
# Option 2: If using venv
elif [ -d "./coconut_env" ]; then
    source ./coconut_env/bin/activate
# Option 3: If using system python with user packages
else
    export PATH="$HOME/.local/bin:$PATH"
fi

# Print info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "Python: $(which python)"
echo "========================================"
echo "Available GPUs:"
nvidia-smi --list-gpus
echo "========================================"

# Verify accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "ERROR: accelerate not found. Please run setup_environment.sh first"
    exit 1
fi

# Set accelerate config
export ACCELERATE_CONFIG_FILE=./accelerate_config.yaml

# Launch with accelerate
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    --multi_gpu \
    --mixed_precision fp16 \
    accelerate_gpu.py

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"