#!/bin/bash

#SBATCH --job-name=coconut-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --account=uvasrg_paid
#SBATCH --constraint=a100_80gb

python3 run_noisy_experiment.py