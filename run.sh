#!/bin/bash
#SBATCH --job-name=summar-eval-test
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=linli.zhang@aalto.fi
#SBATCH --error=summar-eval-test_%j.err
#SBATCH --output=summar-eval-test_%j.out
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short

module load mamba
module load model-huggingface

# module load scicomp-llm-env

# Force transformer to load model(s) from local hub instead of download and load model(s) from remote hub.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source activate ./genaid_env

python -u biomistral_test.py

