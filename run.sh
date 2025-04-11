#!/bin/bash
#SBATCH --job-name=poro-test
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=linli.zhang@aalto.fi
#SBATCH --error=biomistral-test_%j.err
#SBATCH --output=biomistral-test_%j.out



module load model-huggingface

# module load scicomp-llm-env

# Force transformer to load model(s) from local hub instead of download and load model(s) from remote hub.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source activate ./genaid_env

python -u summarization.py

