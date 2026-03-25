#!/bin/bash
#SBATCH --job-name=paper_reviewer
#SBATCH --account=nairr250100-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/anvil/scratch/x-spei/logs/paper_reviewer_%j.out
#SBATCH --error=/anvil/scratch/x-spei/logs/paper_reviewer_%j.err

set -ex

mkdir -p /anvil/scratch/x-spei/logs

PROJECT_DIR=/anvil/projects/x-nairr250100/AgentReviewer
SIF=/anvil/projects/x-nairr250100/skyrl.sif

apptainer exec --nv \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --bind /anvil/scratch/x-spei:/anvil/scratch/x-spei \
    --env PATH="${PROJECT_DIR}/.venv/bin:/home/ray/anaconda3/bin:$PATH" \
    --env HF_HOME=/anvil/scratch/x-spei/hf_cache \
    --env PYTHONPATH=${PROJECT_DIR} \
    --pwd ${PROJECT_DIR} \
    ${SIF} \
    bash -c "source .venv/bin/activate && bash examples/train_integrations/harbor/run_paper_reviewer.sh"
