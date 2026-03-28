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
NGROK=${PROJECT_DIR}/ngrok
LITELLM_PORT=4000

# Start ngrok tunnel for LiteLLM proxy
${NGROK} http ${LITELLM_PORT} --log=stdout --log-level=info > /anvil/scratch/x-spei/logs/ngrok_${SLURM_JOB_ID}.log 2>&1 &
NGROK_PID=$!
sleep 5
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")
echo "ngrok URL: ${NGROK_URL}"

apptainer exec --nv \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --bind /anvil/scratch/x-spei:/anvil/scratch/x-spei \
    --bind /home/x-spei/.config/ngrok:/home/x-spei/.config/ngrok \
    --env PATH="${PROJECT_DIR}/.venv/bin:/home/ray/anaconda3/bin:$PATH" \
    --env HF_HOME=/anvil/scratch/x-spei/hf_cache \
    --env PYTHONPATH=${PROJECT_DIR} \
    --env NGROK_VLLM_URL=${NGROK_URL} \
    --env ANTHROPIC_BASE_URL=${NGROK_URL} \
    --env ANTHROPIC_API_KEY=dummy \
    --env CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
    --env DAYTONA_API_KEY=dtn_97d3466f70f1629c2eeffd3fb3964a2d754edf53a6f95686ba00c669273c0c33 \
    --env DAYTONA_API_URL=https://app.daytona.io/api \
    --pwd ${PROJECT_DIR} \
    ${SIF} \
    bash -c "source .venv/bin/activate && bash examples/train_integrations/harbor/run_paper_reviewer.sh"

kill ${NGROK_PID} 2>/dev/null
