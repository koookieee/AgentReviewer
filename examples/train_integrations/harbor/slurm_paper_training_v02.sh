#!/bin/bash
#SBATCH --job-name=paper_reviewer_v02
#SBATCH --account=nairr250100-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/anvil/scratch/x-spei/logs/paper_reviewer_v02_%j.out
#SBATCH --error=/anvil/scratch/x-spei/logs/paper_reviewer_v02_%j.err

set -ex

mkdir -p /anvil/scratch/x-spei/logs

PROJECT_DIR=/anvil/projects/x-nairr250100/AgentReviewer
SIF=/anvil/projects/x-nairr250100/skyrl.sif
NGROK=${PROJECT_DIR}/ngrok
VLLM_PORT=8000
PROXY_PORT=4001
SEARCH_API_PORT=4002

# ---- Start Anthropic→OpenAI stream proxy ----
apptainer exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --env PATH="${PROJECT_DIR}/.venv/bin:$PATH" \
    --env PYTHONPATH=${PROJECT_DIR} \
    --pwd ${PROJECT_DIR} \
    ${SIF} \
    bash -c "source .venv/bin/activate && python3 examples/train_integrations/harbor/stream_proxy.py http://localhost:${VLLM_PORT} ${PROXY_PORT}" \
    > /anvil/scratch/x-spei/logs/stream_proxy_${SLURM_JOB_ID}.log 2>&1 &
PROXY_PID=$!
sleep 3
echo "Stream proxy started on port ${PROXY_PORT} (PID: ${PROXY_PID})"

# ---- Start arxiv-search-kit API ----
apptainer exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    --env PATH="${PROJECT_DIR}/.venv/bin:$PATH" \
    --env PYTHONPATH=${PROJECT_DIR} \
    --env HF_HOME=/anvil/scratch/x-spei/hf_cache \
    --pwd ${PROJECT_DIR} \
    ${SIF} \
    bash -c "source .venv/bin/activate && python3 examples/train_integrations/harbor/search_api.py --port ${SEARCH_API_PORT}" \
    > /anvil/scratch/x-spei/logs/search_api_${SLURM_JOB_ID}.log 2>&1 &
SEARCH_PID=$!
sleep 3
echo "Search API started on port ${SEARCH_API_PORT} (PID: ${SEARCH_PID})"

# ---- Start ngrok (multi-tunnel: proxy + search API) ----
cat > /tmp/ngrok_tunnels_${SLURM_JOB_ID}.yml <<EOF
version: 3
tunnels:
  proxy:
    addr: ${PROXY_PORT}
    proto: http
  search:
    addr: ${SEARCH_API_PORT}
    proto: http
EOF

${NGROK} start --all \
    --config /home/x-spei/.config/ngrok/ngrok.yml \
    --config /tmp/ngrok_tunnels_${SLURM_JOB_ID}.yml \
    --log=stdout --log-level=info \
    > /anvil/scratch/x-spei/logs/ngrok_v02_${SLURM_JOB_ID}.log 2>&1 &
NGROK_PID=$!
sleep 5

# Extract tunnel URLs
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if str(t['config']['addr']).endswith('${PROXY_PORT}'):
        print(t['public_url'])
        break
")
SEARCH_API_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if str(t['config']['addr']).endswith('${SEARCH_API_PORT}'):
        print(t['public_url'])
        break
")
echo "ngrok proxy URL: ${NGROK_URL} (→ proxy:${PROXY_PORT} → vLLM:${VLLM_PORT})"
echo "ngrok search URL: ${SEARCH_API_URL} (→ search_api:${SEARCH_API_PORT})"

# ---- Config ----
MODEL="Qwen/Qwen3-4B-Thinking-2507"
MODEL_SHORT="Qwen3-4B-Thinking-2507"
NUM_GPUS=2
REWARD_TYPE="llm_judge"
DATA_DIR="${PROJECT_DIR}/data/harbor/PaperReviews"

# ---- Run training inside container ----
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
    --env ANTHROPIC_AUTH_TOKEN=dummy \
    --env CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
    --env E2B_API_KEY=${E2B_API_KEY} \
    --env NGROK_AUTHTOKEN=${NGROK_AUTHTOKEN} \
    --env GEMINI_API_KEY=${GEMINI_API_KEY} \
    --env SEARCH_API_URL=${SEARCH_API_URL} \
    --env DAYTONA_API_KEY=dtn_97d3466f70f1629c2eeffd3fb3964a2d754edf53a6f95686ba00c669273c0c33 \
    --env DAYTONA_API_URL=https://app.daytona.io/api \
    --pwd ${PROJECT_DIR} \
    ${SIF} \
    bash -c 'source .venv/bin/activate && \
    ray start --head --num-gpus='${NUM_GPUS}' && \
    python -m examples.train_integrations.harbor.entrypoints.main_paper_reviewer \
      "data.train_data=[\"'${DATA_DIR}'\"]" \
      trainer.policy.model.path='${MODEL}' \
      generator.inference_engine.served_model_name='${MODEL_SHORT}' \
      harbor_trial_config.trials_dir=$HOME/paper_reviewer/trials_run \
      trainer.export_path=$HOME/paper_reviewer/exports \
      trainer.ckpt_path=$HOME/paper_reviewer/ckpts \
      trainer.log_path=$HOME/paper_reviewer/logs \
      trainer.algorithm.advantage_estimator=grpo \
      trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
      trainer.algorithm.grpo_norm_by_std=false \
      trainer.algorithm.use_kl_loss=false \
      trainer.placement.colocate_all=true \
      trainer.strategy=fsdp2 \
      trainer.placement.policy_num_nodes=1 \
      trainer.placement.policy_num_gpus_per_node='${NUM_GPUS}' \
      generator.inference_engine.num_engines='${NUM_GPUS}' \
      generator.inference_engine.tensor_parallel_size=1 \
      generator.inference_engine.engine_init_kwargs.max_model_len=98304 \
      generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
      generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
      generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
      trainer.epochs=1 \
      trainer.eval_batch_size=2 \
      trainer.eval_before_train=false \
      trainer.eval_interval=10 \
      trainer.update_epochs_per_batch=1 \
      trainer.train_batch_size=2 \
      trainer.policy_mini_batch_size=2 \
      trainer.micro_forward_batch_size_per_gpu=1 \
      trainer.micro_train_batch_size_per_gpu=1 \
      trainer.ckpt_interval=5 \
      trainer.hf_save_interval=5 \
      trainer.algorithm.max_seq_len=98304 \
      trainer.policy.optimizer_config.lr=1.0e-6 \
      generator.n_samples_per_prompt=2 \
      generator.eval_n_samples_per_prompt=1 \
      generator.apply_overlong_filtering=true \
      generator.inference_engine.gpu_memory_utilization=0.85 \
      trainer.logger=console \
      trainer.project_name=paper_reviewer \
      trainer.run_name=paper_reviewer \
      trainer.resume_mode=latest \
      generator.inference_engine.backend=vllm \
      generator.inference_engine.run_engines_locally=true \
      generator.inference_engine.weight_sync_backend=nccl \
      generator.inference_engine.async_engine=true \
      generator.batched=false \
      generator.inference_engine.enforce_eager=false \
      generator.inference_engine.enable_http_endpoint=true \
      generator.inference_engine.http_endpoint_host=127.0.0.1 \
      generator.inference_engine.http_endpoint_port='${VLLM_PORT}' \
      generator.rate_limit.enabled=true \
      generator.rate_limit.trajectories_per_second=2 \
      generator.rate_limit.max_concurrency=2 \
      reward_type='${REWARD_TYPE}''

# ---- Cleanup ----
kill ${NGROK_PID} 2>/dev/null
kill ${PROXY_PID} 2>/dev/null
kill ${SEARCH_PID} 2>/dev/null
