#!/bin/bash
# Start paper reviewer training. Usage:
#   export E2B_API_KEY=your_key NGROK_AUTHTOKEN=your_token
#   bash start_training.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# ---- Config ----
MODEL="Qwen/Qwen3-4B-Thinking-2507"
MODEL_SHORT="Qwen3-4B-Thinking-2507"
NUM_GPUS=2
REWARD_TYPE="dummy"
E2B_API_KEY="${E2B_API_KEY:?Set E2B_API_KEY}"
NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN:?Set NGROK_AUTHTOKEN}"
DATA_DIR="$REPO_ROOT/data/harbor/PaperReviews"
VLLM_PORT=8000

# ---- Cleanup old processes ----
echo "=== Cleaning up ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "litellm|ngrok|main_paper_reviewer|stream_proxy" 2>/dev/null || true
sleep 2

# ---- Anthropic→OpenAI proxy (SkyRL's HTTP endpoint only exposes OpenAI format) ----
PROXY_PORT=4001
echo "=== Starting Anthropic→OpenAI proxy ==="
nohup python3 "$SCRIPT_DIR/stream_proxy.py" "http://localhost:$VLLM_PORT" $PROXY_PORT > /tmp/stream_proxy.log 2>&1 &
sleep 2
echo "Proxy OK on port $PROXY_PORT (→ vLLM on port $VLLM_PORT)"

# ---- arxiv-search-kit API ----
SEARCH_API_PORT=4002
echo "=== Starting search API ==="
nohup /usr/bin/python3 "$SCRIPT_DIR/search_api.py" --port $SEARCH_API_PORT > /tmp/search_api.log 2>&1 &
sleep 2
echo "Search API on port $SEARCH_API_PORT"

# ---- ngrok (multi-tunnel: proxy + search API) ----
echo "=== Starting ngrok ==="
command -v ngrok &>/dev/null || { curl -sO https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz && tar xzf ngrok-v3-stable-linux-amd64.tgz && mv ngrok /usr/local/bin/ngrok; }
ngrok config add-authtoken "$NGROK_AUTHTOKEN" 2>/dev/null

# Write ngrok config for two tunnels
cat > /tmp/ngrok_tunnels.yml <<EOF
version: 3
tunnels:
  proxy:
    addr: $PROXY_PORT
    proto: http
  search:
    addr: $SEARCH_API_PORT
    proto: http
EOF
nohup ngrok start --all --config "$HOME/.config/ngrok/ngrok.yml" --config /tmp/ngrok_tunnels.yml --log=stdout --log-level=info > /tmp/ngrok.log 2>&1 &
sleep 5

# Extract tunnel URLs
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if str(t['config']['addr']).endswith('$PROXY_PORT'):
        print(t['public_url'])
        break
")
SEARCH_API_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if str(t['config']['addr']).endswith('$SEARCH_API_PORT'):
        print(t['public_url'])
        break
")
echo "ngrok proxy URL: $NGROK_URL (→ proxy:$PROXY_PORT → vLLM:$VLLM_PORT)"
echo "ngrok search URL: $SEARCH_API_URL (→ search_api:$SEARCH_API_PORT)"

# ---- Export env vars BEFORE Ray so workers inherit them ----
export ANTHROPIC_BASE_URL="$NGROK_URL"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_AUTH_TOKEN="dummy"
export E2B_API_KEY
export NGROK_VLLM_URL="$NGROK_URL"
export SEARCH_API_URL="$SEARCH_API_URL"
export PYTHONPATH="$REPO_ROOT"

# ---- Ray (started AFTER env vars so workers inherit them) ----
echo "=== Starting Ray ==="
ray start --head --num-gpus=$NUM_GPUS

# ---- Training ----
echo "=== Starting training ==="
exec python -m examples.train_integrations.harbor.entrypoints.main_paper_reviewer \
  "data.train_data=[\"$DATA_DIR\"]" \
  trainer.policy.model.path=$MODEL \
  generator.inference_engine.served_model_name=$MODEL_SHORT \
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
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
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
  generator.inference_engine.http_endpoint_port=$VLLM_PORT \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=2 \
  generator.rate_limit.max_concurrency=2 \
  reward_type=$REWARD_TYPE \
  "$@"
