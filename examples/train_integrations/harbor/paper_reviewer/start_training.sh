#!/bin/bash
# Start paper reviewer training.
#
# Prerequisites:
#   1. Install SkyRL with FSDP backend (see docs.skyrl.ai):
#        uv venv ~/venvs/skyrl --python 3.12
#        source ~/venvs/skyrl/bin/activate
#        cd SkyRL && uv sync --active --extra fsdp --extra harbor
#        uv pip install openai   # for LLM judge
#
#   2. Set env vars:
#        export E2B_API_KEY=your_key
#        export GEMINI_API_KEY=your_gemini_key       # for reward computation (Gemini 3 Flash)
#        export TAVILY_API_KEY=your_key              # optional, for web search in sandbox
#        export PROXY_PUBLIC_URL=http://<public_ip>:<proxy_external_port>
#        export SEARCH_PUBLIC_URL=http://<public_ip>:<search_external_port>
#
#   3. Activate the venv and run:
#        source ~/venvs/skyrl/bin/activate
#        bash start_training.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# ---- Config ----
MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="${MODEL_SHORT:-Qwen3-32B}"
NUM_GPUS="${NUM_GPUS:-4}"
REWARD_TYPE="llm_judge"
E2B_API_KEY="${E2B_API_KEY:?Set E2B_API_KEY}"
DATA_DIR="$REPO_ROOT/data/harbor/PaperReviews"
VLLM_PORT=8000

# ---- Cleanup old processes ----
echo "=== Cleaning up ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "main_paper_reviewer|stream_proxy|search_api" 2>/dev/null || true
sleep 2

# ---- Networking config ----
# Vast.ai exposes these internal ports externally (TCP).
# Set PROXY_PUBLIC_URL and SEARCH_PUBLIC_URL before running with the external addresses.
# e.g.: export PROXY_PUBLIC_URL=http://<public_ip>:<proxy_external_port>
#        export SEARCH_PUBLIC_URL=http://<public_ip>:<search_external_port>
PROXY_PORT="${PROXY_PORT:-8088}"
SEARCH_API_PORT="${SEARCH_API_PORT:-8086}"

# ---- Anthropic→OpenAI proxy ----
echo "=== Starting Anthropic→OpenAI proxy on port $PROXY_PORT ==="
nohup python3 "$SCRIPT_DIR/stream_proxy.py" "http://localhost:$VLLM_PORT" $PROXY_PORT > /tmp/stream_proxy.log 2>&1 &
sleep 2
echo "Proxy OK on port $PROXY_PORT (→ vLLM on port $VLLM_PORT)"

# ---- Determine public URLs ----
if [ -z "${PROXY_PUBLIC_URL:-}" ]; then
    PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "localhost")
    PROXY_PUBLIC_URL="http://${PUBLIC_IP}:${PROXY_PORT}"
    SEARCH_PUBLIC_URL="${SEARCH_PUBLIC_URL:-http://${PUBLIC_IP}:${SEARCH_API_PORT}}"
fi
echo "Proxy URL: $PROXY_PUBLIC_URL"
echo "Search API URL: $SEARCH_PUBLIC_URL"

# ---- Export env vars BEFORE Ray so workers inherit them ----
export ANTHROPIC_BASE_URL="$PROXY_PUBLIC_URL"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_AUTH_TOKEN="dummy"
export E2B_API_KEY
export TAVILY_API_KEY="${TAVILY_API_KEY:-}"
export SEARCH_API_URL="$SEARCH_PUBLIC_URL"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gemini-3-flash-preview}"
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-$GEMINI_API_KEY}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
export PYTHONPATH="$REPO_ROOT"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# ---- Ray (started AFTER env vars so workers inherit them) ----
echo "=== Starting Ray ==="
ray start --head --num-gpus=$NUM_GPUS

# ---- Training ----
echo "=== Starting training ==="
exec python -m examples.train_integrations.harbor.paper_reviewer.entrypoints.main_paper_reviewer \
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
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=$((NUM_GPUS / 2)) \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
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
  trainer.algorithm.max_seq_len=32768 \
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