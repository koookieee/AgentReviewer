#!/bin/bash
# Start paper reviewer RL training with Tinker API.
#
# This runs on the orchestration machine (e.g., vast.ai). Tinker handles
# all GPU compute remotely via the API. The local machine only runs:
#   - The Anthropic proxy (translates Claude Code → Tinker sampling)
#   - Harbor trial orchestration (E2B sandboxes)
#   - LLM judge reward computation (Gemini API)
#   - GRPO advantage calculation + Tinker training API calls
#
# Prerequisites:
#   1. Install dependencies:
#        pip install tinker tinker-cookbook litellm harbor openai \
#                    numpy torch fastapi uvicorn httpx aiohttp
#
#   2. Set environment variables (see below)
#
#   3. Run:
#        bash start_tinker_training.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Paper Reviewer RL Training with Tinker API ==="
echo "Repo root: $REPO_ROOT"
echo "Script dir: $SCRIPT_DIR"

# ── Required env vars ──
: "${TINKER_API_KEY:?Set TINKER_API_KEY (e.g., tml-...)}"
: "${E2B_API_KEY:?Set E2B_API_KEY}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (for LLM judge reward)}"

# ── Config ──
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144}"
LORA_RANK="${LORA_RANK:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_STEPS="${MAX_STEPS:-200}"
GROUP_SIZE="${GROUP_SIZE:-2}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-4}"
PROXY_PORT="${PROXY_PORT:-8082}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/harbor/PaperReviews}"
LOG_DIR="${LOG_DIR:-/tmp/paper-reviewer-tinker-logs}"

# ── Networking ──
# E2B sandboxes need an externally reachable URL for the proxy.
# Set PROXY_PUBLIC_URL if you know the external address, otherwise
# the script auto-detects via ifconfig.me.
if [ -z "${PROXY_PUBLIC_URL:-}" ]; then
    PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "localhost")
    PROXY_PUBLIC_URL="http://${PUBLIC_IP}:${PROXY_PORT}"
fi
echo "Proxy public URL: $PROXY_PUBLIC_URL"

# Search API URL (paper search service running externally)
SEARCH_API_URL="${SEARCH_API_URL:-}"
if [ -n "$SEARCH_API_URL" ]; then
    echo "Search API URL: $SEARCH_API_URL"
fi

# ── Export env vars ──
export TINKER_API_KEY
export E2B_API_KEY
export GEMINI_API_KEY
export SEARCH_API_URL
export PROXY_PUBLIC_URL

# LLM judge config
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gemini-3-flash-preview}"
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-$GEMINI_API_KEY}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"

export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR:$PYTHONPATH"

# ── Verify data directory ──
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Run prepare_paper_reviewer_dataset.py first to create training tasks."
    exit 1
fi
TASK_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Found $TASK_COUNT paper review tasks in $DATA_DIR"

# ── Run training ──
echo ""
echo "=== Starting training ==="
echo "  Model: $MODEL"
echo "  LoRA rank: $LORA_RANK"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max steps: $MAX_STEPS"
echo "  Group size: $GROUP_SIZE"
echo "  Groups/batch: $GROUPS_PER_BATCH"
echo "  Proxy port: $PROXY_PORT"
echo "  Log dir: $LOG_DIR"
echo ""

exec python "$SCRIPT_DIR/train.py" \
    --model_name "$MODEL" \
    --lora_rank "$LORA_RANK" \
    --learning_rate "$LEARNING_RATE" \
    --max_steps "$MAX_STEPS" \
    --group_size "$GROUP_SIZE" \
    --groups_per_batch "$GROUPS_PER_BATCH" \
    --proxy_port "$PROXY_PORT" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    "$@"
