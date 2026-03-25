set -ex

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Search-papers skill API keys (required for literature search).
# export S2_API_KEY=YOUR_KEY_HERE
# export OPENALEX_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

#-----------------------
# Dataset setup
#-----------------------
# Prepare dataset first (downloads from arXiv and creates task directories):
# uv run examples/train_integrations/harbor/prepare_paper_reviewer_dataset.py \
#     --categories cs.CL,cs.CV,cs.LG --num-papers 300 --year-range 2023-2025
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/harbor"
TRAIN_DATA="['$DATA_DIR/PaperReviews']"
EVAL_DATA=""

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="paper_reviewer"
TRIALS_DIR="$HOME/$RUN_NAME/trials_run"
CKPTS_DIR="$HOME/$RUN_NAME/ckpts"
EXPORTS_DIR="$HOME/$RUN_NAME/exports"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

#-----------------------
# Training setup
#-----------------------
MINI_BATCH_SIZE=2
MAX_MODEL_LEN=32768
APPLY_OVERLONG_FILTERING=true

# Dr. GRPO parameters
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

# Reward type: "dummy" for pipeline testing, "format" for structure-based rewards
REWARD_TYPE="dummy"


#----------------
# Infrastructure setup
# 1 node x 2 H100 GPUs
#----------------
NUM_GPUS_PER_NODE=2
NUM_NODES=1
TENSOR_PARALLEL_SIZE=1
NUM_ENGINES=2  # NUM_GPUS_PER_NODE * NUM_NODES / TENSOR_PARALLEL_SIZE

ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=2
MAX_CONCURRENCY=32

# Run SkyRL command
uv run --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_paper_reviewer \
  data.train_data=$TRAIN_DATA \
  trainer.policy.model.path=Qwen/Qwen3-4B-Thinking-2507 \
  generator.inference_engine.served_model_name=Qwen3-4B-Thinking-2507 \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=1 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=2 \
  generator.eval_n_samples_per_prompt=1 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  trainer.logger=console \
  trainer.project_name=paper_reviewer \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  reward_type=$REWARD_TYPE \
  "$@"
