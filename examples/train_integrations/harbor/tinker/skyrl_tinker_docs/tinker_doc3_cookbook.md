Tinker API
Cookbook Scripts
The tinker-cookbook is a library provided by Thinking Machines with ready-to-run training recipes. This page describes how to run a few example recipes on SkyRL, and provides example curves from our experiments.

Setup
Follow the Quickstart to install SkyRL, then clone the cookbook:


git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
Start the Server
Before launching a training script, start the Tinker server from the SkyRL/ directory:


uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
    --base-model "Qwen/Qwen3-0.6B" --backend fsdp
The same server command can be reused across all recipes below. For more detail on configuration options for the training backend, see Configuration.

Recipes
All of the cookbook recipes default to LoRA training (e.g., with lora_rank=32), but full-parameter fine-tuning (FFT) is supported on SkyRL by setting lora_rank=0. Per Thinking Machines' learning rate guide, use a ~10x lower learning rate when switching to FFT.

Supervised Learning Loop (sl_loop)
Fine-tunes a model on the No Robots dataset using cross-entropy loss with a linear learning rate decay.


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets \
    python -m tinker_cookbook.recipes.sl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    train_on_what=LAST_ASSISTANT_MESSAGE
SL NLL over steps

For full-parameter fine-tuning (no LoRA), set lora_rank=0 and lower the learning rate:


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets \
    python -m tinker_cookbook.recipes.sl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    train_on_what=LAST_ASSISTANT_MESSAGE \
    lora_rank=0 learning_rate=1e-5
RL Training Loop (rl_loop)
Trains a model on GSM8K math problems using GRPO-style reward centering with importance sampling.


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B"
RL reward over steps

For full-parameter fine-tuning (no LoRA), set lora_rank=0 and lower the learning rate:


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    lora_rank=0 learning_rate=4e-6
Note: rl_loop uses ephemeral weight sync by default, syncing weights to the inference engine without writing to disk. See Weight Sync for details on ephemeral vs persistent modes.

Math RL (math_rl)
RL training specifically for mathematical reasoning.


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.math_rl.train \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B"
Math RL correct over steps

Code RL (code_rl)
RL training for code generation tasks. Uses the same importance_sampling loss with code execution-based rewards.


TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.code_rl.train \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    lora_rank=0 learning_rate=1e-6