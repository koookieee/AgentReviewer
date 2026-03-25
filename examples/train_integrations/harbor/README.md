## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Structure

```
examples/train_integrations/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  dataset.py                       # HarborTaskDataset: loads task directory paths
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```

### Paper Reviewer Training (SLURM)

Train an agentic paper reviewer with Claude Code + SkyRL + Harbor on a SLURM cluster (e.g. Purdue Anvil).

```bash
# 1. Prepare dataset (duplicate a single paper for testing)
# Task dirs go in data/harbor/PaperReviews/ — each needs instruction.md, task.toml, environment/Dockerfile, latex/

# 2. Submit SLURM job (starts ngrok tunnel + LiteLLM proxy + SkyRL training)
sbatch examples/train_integrations/harbor/slurm_paper_reviewer.sh
```

Key files:
- `run_paper_reviewer.sh` — training config (model, batch size, reward type)
- `slurm_paper_reviewer.sh` — SLURM config, ngrok tunnel, Daytona credentials
- `harbor_trial_config/paper_reviewer.yaml` — Harbor agent/sandbox config
- `paper_reviewer_generator.py` — custom reward from review output (no verifier)
- `paper_reviewer_instruction_template.md` — review prompt template
