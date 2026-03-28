## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Paper Reviewer Training

Train an agentic paper reviewer using Claude Code + SkyRL + Harbor.

**Prerequisites:** Daytona API key, ngrok auth token, 2+ GPUs.

```bash
# On any machine with GPUs (vast.ai, cloud VM, etc.)
export DAYTONA_API_KEY=your_key
export DAYTONA_API_URL=https://app.daytona.io/api
export NGROK_AUTHTOKEN=your_token
bash examples/train_integrations/harbor/start_training.sh

# On SLURM (e.g. Purdue Anvil)
sbatch examples/train_integrations/harbor/slurm_paper_reviewer.sh
```

See [paper_review_agent.md](paper_review_agent.md) for dataset format and training instructions.

### Key Files

| File | Purpose |
|------|---------|
| `start_training.sh` | One-script training launch (non-SLURM) |
| `slurm_paper_reviewer.sh` | SLURM job script |
| `run_paper_reviewer.sh` | Training hyperparams (model, batch size, reward) |
| `harbor_trial_config/paper_reviewer.yaml` | Harbor agent/sandbox config |
| `paper_reviewer_generator.py` | Custom reward from review output |
| `paper_reviewer_instruction_template.md` | Review prompt template |
| `paper_review_agent.md` | Dataset format and training instructions |

### Architecture

```
Claude Code (in Daytona sandbox)
  → ngrok tunnel
    → LiteLLM proxy (Anthropic → OpenAI translation)
      → vLLM (serves training model)
        → SkyRL GRPO training loop
```
