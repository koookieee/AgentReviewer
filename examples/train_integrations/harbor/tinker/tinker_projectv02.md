# Claude Code RL with Tinker + DeepInfra Sampler

RL training using **Claude Code** as the agent harness, **DeepInfra** for cheap sampling with logprobs, and **Tinker API** for GPU training compute.

## Architecture

```
Claude Code CLI (inside E2B sandbox, reads paper + instruction)
    | POST /v1/messages (Anthropic Messages API)
    v
anthropic_proxy.py (runs on orchestration machine)
    |-- Translates Anthropic -> OpenAI format (LiteLLM AnthropicAdapter)
    |-- Dispatches to sampler backend:
    |     "deepinfra": calls DeepInfra API (Qwen3.5-35B-A3B) with native tool calling + logprobs
    |     "tinker":    calls Tinker SamplingClient via tinker-cookbook renderer (original path)
    |-- Captures per-session (prompt_token_ids, output_token_ids, logprobs)
    |-- Returns Anthropic-format response with tool_use blocks
    v
train.py (orchestration loop)
    |-- Harbor PaperReviewerTrial.run() -> E2B sandbox + paper upload + search tools
    |-- LLM judge reward (Gemini) -> composite score [0, 1]
    |-- GRPO advantages (group-relative policy optimization)
    |-- Proxy-captured sessions -> Tinker Datum objects (token_ids, logprobs, advantages)
    |-- Tinker forward_backward(datums, loss_fn="importance_sampling") -> GPU training
    |-- Tinker optim_step(Adam) -> weight update
    |-- Weight sync -> next rollout
```

## Sampler Backends

### DeepInfra (default for cost savings)

- Uses DeepInfra's OpenAI-compatible API: `https://api.deepinfra.com/v1/openai/chat/completions`
- Model: `Qwen/Qwen3.5-35B-A3B` (same model as Tinker trainer)
- Returns logprobs on both text AND tool call responses (verified working)
- Native tool calling via `tools` parameter (no renderer hack needed)
- Pay-per-token pricing instead of GPU hourly cost

### Tinker (original, on-policy)

- Uses Tinker SamplingClient on remote GPU
- Weights updated after each training step (on-policy)
- More expensive but no off-policy gap

### Off-Policy Considerations (DeepInfra)

DeepInfra serves frozen base weights. After Tinker updates the LoRA weights, the sampling
policy (DeepInfra) and training policy (Tinker) diverge. The importance sampling loss
corrects for this: `loss = -(exp(log_p_new - log_p_old) * advantage)`. This works well
for early training when LoRA changes are small. For longer runs, consider periodically
switching to Tinker sampling for on-policy rollouts.

## Hyperparameters

### Model

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Training model (Tinker) | `Qwen/Qwen3.5-35B-A3B` | `train.py --model_name` or `.env MODEL=` |
| Sampling model (DeepInfra) | `Qwen/Qwen3.5-35B-A3B` | `train.py --deepinfra_model` or `.env DEEPINFRA_MODEL=` |
| Sampler backend | `deepinfra` | `train.py --sampler_backend` or `.env SAMPLER_BACKEND=` |

### Context Window & Token Limits

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Model context window | 262,144 tokens | `train.py --deepinfra_max_ctx` (also in proxy `__init__`) |
| Max input tokens (prompt) | 253,952 (= 262144 - 8192) | `train.py --max_input_tokens` |
| Max output tokens (per turn) | 8,192 | `train.py --max_tokens` |
| DeepInfra API output cap | 16,384 (hard limit) | Cannot change (DeepInfra enforced). Our 8192 is within it. |

### Sampling Parameters

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Temperature | 0.7 | `train.py --temperature` |
| top_p | 1.0 (disabled) | `train.py` Config `llm_kwargs.top_p` |
| top_k | -1 (disabled) | `train.py` Config `llm_kwargs.top_k` |
| min_p | 0.0 (disabled) | `train.py` Config `llm_kwargs.min_p` |
| logprobs | true | Hardcoded in `anthropic_proxy.py` DeepInfra payload |
| top_logprobs | 1 | Hardcoded in `anthropic_proxy.py` DeepInfra payload |

### Training (RL)

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| LoRA rank | 32 | `train.py --lora_rank` or `.env LORA_RANK=` |
| Learning rate | 1e-6 | `train.py --learning_rate` or `.env LEARNING_RATE=` |
| Optimizer | Adam (beta1=0.9, beta2=0.95, eps=1e-8) | `train.py` `run()` function |
| Loss function | `importance_sampling` | `train.py --loss_fn` |
| Max training steps | 200 | `train.py --max_steps` or `.env MAX_STEPS=` |
| Checkpoint every N steps | 5 | `train.py --save_every` |

### GRPO (Group Relative Policy Optimization)

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Group size | 2 (trajectories per paper) | `train.py --group_size` or `.env GROUP_SIZE=` |
| Groups per batch | 4 (papers per step) | `train.py --groups_per_batch` or `.env GROUPS_PER_BATCH=` |
| Total trajectories per step | 8 (= group_size x groups_per_batch) | Derived |
| Normalize advantages | true | `train.py` Config |

### Agent / Sandbox

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Agent | claude-code | `train.py --agent_name` |
| Max turns per review | 64 | `train.py --max_turns` |
| Agent timeout | 2400s (40 min) | `train.py --agent_timeout` |
| Environment type | e2b | `train.py --environment_type` |
| Max concurrent trials | 8 | `train.py --max_concurrent_trials` |
| Max retries per trial | 2 | `train.py --max_retries` |

### LLM Judge (Reward)

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Judge model | gemini-3-flash-preview | `.env LLM_JUDGE_MODEL=` |
| Judge API | Google AI Studio | `.env LLM_JUDGE_BASE_URL=` |
| Judge API key | Gemini API key | `.env GEMINI_API_KEY=` |
| Scoring criteria (with human reviews) | comprehension(0.20) + substance(0.25) + insight(0.25) + issue_overlap(0.20) + calibration(0.10) | `paper_reviewer_utils.py` |
| Scoring criteria (no human reviews) | comprehension(0.30) + substance(0.35) + insight(0.35) | `paper_reviewer_utils.py` |

### Networking / Port Forwarding

| Parameter | Value | Where to change |
|-----------|-------|-----------------|
| Proxy listen port (internal) | 8082 | `train.py --proxy_port` or `.env PROXY_PORT=` |
| Proxy public URL | auto-detected, or set manually | `.env PROXY_PUBLIC_URL=` |

On vast.ai, the internal port must map to an externally reachable port. Example:
- Internal: `0.0.0.0:8082`
- External: `http://<public-ip>:17986` (vast.ai maps `17986 -> 8082/tcp`)
- Set: `export PROXY_PUBLIC_URL=http://<public-ip>:17986`

E2B sandboxes are remote containers — they reach the proxy via this public URL.

## Files

| File | Purpose |
|------|---------|
| `anthropic_proxy.py` | Anthropic Messages API proxy. Dispatches to DeepInfra or Tinker for sampling. Captures per-session (token_ids, logprobs) for training. |
| `train.py` | Main training loop: Harbor Trial -> proxy-captured data -> GRPO -> Tinker training. |
| `paper_reviewer_utils.py` | PaperReviewerTrial (uploads paper to E2B), LLM judge reward, chat history extraction. |
| `start_tinker_training.sh` | Launch script with env var defaults and validation. |
| `.env` | API keys and config overrides. **Never commit with real keys.** |

## Quick Start

```bash
# 1. Install dependencies on orchestration machine
pip install tinker tinker-cookbook harbor litellm openai \
            numpy torch fastapi uvicorn httpx

# 2. Fill in API keys
cd examples/train_integrations/harbor/tinker
cp .env .env.local   # make a local copy
# Edit .env.local with your keys:
#   TINKER_API_KEY, E2B_API_KEY, GEMINI_API_KEY, DEEPINFRA_API_KEY

# 3. Set the proxy public URL for your machine
# On vast.ai, check your port mappings (e.g., 17986 -> 8082/tcp)
export PROXY_PUBLIC_URL=http://<your-public-ip>:<external-port>

# 4. Run training
source .env.local
bash start_tinker_training.sh

# Or run train.py directly with overrides:
python train.py \
    --sampler_backend deepinfra \
    --data_dir /path/to/paper/tasks \
    --max_steps 100 \
    --group_size 2 \
    --groups_per_batch 4
```

## Re-running Training (Checklist)

1. **API keys**: Fill `.env` with fresh keys (Tinker, E2B, Gemini, DeepInfra)
2. **Port forwarding**: Set `PROXY_PUBLIC_URL` to your machine's externally reachable address + mapped port for 8082
3. **Data directory**: Point `--data_dir` to your paper review tasks (each task needs `instruction.md`, `task.toml`, `latex/template.tex`)
4. **Search API**: Set `SEARCH_API_URL` if you have the arxiv search service running
5. **Adjust hyperparameters**: Edit `.env` or pass CLI flags to `train.py`
6. **WandB (optional)**: Pass `--use_wandb` and set `--wandb_project` / `--wandb_name`

## Verified Behavior (e2e test run)

Tested on vast.ai with DeepInfra sampler (2026-04-13):
- Model: `Qwen/Qwen3.5-35B-A3B` (both sampler and trainer)
- Agent completed 16-turn paper review in ~30s via DeepInfra
- Proxy captured 2 interactions with token_ids + logprobs
- 2 training datums constructed
- Tinker `forward_backward` completed in 6.8s
- Weight sync in 1.3s
- Total step time: 45.5s

```
[Step 0] 1 ok, 0 masked, reward: mean=0.000 max=0.000
[Step 0] datums=2 train=6.8s sync=1.3s total=45.5s
```

(Reward was 0.0 due to expired Gemini API key — the pipeline itself is working end-to-end.)

## Critical Design Decisions

1. **DeepInfra returns logprobs on tool calls** — Unlike OpenRouter/Alibaba which drops logprobs on native tool call responses, DeepInfra returns them. No need to inject tools into system prompt.
2. **Session routing via agent env dict** — Harbor's claude_code agent passes env vars from the config's `agent.env` dict (which overrides `os.environ`). The session_id must be set in `config["agent"]["env"]["ANTHROPIC_API_KEY"]`, not just `os.environ`.
3. **Tokenizer alignment** — DeepInfra and Tinker use the same Qwen tokenizer, but minor differences are handled by character-level logprob alignment (`_align_logprobs`).
4. **DeepInfra max output cap is 16,384 tokens** — Our `max_tokens=8192` is safely within this limit.
5. **Renderer still used for prompt tokenization** — Even with DeepInfra sampling, the tinker-cookbook renderer tokenizes the prompt to produce `prompt_token_ids` for training datum construction.

## References

- **AReaL** — Proxy architecture, per-session capture, "individual" export style
- **SkyRL** — Harbor integration, datum construction, GRPO
- **tinker-cookbook** — Renderer pipeline, tool schema injection
- **DeepInfra docs** — [Function calling](https://deepinfra.com/docs/advanced/function_calling), [Log probs](https://deepinfra.com/docs/advanced/log_probs), [Max tokens](https://deepinfra.com/docs/advanced/max_tokens_limit)
