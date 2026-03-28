# Paper Review Agent Training

RL training of an agentic paper reviewer using Claude Code + SkyRL + Harbor.

## Prerequisites

- 2+ GPUs (H100/A100/RTX 6000)
- E2B API key ([e2b.dev](https://e2b.dev))
- ngrok auth token ([ngrok.com](https://ngrok.com))

## Dataset

Each task = a subdirectory with `instruction.md` + `task.toml` + paper source.

```
data/harbor/PaperReviews/
  <paper_id>/
    instruction.md          # Review prompt
    task.toml               # Harbor config
    environment/Dockerfile  # Sandbox image
    latex/template.tex      # Paper LaTeX
```

**task.toml:**
```toml
version = "1.0"
[metadata]
arxiv_id = "2603.10165"
task_type = "paper_review"
[agent]
timeout_sec = 1200.0
[verifier]
timeout_sec = 60.0
[environment]
docker_image = "kureiu/claudecode-sandbox:latest"
cpus = 1
memory_mb = 4096
allow_internet = true
```

**Quick test data** (1 paper x 200):
```bash
cd data/harbor/PaperReviews
for i in $(seq -w 001 199); do cp -r 2603.10165v1 "2603.10165v1_dup_${i}"; done
for d in */; do
  cp 2603.10165v1/task.toml "$d/task.toml"
  mkdir -p "$d/environment"
  echo 'FROM kureiu/claudecode-sandbox:latest' > "$d/environment/Dockerfile"
done
```

**Real data** from arXiv:
```bash
python examples/train_integrations/harbor/prepare_paper_reviewer_dataset.py \
    --categories cs.CL,cs.CV,cs.LG --num-papers 300 --year-range 2023-2025
```

## Start Training

**On any GPU machine:**
```bash
export E2B_API_KEY=your_key
export NGROK_AUTHTOKEN=your_token
bash examples/train_integrations/harbor/start_training.sh
```

**On SLURM:**
```bash
sbatch examples/train_integrations/harbor/slurm_paper_reviewer.sh
```

**Remote (e.g. Vast.ai):**
```bash
# 1. SCP files to remote
scp -P <port> \
  examples/train_integrations/harbor/start_training.sh \
  examples/train_integrations/harbor/stream_proxy.py \
  examples/train_integrations/harbor/harbor_generator.py \
  examples/train_integrations/harbor/paper_reviewer_generator.py \
  root@<host>:/root/AgentReviewer/examples/train_integrations/harbor/

scp -P <port> \
  examples/train_integrations/harbor/harbor_trial_config/paper_reviewer.yaml \
  root@<host>:/root/AgentReviewer/examples/train_integrations/harbor/harbor_trial_config/

# 2. Start training
ssh -p <port> root@<host> 'killall -9 python python3 ngrok 2>/dev/null; sleep 2; \
  cd /root/AgentReviewer && \
  export E2B_API_KEY=your_key && \
  export NGROK_AUTHTOKEN=your_token && \
  nohup bash examples/train_integrations/harbor/start_training.sh > /root/training.log 2>&1 &'

# 3. Monitor
ssh -p <port> root@<host> 'tail -30 /root/training.log'
ssh -p <port> root@<host> 'tail -20 /tmp/stream_proxy.log'
```

## Config

Edit `start_training.sh` for: model, batch size, reward type, GPU count.
Edit `harbor_trial_config/paper_reviewer.yaml` for: agent timeout, sandbox memory, max turns.

## Architecture

```
Claude Code (E2B sandbox) --> ngrok --> Anthropic-to-OpenAI proxy (stream_proxy.py) --> vLLM --> SkyRL GRPO
```

The proxy (`stream_proxy.py`) translates between Anthropic Messages API (what Claude Code speaks) and OpenAI Chat Completions API (what vLLM speaks). It also caps `max_tokens` to 16384 to prevent context length overflow.

---

## Debugging Notes: Qwen3 Thinking Models + vLLM Tool Calling

These notes document issues encountered and fixed when training with `Qwen3-4B-Thinking-2507` (and apply to all Qwen3 thinking models like Qwen3-32B, Qwen3-8B, etc.).

### Issue 1: Wrong vLLM tool_call_parser

**Symptom:** Model never generates tool calls. Claude Code sees plain text instead of structured tool_use blocks.

**Root cause:** We were using `tool_call_parser=qwen3_xml`. That parser is for **Qwen3-Coder** models only (e.g. `Qwen3-Coder-30B-A3B-Instruct`). Regular Qwen3 / Qwen3-Thinking models use Hermes-style `<tool_call>` XML format in their built-in chat template.

**Fix:** Use `tool_call_parser=hermes` with `enable_auto_tool_choice=true`.

```yaml
# In start_training.sh engine_init_kwargs:
generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true
generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes
```

**Which parser for which Qwen model:**
| Model | Parser |
|-------|--------|
| Qwen3-4B-Thinking, Qwen3-8B, Qwen3-32B (regular/thinking) | `hermes` |
| Qwen3-Coder-30B-A3B-Instruct, Qwen3-Coder-480B | `qwen3_xml` |
| Qwen2.5-\*, QwQ-32B | `hermes` |

Reference: [vLLM Tool Calling docs](https://docs.vllm.ai/en/stable/features/tool_calling.html)

### Issue 2: Custom chat template breaks tool calling

**Symptom:** Same as above -- no tool calls generated.

**Root cause:** A custom `qwen3_thinking_chat_template.jinja` was passed via `engine_init_kwargs.chat_template=...`. This custom template only handled basic `<|im_start|>/<|im_end|>` message wrapping with **zero tool support** -- no `<tools>` definitions, no `<tool_call>` parsing.

**Fix:** Do NOT pass a custom chat template. Remove the `chat_template` kwarg entirely. The model's **built-in tokenizer chat template** (in `tokenizer_config.json`) already handles everything:
- Tool definitions via `<tools>...</tools>` XML in the system prompt
- Tool calls via `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
- Tool results via `<tool_response>...</tool_response>`
- Thinking blocks via `<think>...</think>`

### Issue 3: `enable_reasoning` is not a valid AsyncEngineArgs parameter

**Symptom:** `TypeError: AsyncEngineArgs.__init__() got an unexpected keyword argument 'enable_reasoning'`

**Root cause:** SkyRL passes `engine_init_kwargs` to vLLM's `AsyncEngineArgs`. SkyRL's `pop_openai_kwargs()` (in `skyrl/backends/.../vllm/utils.py`) pops known serve-level params (`enable_auto_tool_choice`, `tool_call_parser`, `reasoning_parser`, `chat_template`) before passing the rest to `AsyncEngineArgs`. But `enable_reasoning` is NOT handled -- it's a CLI flag for `vllm serve`, not an engine arg, and SkyRL doesn't pop it.

**Fix:** Don't pass `enable_reasoning`. Just omit it. The model's built-in chat template handles `<think>` tags natively without needing an explicit reasoning parser.

Valid params that SkyRL pops and handles:
- `enable_auto_tool_choice` -> popped, passed to OpenAI serving layer
- `tool_call_parser` -> popped, passed to OpenAI serving layer
- `reasoning_parser` -> popped, passed to OpenAI serving layer (also a valid `AsyncEngineArgs` param in vLLM 0.18+)
- `chat_template` -> popped, loaded from file, passed to OpenAI serving layer

NOT valid (will crash if passed):
- `enable_reasoning` -- not popped by SkyRL, not a valid `AsyncEngineArgs` param

### Issue 4: Tokenization assertion error on assistant messages without `<think>` tags

**Symptom:**
```
AssertionError: Assistant message tokens should start with generation prompt.
Expected [151644, 77091, 198, 151667, 198], got [151644, 77091, 198, 151667, 271]
```
(Token 198 = `\n`, token 271 = `\n\n`)

**Root cause:** The Qwen3-Thinking chat template always wraps assistant messages with `<think>...</think>`, even in history. The generation prompt is `<|im_start|>assistant\n<think>\n` (ends with single `\n`, token 198).

When assistant messages from the Claude Code session JSONL don't contain `<think>` tags (because the proxy strips them / vLLM returns them as part of regular content), the chat template injects an **empty** thinking block: `<think>\n\n</think>`. The `\n\n` after `<think>` tokenizes as token 271 (double newline), not 198 (single newline), causing the assertion to fail.

**Fix:** In `paper_reviewer_generator.py`'s `_extract_chat_history_from_trial()`, wrap assistant messages that lack `<think>` tags with a non-empty placeholder:
```python
if "<think>" not in combined:
    combined = f"<think>\n...\n</think>\n\n{combined}"
```

The key is that the content between `<think>` and `</think>` must be **non-empty** (even just `...`). An empty `<think>\n\n</think>` will NOT work because `\n\n` tokenizes as a single token (271) instead of two `\n` tokens (198, 198).

**This applies to ALL Qwen3-Thinking models** (4B, 8B, 32B, etc.) since they all use the same chat template structure.

### Issue 5: Context length overflow (400 error from vLLM)

**Symptom:**
```
400: You passed 33537 input tokens and requested 32000 output tokens.
However, the model's context length is only 65536 tokens
```

**Root cause:** Claude Code requests `max_tokens=32000` by default. With 22 tools + a paper's worth of context, input tokens can reach 30K+, and 30K + 32K > 65536.

**Fix:** Cap `max_tokens` in `stream_proxy.py`:
```python
"max_tokens": min(body_json.get("max_tokens", 4096), 16384),
```

### Training Output / Trace Analysis

A successful trajectory looks like this (from the Claude Code session JSONL):

```
[0] queue-operation: enqueue (review task prompt)
[1] queue-operation: dequeue
[2] USER: # Paper Review Task ... (the review prompt)
[3] ASSISTANT: <thinking about what to do> (3-5K chars)
[4] ASSISTANT: TOOL_USE(Read, file_path="latex/template.tex")
[5] USER/TOOL_RESULT: <paper LaTeX content>
[6] PROGRESS
[7] ASSISTANT: <full review output> (9-11K chars)
[8] LAST-PROMPT
```

The model reads the paper via the Read tool, then generates the review. With the 4B model, it typically does 1-2 tool calls per trajectory. Larger models (32B) should follow the multi-phase review procedure more faithfully with 10+ search tool calls.

Proxy logs show multi-turn tool calling working:
```
[proxy] model=Qwen3-4B-Thinking-2507 tools=22 msgs=1 stream=True   # initial
[proxy] model=Qwen3-4B-Thinking-2507 tools=22 msgs=3 stream=True   # after tool result
[proxy] model=Qwen3-4B-Thinking-2507 tools=22 msgs=5 stream=True   # more turns
[proxy] model=Qwen3-4B-Thinking-2507 tools=22 msgs=7 stream=True   # continuing
```