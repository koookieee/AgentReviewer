# Paper Reviewer Training — Project Instructions

## File Layout

```
examples/train_integrations/harbor/
  harbor_generator.py                         # shared base class (don't touch)
  
  paper_reviewer/
    entrypoints/main_paper_reviewer.py        # training entrypoint
    paper_reviewer_generator.py               # generator + LLM judge reward
    start_training.sh                         # launch script
    stream_proxy.py                           # Anthropic→OpenAI proxy for Claude Code→vLLM
    search_api.py                             # arxiv-search-kit HTTP API
    
    paper_reviewer_instruction_template.md    # review task prompt (uploaded to each sandbox)
    llm_judge_instruction.md                  # LLM judge prompt for reward scoring
    search/SKILL.md                           # search tools docs (uploaded to each sandbox)
    
    harbor_trial_config/paper_reviewer.yaml   # Harbor agent/sandbox config
    prepare_paper_reviewer_dataset.py         # dataset builder
```

## Dataset Structure

Each task directory MUST have these files (Harbor requires `task.toml` and `environment/Dockerfile`):

```
data/harbor/PaperReviews/
  <arxiv_id>/
    instruction.md              # generated from template
    task.toml                   # Harbor task config (required!)
    environment/Dockerfile      # E2B sandbox base image (required!)
    task_metadata.json          # {"title", "abstract", "human_reviews": [...]}
    latex/template.tex          # paper source
    human_reviews/              # optional: review_1.md, review_2.md
```

Without human reviews, the LLM judge scores on 3 criteria (comprehension, substance, insight). With human reviews, it scores on all 5 (adds issue_overlap, calibration).

## How to Create the Dataset

**Option A — Dummy dataset (1 paper duplicated, no human reviews):**

```bash
# Already have one paper in data/harbor/PaperReviews/2603.10165v1/
cd data/harbor/PaperReviews
for i in $(seq -w 001 049); do cp -r 2603.10165v1 "2603.10165v1_dup_${i}"; done
```

**Option B — Real dataset from arXiv:**

```bash
python examples/train_integrations/harbor/paper_reviewer/prepare_paper_reviewer_dataset.py \
    --categories cs.CL,cs.CV,cs.LG --num-papers 300 --year-range 2023-2025
```

This downloads LaTeX source from arXiv and generates `instruction.md`, `task.toml`, and `environment/Dockerfile` for each paper.

**Option C — Full dataset with human reviews:**
Not implemented yet. Would need `prepare_paper_reviewer_dataset.py` updated to query OpenReview API for papers with reviews.

## What Prompts/Instructions Reach Claude Code

Claude Code receives these files in the E2B sandbox (uploaded by `_setup_environment` in `paper_reviewer_generator.py`):

- `instruction.md` — the review task prompt (6-phase review procedure)
- `search-papers-skill.md` — search tools documentation (Paper Search API + Tavily)
- `search_api_url.txt` — the search API URL so Claude can curl it
- `latex/template.tex` — the actual paper to review
- Tavily CLI — installed and authenticated automatically in the sandbox

Flow: Claude Code starts → reads instruction.md → reads search-papers-skill.md → reads the paper → runs searches → writes the review.

## Infrastructure Setup

### 1. Search API Machine (GPU with ~8GB VRAM, e.g. RTX 4060)

Hosts arxiv-search-kit (928K CS papers, SPECTER2 embeddings on GPU).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install 'arxiv-search-kit[gpu]' aiohttp
export S2_API_KEY=your_semantic_scholar_key  # optional, for enrichment
# Copy search_api.py to the machine, then:
python search_api.py --port 8086
```

The API auto-downloads a ~4GB index from HuggingFace on first use.

### 2. Training Machine (2-4x A100/B200 GPUs)

**Install SkyRL (follow docs.skyrl.ai):**

```bash
# System deps
sudo apt update && sudo apt-get install build-essential libnuma-dev

# Clone and install
git clone https://github.com/novasky-ai/SkyRL.git
cd SkyRL
uv venv ~/venvs/skyrl --python 3.12
source ~/venvs/skyrl/bin/activate
uv sync --active --extra fsdp --extra harbor
uv pip install openai  # for LLM judge

# Set Ray uv hook
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```

**Copy project files into SkyRL:**

```bash
# Copy examples/train_integrations/harbor/ and data/harbor/ into the SkyRL repo
# The paper_reviewer/ directory and dataset must be inside SkyRL/
```

**Pre-download the model:**

```bash
source ~/venvs/skyrl/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B')"
```

### 3. Start Training

```bash
source ~/venvs/skyrl/bin/activate
cd SkyRL

# Set env vars
export E2B_API_KEY=...
export GEMINI_API_KEY=...                        # for reward scoring (Gemini 3 Flash)
export TAVILY_API_KEY=...                        # optional, web search in sandbox
export PROXY_PUBLIC_URL=http://<ip>:<ext_port_8088>
export SEARCH_PUBLIC_URL=http://<ip>:<ext_port_8086>
export PYTHONPATH=$(pwd)
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Run
bash examples/train_integrations/harbor/paper_reviewer/start_training.sh
```

Override model/GPU count: `MODEL=Qwen/Qwen3-14B MODEL_SHORT=Qwen3-14B NUM_GPUS=2 bash start_training.sh`

### Port Mapping (Vast.ai)

The E2B sandboxes need to reach the stream proxy from the internet:
- **Proxy** (8088/tcp): Anthropic→OpenAI translation, so Claude Code in E2B can talk to vLLM
- **Search API** (8086/tcp): arxiv-search-kit, can run on a separate machine

Set `PROXY_PUBLIC_URL` and `SEARCH_PUBLIC_URL` to the externally reachable addresses.

## LLM Judge

Uses Gemini 3 Flash Preview (`gemini-3-flash-preview`) via OpenAI-compatible API.
Override with: `LLM_JUDGE_MODEL`, `LLM_JUDGE_API_KEY`, `LLM_JUDGE_BASE_URL`.

## SkyRL Patch (Required for Tool Calling)

SkyRL has a bug where `OpenAIServingRender` doesn't receive `enable_auto_tools` and `tool_parser`, so vLLM rejects `tool_choice: "auto"` requests. Apply this patch after cloning SkyRL:

```bash
cd SkyRL
# In skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py
# Find the OpenAIServingRender() constructor (~line 401) and add these two lines:
#   enable_auto_tools=openai_kwargs.get("enable_auto_tools", False),
#   tool_parser=openai_kwargs.get("tool_parser", None),
# Or apply the patch file:
git apply /path/to/patches/skyrl_tool_calling.patch
```

The patch file is at `patches/skyrl_tool_calling.patch`. The full patched file is at `patches/vllm_engine.py`.

## Key Gotchas

1. **task.toml is required** — Harbor crashes without it. `prepare_paper_reviewer_dataset.py` generates it automatically. For dummy datasets, ensure the source dir has one.
2. **environment/Dockerfile is required** — Harbor needs it for E2B sandbox creation.
3. **SkyRL tool calling patch is required** — Without it, vLLM rejects all tool_choice="auto" requests with a 400 error. See patch section above.
4. **Don't use `uv run --isolated`** — It creates temp envs that can't share caches. Use `source ~/venvs/skyrl/bin/activate && python -m ...` instead.
5. **Pre-download the model** — vLLM engine init can time out if it has to download during Ray actor creation.
6. **Clean /tmp/ray regularly** — Old Ray sessions accumulate and fill disk. Run `ray stop --force` and `rm -rf /tmp/ray/session_*` before restarting.
7. **tensor_parallel_size * num_engines must equal NUM_GPUS** — e.g., 2 GPUs = 2 engines x TP=1, or 4 GPUs = 2 engines x TP=2.
8. **32K context is tight** — The model reads a full LaTeX paper + tool results, often exceeding 32K. Consider `max_model_len=65536` if the model supports it.