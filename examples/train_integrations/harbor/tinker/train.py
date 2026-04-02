"""
RL training for Paper Reviewer using Harbor + Tinker API.

Architecture
============

    Claude Code CLI (inside E2B sandbox, reads paper + instruction)
        │ Anthropic Messages API
        ▼
    AnthropicTinkerProxy (anthropic_proxy.py)
        │ Captures tokens + logprobs per session at generation time
        ▼
    Tinker SamplingClient → remote GPU (Qwen3-32B etc.)

    Harbor PaperReviewerTrial.run() orchestrates:
        E2B sandbox + paper upload + search tools + tavily
    Proxy session data → per-turn (token_ids, logprobs)
    LLM judge (Gemini) → reward
    GRPO advantages → tinker.Datum → Tinker forward_backward + optim_step

Key Differences from agentsmd-rl reference
===========================================

1. **PaperReviewerTrial** instead of base Trial — uploads paper content,
   search skill docs, search API URL, and tavily CLI to E2B sandbox.

2. **LLM judge reward** instead of test verifier — calls Gemini to evaluate
   review quality on 5 criteria (comprehension, substance, insight,
   issue_overlap, calibration). Composite reward in [0, 1].

3. **E2B sandbox** instead of Docker — paper tasks use E2B for isolation.

4. **Task loading** from data/harbor/PaperReviews/ — each task dir has
   instruction.md, task.toml, latex/template.tex, task_metadata.json.

Prerequisites
=============
    export TINKER_API_KEY=tml-...
    export E2B_API_KEY=...
    export GEMINI_API_KEY=...         # for LLM judge reward
    export TAVILY_API_KEY=...         # optional, for web search in sandbox
    export SEARCH_API_URL=http://...  # paper search API endpoint

Usage
=====
    python train.py
    python train.py --model_name Qwen/Qwen3-32B --max_steps 100
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import tinker
import torch
from tinker.types.tensor_data import TensorData

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

# Local imports — these modules sit next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anthropic_proxy import AnthropicTinkerProxy, SessionData
from paper_reviewer_utils import (
    PaperReviewerTrial,
    compute_llm_judge_reward,
    extract_chat_history_from_trial,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Default data directory — try repo layout first, fall back to /root/data for remote machines
_repo_data = Path(__file__).resolve().parents[4] / "data" / "harbor" / "PaperReviews" if len(Path(__file__).resolve().parents) > 4 else None
DEFAULT_DATA_DIR = _repo_data if (_repo_data and _repo_data.is_dir()) else Path("/root/data/harbor/PaperReviews")


# ── Config ───────────────────────────────────────────────────────────────


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-32B"
    lora_rank: int = 32
    learning_rate: float = 1e-6
    max_steps: int = 200
    loss_fn: Literal["importance_sampling"] = "importance_sampling"

    # GRPO
    group_size: int = 2
    groups_per_batch: int = 4
    normalize_advantages: bool = True

    # Agent / sandbox
    agent_name: str = "claude-code"
    max_turns: int = 64
    max_tokens: int = 8192
    max_input_tokens: int = 32768
    temperature: float = 0.7
    agent_timeout: int = 2400  # 40 min for paper review
    environment_type: str = "e2b"

    # Proxy
    proxy_port: int = 8321

    # Data
    data_dir: str = str(DEFAULT_DATA_DIR)

    # Logging
    save_every: int = 5
    use_wandb: bool = False
    wandb_project: str = "paper_reviewer"
    wandb_name: str | None = None
    log_dir: str = "/tmp/paper-reviewer-tinker-logs"

    max_retries: int = 2
    max_concurrent_trials: int = 8


# ── Harbor Trial Config for Paper Reviewer ──────────────────────────────


def make_harbor_config(cfg: Config, proxy_base_url: str) -> dict[str, Any]:
    """Build Harbor TrialConfig template for paper reviewer with Claude Code.

    The proxy speaks Anthropic Messages API at proxy_base_url.
    Claude Code CLI is told to use it via ANTHROPIC_BASE_URL env var.
    """
    return {
        "trials_dir": os.path.join(cfg.log_dir, "trials"),
        "timeout_multiplier": 1.5,
        "agent": {
            "name": cfg.agent_name,
            "override_timeout_sec": cfg.agent_timeout,
            "model_name": cfg.model_name.split("/")[-1],
            "kwargs": {
                "max_turns": cfg.max_turns,
                "suppress_max_turns_warning": True,
                "enable_summarize": True,
                "store_all_messages": True,
                "temperature": cfg.temperature,
                "model_info": {
                    "max_input_tokens": cfg.max_input_tokens,
                    "max_output_tokens": cfg.max_tokens,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_kwargs": {
                    "timeout": 1200,
                    "max_retries": 0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                },
            },
            "env": {
                "ANTHROPIC_BASE_URL": proxy_base_url,
                "ANTHROPIC_API_KEY": "dummy-for-proxy",  # overridden per-trial with session_id
                "ANTHROPIC_AUTH_TOKEN": "dummy",
            },
        },
        "environment": {
            "type": cfg.environment_type,
            "force_build": True,
            "override_cpus": 1,
            "override_memory_mb": 2048,
            "override_storage_mb": 2048,
            "suppress_override_warnings": True,
            "kwargs": {
                "auto_stop_interval_mins": 45,
            },
        },
        "verifier": {"disable": True},  # reward from LLM judge, not test verifier
    }


# ── Task loading ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperTask:
    name: str
    path: Path


def load_tasks(tasks_dir: Path) -> list[PaperTask]:
    """Load paper review tasks from data directory.

    Each task dir should have at minimum: instruction.md, task.toml, latex/template.tex
    """
    tasks = []
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        logger.error(f"Tasks directory not found: {tasks_dir}")
        return tasks

    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        # Require at least instruction.md and the paper
        has_instruction = (d / "instruction.md").exists()
        has_task_toml = (d / "task.toml").exists()
        has_paper = (d / "latex" / "template.tex").exists()

        if has_instruction and has_task_toml and has_paper:
            tasks.append(PaperTask(name=d.name, path=d))
        else:
            missing = []
            if not has_instruction: missing.append("instruction.md")
            if not has_task_toml: missing.append("task.toml")
            if not has_paper: missing.append("latex/template.tex")
            logger.debug(f"Skipping {d.name}: missing {', '.join(missing)}")

    return tasks


# ── Single trial ─────────────────────────────────────────────────────────


@dataclass
class TrajectoryResult:
    reward: float = 0.0
    session_id: str = ""
    stop_reason: str = "error"
    success: bool = False
    chat_history: list[dict] | None = None


async def run_single_trial(
    task: PaperTask,
    config_template: dict[str, Any],
    proxy: AnthropicTinkerProxy,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> TrajectoryResult:
    """Run one Harbor trial with Claude Code agent reviewing a paper."""
    for attempt in range(max_retries):
        session_id = uuid4().hex
        proxy.create_session(session_id)

        try:
            config = deepcopy(config_template)
            config["task"] = {"path": str(task.path)}

            # Set ANTHROPIC_API_KEY to session_id so proxy can route
            config["agent"]["env"]["ANTHROPIC_API_KEY"] = session_id
            config["agent"]["kwargs"]["session_id"] = session_id

            trial_config = TrialConfig.model_validate(config)
            trial = PaperReviewerTrial(trial_config)

            async with semaphore:
                results = await trial.run()

            exc_type = results.exception_info.exception_type if results.exception_info else None

            if exc_type == "AgentTimeoutError":
                logger.debug(f"Trial {task.name} timed out")
                return TrajectoryResult(stop_reason="agent_timeout", session_id=session_id)

            if exc_type == "ContextLengthExceededError":
                session = proxy.get_session(session_id)
                has_data = session is not None and len(session.interactions) > 0
                return TrajectoryResult(
                    reward=0.0, session_id=session_id,
                    stop_reason="context_length", success=has_data,
                )

            # Extract chat history for LLM judge reward
            chat_history = None
            if results.agent_result and results.agent_result.metadata:
                chat_history = results.agent_result.metadata.get("all_messages")
            if not chat_history:
                chat_history, _, _ = extract_chat_history_from_trial(trial)

            # Compute reward via LLM judge
            if chat_history and len(chat_history) > 1:
                reward = await compute_llm_judge_reward(chat_history, task.path)
            else:
                logger.warning(f"Trial {task.name} attempt {attempt+1}: no valid chat history")
                proxy.pop_session(session_id)
                continue

            session = proxy.get_session(session_id)
            has_data = session is not None and len(session.interactions) > 0

            if has_data:
                return TrajectoryResult(
                    reward=reward, session_id=session_id,
                    stop_reason="complete", success=True,
                    chat_history=chat_history,
                )
            else:
                logger.warning(f"Trial {task.name}: no proxy interactions captured")
                proxy.pop_session(session_id)
                continue

        except Exception as e:
            logger.warning(f"Trial {task.name} attempt {attempt+1}: {e}")
            proxy.pop_session(session_id)
            continue

    return TrajectoryResult(stop_reason="error")


# ── GRPO ─────────────────────────────────────────────────────────────────


def compute_grpo_advantages(rewards: list[float], group_size: int) -> list[float]:
    """Compute GRPO advantages: normalize rewards within each group."""
    rewards_np = np.array(rewards)
    n_groups = len(rewards_np) // group_size
    advantages = []
    for i in range(n_groups):
        group = rewards_np[i * group_size:(i + 1) * group_size]
        group_std = group.std()
        if group_std < 1e-8:
            advantages.extend([0.0] * group_size)
        else:
            advantages.extend(((group - group.mean()) / (group_std + 1e-8)).tolist())
    return advantages


# ── Build Datum from proxy-captured session ──────────────────────────────


def session_to_datums(
    session: SessionData,
    advantage: float,
    tokenizer,
) -> list[tinker.types.Datum]:
    """Convert proxy-captured session data into Tinker Datums for training.

    Uses AReaL's "individual" style: each interaction (LLM call) becomes
    its own Datum, preserving per-turn logprobs from sampling time.

    For each interaction the proxy captured:
    - prompt_token_ids: full conversation so far (chat-templated)
    - output_token_ids: tokens the model generated this turn
    - output_logprobs: per-token logprobs from sampling time

    Each Datum has:
    - model_input: full_seq[:-1] (shifted for next-token prediction)
    - target_tokens: full_seq[1:]
    - logprobs: 0 for prompt, real for output (shifted)
    - advantages: applied only on output tokens
    """
    if not session.interactions:
        return []

    datums = []
    for interaction in session.interactions:
        prompt_ids = interaction.prompt_token_ids
        output_ids = interaction.output_token_ids
        output_lps = interaction.output_logprobs

        if not output_ids:
            continue

        full_seq = prompt_ids + output_ids
        prompt_len = len(prompt_ids)

        # Shift for next-token prediction (Tinker convention)
        target_tokens = full_seq[1:]
        logprobs = ([0.0] * prompt_len + list(output_lps))[1:]

        # Advantages: uniform on output tokens, 0 on prompt
        adv_tensor = torch.zeros(len(full_seq))
        for j in range(prompt_len, len(full_seq)):
            adv_tensor[j] = advantage
        adv_tensor = adv_tensor[1:]

        datums.append(tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=full_seq[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
                "advantages": TensorData.from_torch(adv_tensor),
            },
        ))

    return datums


# ── Training loop ────────────────────────────────────────────────────────


async def run(cfg: Config) -> None:
    tasks = load_tasks(Path(cfg.data_dir))
    logger.info(f"Loaded {len(tasks)} paper review tasks from {cfg.data_dir}")
    if not tasks:
        logger.error("No valid tasks found")
        sys.exit(1)

    os.makedirs(cfg.log_dir, exist_ok=True)

    # ── Tinker service ──
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank,
    )
    adam_params = tinker.types.AdamParams(
        learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # Initial sampling client + tokenizer
    sc = tc.save_weights_and_get_sampling_client()
    tokenizer = sc.get_tokenizer()

    # ── Start Anthropic proxy ──
    proxy = AnthropicTinkerProxy(
        sc,
        model_name=cfg.model_name.split("/")[-1],
        base_model=cfg.model_name,
    )
    proxy.start(host="0.0.0.0", port=cfg.proxy_port)

    # Determine public URL for proxy — E2B sandboxes need an externally reachable URL
    proxy_public_url = os.environ.get("PROXY_PUBLIC_URL")
    if not proxy_public_url:
        # Try to detect public IP
        import subprocess as _sp
        try:
            public_ip = _sp.check_output(
                ["curl", "-s", "--max-time", "5", "ifconfig.me"],
                text=True, timeout=10,
            ).strip()
        except Exception:
            public_ip = "127.0.0.1"
        proxy_public_url = f"http://{public_ip}:{cfg.proxy_port}"
    logger.info(f"Proxy URL for E2B sandboxes: {proxy_public_url}")

    harbor_template = make_harbor_config(cfg, proxy_public_url)
    semaphore = asyncio.Semaphore(cfg.max_concurrent_trials)

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name or f"paper-reviewer-{datetime.now():%m%d-%H%M}",
        )

    # ── Training loop ──
    task_idx = 0
    for step in range(cfg.max_steps):
        t0 = time.time()
        metrics: dict[str, Any] = {"step": step}

        # Select batch of tasks
        batch_tasks: list[PaperTask] = []
        for _ in range(cfg.groups_per_batch):
            batch_tasks.append(tasks[task_idx % len(tasks)])
            task_idx += 1

        # Run trials (group_size trajectories per task)
        total_trials = len(batch_tasks) * cfg.group_size
        logger.info(f"[Step {step}] {len(batch_tasks)} tasks × {cfg.group_size} = {total_trials} trajectories")

        trial_coros = []
        for task in batch_tasks:
            for _ in range(cfg.group_size):
                trial_coros.append(run_single_trial(
                    task, harbor_template, proxy, semaphore, max_retries=cfg.max_retries,
                ))
        trial_results: list[TrajectoryResult] = await asyncio.gather(*trial_coros)
        metrics["time/rollout"] = time.time() - t0

        # Instance-level masking: if any trajectory in a group fails, mask the whole group
        masked_groups: set[int] = set()
        for i, r in enumerate(trial_results):
            group_idx = i // cfg.group_size
            if r.stop_reason in ("error", "agent_timeout"):
                masked_groups.add(group_idx)

        # Collect successful sessions + rewards
        session_data_list: list[SessionData | None] = []
        all_rewards: list[float] = []
        n_success = 0
        n_masked = 0

        for i, r in enumerate(trial_results):
            group_idx = i // cfg.group_size
            if group_idx in masked_groups or not r.success:
                session_data_list.append(None)
                all_rewards.append(0.0)
                n_masked += 1
                if r.session_id:
                    proxy.pop_session(r.session_id)
                continue

            session = proxy.get_session(r.session_id)
            if session and session.interactions:
                session.reward = r.reward
                session_data_list.append(session)
                all_rewards.append(r.reward)
                n_success += 1
            else:
                session_data_list.append(None)
                all_rewards.append(0.0)
                n_masked += 1

        metrics["n_success"] = n_success
        metrics["n_masked"] = n_masked
        metrics["reward/mean"] = float(np.mean(all_rewards)) if all_rewards else 0.0
        metrics["reward/max"] = float(np.max(all_rewards)) if all_rewards else 0.0
        metrics["reward/min"] = float(np.min(all_rewards)) if all_rewards else 0.0
        logger.info(
            f"[Step {step}] {n_success} ok, {n_masked} masked, "
            f"reward: mean={metrics['reward/mean']:.3f} max={metrics['reward/max']:.3f}"
        )

        if n_success == 0:
            logger.warning(f"[Step {step}] All trials failed, skipping training step")
            continue

        # GRPO advantages
        advantages = compute_grpo_advantages(all_rewards, group_size=cfg.group_size)

        # Build datums from proxy-captured sessions
        training_datums = []
        for idx, (session, adv) in enumerate(zip(session_data_list, advantages)):
            if session is None:
                continue
            datums = session_to_datums(session, adv, tokenizer)
            training_datums.extend(datums)
            proxy.pop_session(session.session_id)

        if not training_datums:
            logger.warning(f"[Step {step}] No training datums, skipping")
            continue

        metrics["n_datums"] = len(training_datums)

        # ── Training step ──
        t_train = time.time()
        fwd_bwd_future = tc.forward_backward(training_datums, loss_fn=cfg.loss_fn)
        optim_future = tc.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        metrics["time/train"] = time.time() - t_train

        # ── Weight sync ──
        t_sync = time.time()
        sc = tc.save_weights_and_get_sampling_client()
        proxy.update_client(sc)
        metrics["time/weight_sync"] = time.time() - t_sync
        metrics["time/total"] = time.time() - t0

        logger.info(
            f"[Step {step}] datums={len(training_datums)} "
            f"train={metrics['time/train']:.1f}s sync={metrics['time/weight_sync']:.1f}s "
            f"total={metrics['time/total']:.1f}s"
        )

        # Save checkpoint
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_name = f"step_{step:04d}"
            tc.save_state(ckpt_name).result()
            logger.info(f"[Step {step}] Saved checkpoint: {ckpt_name}")

        if cfg.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    # Save final checkpoint
    tc.save_state("final").result()
    logger.info("Saved final checkpoint")

    proxy.stop()
    if cfg.use_wandb:
        import wandb
        wandb.finish()
    logger.info("Training complete")


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser(description="Paper Reviewer RL Training with Tinker")
    cfg = Config()
    for name, default in vars(cfg).items():
        if isinstance(default, bool):
            parser.add_argument(f"--{name}", action="store_true", default=default)
        elif isinstance(default, (int, float, str)):
            parser.add_argument(f"--{name}", type=type(default), default=default)
    args = parser.parse_args()
    return Config(**{k: v for k, v in vars(args).items()})


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(run(cfg))
