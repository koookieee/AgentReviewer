"""
Anthropic Messages API proxy backed by Tinker SamplingClient.

Claude Code CLI always speaks the Anthropic protocol (POST /v1/messages).
This proxy translates requests → tinker-cookbook's renderer pipeline
(tool injection, tokenization, sampling, tool call parsing) and captures
per-session (messages, token_ids, logprobs) for RL training.

Architecture
============

    Claude Code CLI (inside Docker/Harbor sandbox)
        │ POST /v1/messages (Anthropic Messages API)
        ▼
    AnthropicTinkerProxy (this module)
        ├── Translates Anthropic → OpenAI format (LiteLLM AnthropicAdapter)
        ├── Routes through tinker-cookbook renderer pipeline:
        │     renderer.create_conversation_prefix_with_tools() → tool schema injection
        │     renderer.build_generation_prompt() → tokenization
        │     SamplingClient.sample_async() → token generation + logprobs
        │     renderer.parse_response() → tool call extraction
        ├── Captures per-session (prompt_ids, output_ids, logprobs) for RL
        └── Returns Anthropic-format response with tool_use blocks

Critical Design Decisions & Bugs Fixed
=======================================

1. **Renderer pipeline, not raw apply_chat_template** (CRITICAL)
   Raw ``tokenizer.apply_chat_template(messages, tools=tools)`` crashes on
   multi-turn conversations with tool_call/tool_result messages — Qwen3's
   Jinja template can't handle OpenAI-format tool messages. The renderer
   pipeline (``_sample_chat_completion``) handles this correctly by injecting
   tools into the system prompt in the model's expected format and using
   ``openai_messages_to_tinker()`` for message conversion.

2. **Tool call parsing from model output** (CRITICAL)
   Tinker's SamplingClient returns raw tokens. The model outputs tool calls
   in its native format (e.g., ``<tool_call>{"name":"bash",...}</tool_call>``
   for Qwen3). The renderer's ``parse_response()`` extracts these into
   structured ``ToolCall`` objects, which we convert to Anthropic ``tool_use``
   content blocks. Without this, Claude Code sees no tool calls and the
   conversation ends after 1 turn.

3. **Dynamic max_tokens capping** (CRITICAL)
   Claude Code requests ``max_tokens=32000+`` which overflows the model's
   context window when the prompt is large (especially in later turns).
   We estimate prompt length via the renderer, then cap max_tokens so
   ``prompt + max_tokens <= context_window``. The context window is
   auto-detected from Tinker's error message on the first overflow (no
   hardcoded limits).

4. **Session routing by API key** (CRITICAL)
   Harbor sets ``ANTHROPIC_API_KEY=session_id`` in the container env.
   Claude Code sends this as the ``x-api-key`` header. We use direct dict
   lookup (``api_key in self._sessions``) to route to the correct session.
   An earlier bug used ``if sid == api_key or api_key:`` which was always
   true for any non-empty key, causing cross-session contamination.

5. **Raw token decode for Anthropic response text** (CORRECTNESS)
   The renderer's ``parse_response()`` → ``render_message()`` is NOT a
   lossless round-trip: it strips ``<think>``/``<tool_call>`` tags and
   re-adds them with different whitespace (extra ``\\n`` before tool_call
   blocks, ``json.dumps`` normalizes JSON formatting). When Claude Code
   sends this modified text back in the next turn and the renderer
   re-tokenizes it, the tokens differ from the original output.

   Fix: we decode raw output tokens ourselves (``tokenizer.decode(tokens,
   skip_special_tokens=True)``) and strip only ``<tool_call>`` blocks
   (which become separate ``tool_use`` content blocks). Everything else —
   including ``<think>`` tags, whitespace, punctuation — is preserved
   exactly. This ensures turn N+1's prompt contains turn N's output tokens
   verbatim after re-tokenization.

   Impact: small in practice (AReaL/SkyRL don't bother, each turn's datum
   uses its own captured tokens regardless). But it's a clean ~15-line fix
   that eliminates token-level context drift between turns.

6. **Logprob divergence is expected** (NOT A BUG)
   Per-turn logprobs from sampling (captured here) differ from
   ``compute_logprobs_async`` or ``forward_backward`` by ~0.01 mean / ~0.3
   max per token. This is the well-documented "rollout-training mismatch"
   caused by different attention kernels (sampling with incremental KV cache
   vs. prefill forward pass). All production RL systems (AReaL, SkyRL, veRL)
   handle this via importance ratio clipping (TIS cap ~2.0).

   Refs:
   - AReaL: decoupled loss with version tracking
   - SkyRL: ``OffPolicyCorrectionConfig`` with token-level TIS
   - veRL: rollout correction math docs
   - Tinker: ``compare_sampling_training_logprobs.py`` test

References
==========
- AReaL/areal/experimental/openai/proxy/proxy_rollout_server.py
- tinker-cookbook/third_party/litellm/provider.py (_sample_chat_completion)
- SkyRL/examples/train_integrations/harbor/harbor_generator.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import tinker
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tinker_cookbook.renderers import Renderer, get_renderer, format_content_as_string
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker, openai_tools_to_tinker
from tinker_cookbook.third_party.litellm.provider import _prepare_messages_with_tools, _SamplingResult

logger = logging.getLogger(__name__)


# ── Per-session data capture ─────────────────────────────────────────────


@dataclass
class SessionInteraction:
    """One LLM call captured at the proxy layer."""
    interaction_id: str
    messages: list[dict]
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_logprobs: list[float]
    finish_reason: str = "stop"


@dataclass
class SessionData:
    """All interactions for a single Claude Code session."""
    session_id: str
    interactions: list[SessionInteraction] = field(default_factory=list)
    reward: float | None = None
    created_at: float = field(default_factory=time.time)


# ── DeepInfra sampler ──────────────────────────────────────────────────


DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"


async def _deepinfra_sample_chat_completion(
    renderer: Renderer,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 1.0,
    max_tokens: int = 128,
    top_p: float = 1.0,
    tools: list[dict[str, Any]] | None = None,
    model_name: str = "tinker",
    deepinfra_model: str = "Qwen/Qwen3.5-35B-A3B",
    deepinfra_api_key: str = "",
) -> _SamplingResult:
    """Sample via DeepInfra API instead of Tinker SamplingClient.

    DeepInfra returns logprobs on both text AND tool call responses,
    so we can use native tool calling (pass tools param directly).

    Steps:
    1. Build tokenized prompt via renderer (for prompt_token_ids used by training)
    2. Call DeepInfra chat completions with messages + tools + logprobs
    3. Re-tokenize output text using Tinker's tokenizer for training-compatible token_ids
    4. Align DeepInfra's per-token logprobs to Tinker's tokenization
    5. Parse tool calls from the response

    Returns _SamplingResult with the same interface as _sample_chat_completion.
    """
    # Step 1: Build tokenized prompt for training datum construction.
    # The renderer tokenizes the conversation in the model's native format,
    # giving us prompt_token_ids that Tinker's forward_backward uses.
    tinker_msgs = openai_messages_to_tinker(messages)
    if tools:
        tinker_msgs = _prepare_messages_with_tools(renderer, tinker_msgs, tools)
    model_input = renderer.build_generation_prompt(tinker_msgs)
    prompt_token_ids: list[int] = model_input.to_ints()

    # Step 2: Call DeepInfra with native tool calling + logprobs.
    payload: dict[str, Any] = {
        "model": deepinfra_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": 1,
    }
    if tools:
        payload["tools"] = tools

    headers = {
        "Authorization": f"Bearer {deepinfra_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(DEEPINFRA_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        raise RuntimeError(f"DeepInfra error: {data['error']}")

    choice = data["choices"][0]
    msg = choice["message"]
    raw_text = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls", [])
    finish_reason = choice.get("finish_reason", "stop")

    # Step 3: Reconstruct the full output text the model generated.
    # DeepInfra returns tool calls as structured objects, but the model
    # actually generated them as text tokens (with logprobs). We reconstruct
    # the full text so we can re-tokenize it with Tinker's tokenizer.
    #
    # For tool calls, the model generated something like:
    #   <tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>
    # DeepInfra parsed this into the tool_calls field. We reconstruct it
    # so the token IDs match what the training model expects.
    full_output_text = raw_text
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_text = json.dumps({"name": fn.get("name", ""), "arguments": fn.get("arguments", "")})
            full_output_text += f"\n<tool_call>\n{tc_text}\n</tool_call>"

    completion_token_ids: list[int] = tokenizer.encode(
        full_output_text, add_special_tokens=False,
    ) if full_output_text else []

    # Step 4: Extract and align logprobs.
    # DeepInfra uses the same Qwen tokenizer, so logprobs should align 1:1
    # with our re-tokenized output. But we use character-level alignment
    # as a safety net in case of minor tokenization differences.
    lp_data = choice.get("logprobs", {})
    lp_tokens = lp_data.get("content", []) if lp_data else []

    if lp_tokens and completion_token_ids:
        logprobs = _align_logprobs(lp_tokens, full_output_text, completion_token_ids, tokenizer)
    elif lp_tokens:
        logprobs = [t["logprob"] for t in lp_tokens]
    else:
        logger.warning("DeepInfra returned no logprobs")
        logprobs = [0.0] * len(completion_token_ids)

    # Step 5: Build parsed_message for the Anthropic response.
    # We construct it directly from DeepInfra's structured response
    # instead of using renderer.parse_response(), since DeepInfra already
    # parsed tool calls for us.
    parsed_tool_calls = []
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            parsed_tool_calls.append(ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                function=ToolCall.FunctionBody(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
            ))

    parsed_message = {"role": "assistant", "content": raw_text}
    if parsed_tool_calls:
        parsed_message["tool_calls"] = parsed_tool_calls

    parse_success = finish_reason in ("stop", "tool_calls")

    return _SamplingResult(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        logprobs=logprobs,
        parsed_message=parsed_message,
        parse_success=parse_success,
        model_name=model_name,
    )


def _align_logprobs(
    api_tokens: list[dict],
    full_text: str,
    completion_token_ids: list[int],
    tokenizer,
) -> list[float]:
    """Align API per-token logprobs to Tinker tokenizer token IDs.

    Both tokenizers produce tokens that cover the same output text but may
    split it differently. We build a character-level logprob array from the
    API tokens, then aggregate per Tinker token.
    """
    # Build character-level logprob array from API tokens
    char_logprobs: list[float] = []
    for tok_info in api_tokens:
        tok_text = tok_info.get("token", "")
        tok_lp = tok_info.get("logprob", 0.0)
        for _ in tok_text:
            char_logprobs.append(tok_lp)

    # Map each Tinker token to its character span and sum logprobs
    aligned: list[float] = []
    char_offset = 0
    for tid in completion_token_ids:
        tok_text = tokenizer.decode([tid])
        tok_len = len(tok_text)
        if tok_len == 0:
            aligned.append(0.0)
            continue
        end = min(char_offset + tok_len, len(char_logprobs))
        if char_offset < end:
            aligned.append(sum(char_logprobs[char_offset:end]))
        else:
            aligned.append(0.0)
        char_offset = end

    return aligned


# ── Anthropic ↔ OpenAI translation ───────────────────────────────────────


def _translate_anthropic_to_openai(anthropic_request: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic Messages API request to OpenAI chat format.

    Uses LiteLLM's AnthropicAdapter for the heavy lifting.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    adapter = AnthropicAdapter()
    openai_request = adapter.translate_completion_input_params(anthropic_request.copy())
    if openai_request is None:
        raise ValueError("Failed to translate Anthropic request to OpenAI format")
    openai_request = dict(openai_request)

    # Fix content blocks: Claude Code sends [{"type":"text","text":"...","cache_control":{...}}]
    if "messages" in openai_request:
        for msg in openai_request["messages"]:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                msg["content"] = "\n".join(text_parts)

    return openai_request


def _openai_response_to_anthropic(
    openai_response: dict[str, Any],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Convert OpenAI-style response dict to Anthropic Message format."""
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content_text = message.get("content", "")
    finish_reason = choice.get("finish_reason", "stop")

    stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    content_blocks = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            input_obj = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_obj = {"raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": fn.get("name", "unknown"),
            "input": input_obj,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ── Sampler dispatch helpers ────────────────────────────────────────────


async def _sample_tinker(
    proxy,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_requested: int,
    est_prompt_len: int,
) -> _SamplingResult | JSONResponse:
    """Sample using Tinker SamplingClient (original path)."""
    from tinker_cookbook.third_party.litellm.provider import _sample_chat_completion

    try:
        return await _sample_chat_completion(
            sampling_client=proxy._client,
            renderer=proxy._renderer,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model_name=proxy._model_name,
        )
    except Exception as e:
        err_str = str(e)
        # Auto-detect context window from Tinker's error message
        if "context window" in err_str and proxy._max_ctx == 0:
            import re as _re
            m = _re.search(r'> (\d+)', err_str)
            if m:
                proxy._max_ctx = int(m.group(1))
                logger.info(f"Auto-detected context window: {proxy._max_ctx}")
                max_tokens = min(max_tokens_requested,
                                 proxy._max_ctx - est_prompt_len - 64)
                max_tokens = max(max_tokens, 1)
                try:
                    return await _sample_chat_completion(
                        sampling_client=proxy._client,
                        renderer=proxy._renderer,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        model_name=proxy._model_name,
                    )
                except Exception as e2:
                    logger.error(f"Sampling retry failed: {e2}")
                    return JSONResponse(
                        {"type": "error", "error": {"type": "api_error", "message": str(e2)}},
                        status_code=500,
                    )
        logger.error(f"Sampling error: {e}")
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": err_str}},
            status_code=500,
        )


async def _sample_deepinfra(
    proxy,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_tokens_requested: int,
    est_prompt_len: int,
) -> _SamplingResult | JSONResponse:
    """Sample using DeepInfra API with logprobs."""
    try:
        return await _deepinfra_sample_chat_completion(
            renderer=proxy._renderer,
            tokenizer=proxy._tokenizer,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model_name=proxy._model_name,
            deepinfra_model=proxy._deepinfra_model,
            deepinfra_api_key=proxy._deepinfra_api_key,
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"DeepInfra HTTP error: {e.response.status_code} {e.response.text[:500]}")
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": f"DeepInfra: {e.response.status_code}"}},
            status_code=500,
        )
    except Exception as e:
        import traceback
        logger.error(f"DeepInfra sampling error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            status_code=500,
        )


# ── Proxy ────────────────────────────────────────────────────────────────


class AnthropicTinkerProxy:
    """Anthropic Messages API proxy backed by Tinker SamplingClient or DeepInfra.

    Uses tinker-cookbook's renderer for:
    - Tool schema injection (model-specific format)
    - Tokenization (via renderer.build_generation_prompt, not raw apply_chat_template)
    - Tool call parsing from raw model output

    Captures per-session (messages, token_ids, logprobs) for RL training.

    Supports two sampler backends:
    - "tinker": Tinker SamplingClient (default, uses GPU)
    - "deepinfra": DeepInfra API (cheaper, pay-per-token, returns logprobs on tool calls)
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        model_name: str = "default",
        base_model: str | None = None,
        sampler_backend: str = "tinker",
        deepinfra_model: str = "Qwen/Qwen3.5-35B-A3B",
        deepinfra_api_key: str = "",
        deepinfra_max_ctx: int = 262144,
    ):
        self._client = sampling_client
        self._model_name = model_name
        self._sampler_backend = sampler_backend

        # DeepInfra config
        self._deepinfra_model = deepinfra_model
        self._deepinfra_api_key = deepinfra_api_key

        # Resolve base model for renderer
        self._base_model = base_model or sampling_client.get_base_model()
        self._tokenizer = sampling_client.get_tokenizer()
        renderer_name = get_recommended_renderer_name(self._base_model)
        self._renderer = get_renderer(renderer_name, self._tokenizer)

        # Context window limit. Auto-detected from Tinker's error response
        # on first overflow, or set explicitly via max_ctx parameter.
        if sampler_backend == "deepinfra":
            self._max_ctx = deepinfra_max_ctx
        else:
            self._max_ctx = 0  # 0 = not yet detected

        self._sessions: dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        if sampler_backend == "deepinfra":
            logger.info(
                f"Using DeepInfra sampler: model={deepinfra_model} "
                f"max_ctx={deepinfra_max_ctx}"
            )
        else:
            logger.info("Using Tinker SamplingClient sampler")

    def update_client(self, new_client: tinker.SamplingClient) -> None:
        """Hot-swap SamplingClient after weight sync.

        Only relevant for tinker backend. For deepinfra, this is a no-op
        for the sampler but we still update the tokenizer for datum construction.
        """
        self._client = new_client
        self._tokenizer = new_client.get_tokenizer()

    def create_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionData(session_id=session_id)

    def get_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def pop_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def start(self, host: str = "127.0.0.1", port: int = 8321) -> None:
        config = uvicorn.Config(self._app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                break
            time.sleep(0.05)
        logger.info("AnthropicTinkerProxy listening on %s:%d", host, port)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def _resolve_session_id(self, request_headers: dict) -> str | None:
        """Extract session_id from the API key header."""
        api_key = request_headers.get("x-api-key", "")
        if not api_key:
            auth = request_headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                api_key = auth[7:].strip()
        if not api_key:
            logger.debug("No API key found in request headers")
            return None
        with self._lock:
            if api_key in self._sessions:
                return api_key
            else:
                registered = list(self._sessions.keys())
                logger.warning(
                    f"API key from request not in sessions. "
                    f"received='{api_key[:16]}...' registered={[k[:16]+'...' for k in registered]}"
                )
        return None

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        proxy = self

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": proxy._model_name, "object": "model"}]}

        @app.post("/v1/messages")
        async def anthropic_messages(request: Request):
            """Anthropic Messages API endpoint.

            1. Translate Anthropic → OpenAI format
            2. Route through tinker-cookbook renderer (handles tools + tokenization + parsing)
            3. Capture (token_ids, logprobs) for training
            4. Translate response → Anthropic format
            """
            if proxy._client is None:
                return JSONResponse({"error": "No sampling client"}, status_code=503)

            body = await request.json()
            is_streaming = body.get("stream", False)

            # Translate Anthropic request → OpenAI messages + tools
            try:
                openai_req = _translate_anthropic_to_openai(body)
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return JSONResponse(
                    {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
                    status_code=400,
                )

            messages = openai_req.get("messages", [])
            tools = openai_req.get("tools", None)

            # Cap max_tokens so prompt + max_tokens <= model context window.
            # Claude Code often requests max_tokens=32000+ which overflows.
            try:
                tinker_msgs = openai_messages_to_tinker(messages)
                if tools:
                    tinker_msgs = _prepare_messages_with_tools(proxy._renderer, tinker_msgs, tools)
                est_prompt = proxy._renderer.build_generation_prompt(tinker_msgs)
                est_prompt_len = len(est_prompt.to_ints())
            except Exception:
                est_prompt_len = 0

            max_tokens_requested = body.get("max_tokens", 4096)
            if est_prompt_len > 0 and proxy._max_ctx > 0:
                remaining = proxy._max_ctx - est_prompt_len - 64
                if remaining <= 0:
                    logger.warning(
                        f"Prompt ({est_prompt_len} tokens) exceeds context window "
                        f"({proxy._max_ctx}). Returning context_length error."
                    )
                    return JSONResponse({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"prompt is too long: {est_prompt_len} tokens > {proxy._max_ctx} context window",
                        },
                    }, status_code=400)
                max_tokens = min(max_tokens_requested, remaining)
            else:
                max_tokens = max_tokens_requested

            temperature = body.get("temperature", 1.0)
            top_p = body.get("top_p", 1.0)

            # ── Dispatch to sampler backend ──
            if proxy._sampler_backend == "deepinfra":
                result = await _sample_deepinfra(
                    proxy, messages, tools, temperature, top_p,
                    max_tokens, max_tokens_requested, est_prompt_len,
                )
            else:
                result = await _sample_tinker(
                    proxy, messages, tools, temperature, top_p,
                    max_tokens, max_tokens_requested, est_prompt_len,
                )

            if isinstance(result, JSONResponse):
                return result

            # Extract results
            prompt_token_ids = result.prompt_token_ids
            output_token_ids = result.completion_token_ids
            output_logprobs = list(result.logprobs) if result.logprobs else []

            if not output_logprobs:
                logger.error("Sampler returned NO logprobs — this will break importance sampling loss.")

            # Build the Anthropic response from the RAW decoded text.
            # Critical: we decode the raw tokens ourselves instead of using
            # the renderer's parsed content, because parse_response() →
            # render_message() is NOT a lossless round-trip: it strips
            # <think>/<tool_call> tags and re-adds them with different
            # whitespace, causing re-tokenization to produce different tokens.
            #
            # Instead, we:
            # 1. Decode raw tokens → exact text the model generated
            # 2. Extract tool_calls via regex (without modifying the text)
            # 3. Build Anthropic content blocks from the raw text
            raw_text = proxy._tokenizer.decode(
                output_token_ids, skip_special_tokens=True,
            )

            # Use the renderer's parsed result ONLY for structured tool_calls
            # (it handles JSON parsing robustly). But keep the raw text for content.
            parsed_msg = result.parsed_message
            tool_calls_parsed = parsed_msg.get("tool_calls") if parsed_msg else None

            # Determine finish reason from parsed result
            if tool_calls_parsed:
                finish_reason = "tool_calls"
            elif result.parse_success:
                finish_reason = "stop"
            else:
                finish_reason = "length"

            # Build Anthropic response with raw text content + parsed tool_use blocks.
            # For the text content, we strip <tool_call>...</tool_call> blocks
            # (since they become separate tool_use blocks) but preserve everything
            # else exactly (including <think> tags, whitespace, etc.)
            import re as _re
            text_for_anthropic = _re.sub(
                r'<tool_call>.*?</tool_call>', '', raw_text, flags=_re.DOTALL
            ).rstrip()

            content_blocks = []
            if text_for_anthropic:
                content_blocks.append({"type": "text", "text": text_for_anthropic})

            if tool_calls_parsed:
                for tc in tool_calls_parsed:
                    try:
                        input_obj = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        input_obj = {"raw": tc.function.arguments}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id or f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": tc.function.name,
                        "input": input_obj,
                    })

            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})

            stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

            anthropic_resp = {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": body.get("model", proxy._model_name),
                "stop_reason": stop_reason_map.get(finish_reason, "end_turn"),
                "stop_sequence": None,
                "usage": {
                    "input_tokens": len(prompt_token_ids),
                    "output_tokens": len(output_token_ids),
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }

            # Capture interaction for training
            session_id = proxy._resolve_session_id(dict(request.headers))
            if session_id:
                interaction = SessionInteraction(
                    interaction_id=f"msg_{uuid.uuid4().hex[:12]}",
                    messages=messages,
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=output_token_ids,
                    output_logprobs=output_logprobs,
                    finish_reason=finish_reason,
                )
                with proxy._lock:
                    if session_id in proxy._sessions:
                        proxy._sessions[session_id].interactions.append(interaction)

            if is_streaming:
                return StreamingResponse(
                    _anthropic_sse_stream(anthropic_resp),
                    media_type="text/event-stream",
                )

            return JSONResponse(anthropic_resp)

        return app


# ── Anthropic SSE streaming ──────────────────────────────────────────────


async def _anthropic_sse_stream(response: dict[str, Any]):
    """Convert a complete Anthropic message to SSE events."""
    msg_id = response.get("id", f"msg_{uuid.uuid4().hex[:12]}")
    model = response.get("model", "default")
    usage = response.get("usage", {})
    content = response.get("content", [])

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': usage.get('input_tokens', 0), 'output_tokens': 0}}})}\n\n"

    for idx, block in enumerate(content):
        if block.get("type") == "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'text_delta', 'text': block['text']}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
        elif block.get("type") == "tool_use":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': response.get('stop_reason', 'end_turn'), 'stop_sequence': None}, 'usage': {'output_tokens': usage.get('output_tokens', 0)}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
