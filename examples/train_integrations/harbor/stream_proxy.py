"""
Anthropic-to-OpenAI proxy for Claude Code → vLLM.

Claude Code speaks Anthropic Messages API (/v1/messages).
vLLM speaks OpenAI Chat Completions API (/v1/chat/completions).

This proxy:
1. Receives Anthropic Messages requests from Claude Code (via ngrok)
2. Converts to OpenAI Chat Completions format (including tools)
3. Sends to vLLM (non-streaming)
4. Converts the OpenAI response back to Anthropic SSE streaming format
"""

import json
import sys
import uuid
from aiohttp import web, ClientSession

VLLM_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def anthropic_tools_to_openai(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


def anthropic_messages_to_openai(messages: list) -> list:
    """Convert Anthropic messages to OpenAI format."""
    openai_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Anthropic content blocks
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif btype == "tool_result":
                    tool_result_content = block.get("content", "")
                    if isinstance(tool_result_content, list):
                        tool_result_content = "\n".join(
                            b.get("text", "") for b in tool_result_content if isinstance(b, dict)
                        )
                    tool_results.append({
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tool_result_content),
                    })

            if role == "assistant":
                m = {"role": "assistant"}
                if text_parts:
                    m["content"] = "\n".join(text_parts)
                if tool_calls:
                    m["tool_calls"] = tool_calls
                    if not text_parts:
                        m["content"] = None
                openai_msgs.append(m)
            elif tool_results:
                for tr in tool_results:
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["content"],
                    })
            else:
                openai_msgs.append({"role": role, "content": "\n".join(text_parts)})
    return openai_msgs


def openai_response_to_anthropic(data: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})

    content_blocks = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []) or []:
        func = tc.get("function", {})
        try:
            inp = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            inp = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": func.get("name", ""),
            "input": inp,
        })

    finish = choice.get("finish_reason", "end_turn")
    stop_reason = "tool_use" if finish == "tool_calls" else "end_turn" if finish == "stop" else finish

    return {
        "id": data.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    try:
        body_json = json.loads(body) if body else {}
    except json.JSONDecodeError:
        body_json = {}

    is_streaming = body_json.get("stream", False)
    model = body_json.get("model", "")
    anthropic_tools = body_json.get("tools", [])
    anthropic_messages = body_json.get("messages", [])

    # Convert Anthropic → OpenAI
    openai_request = {
        "model": model.split("/")[-1] if "/" in model else model,  # strip provider prefix
        "messages": anthropic_messages_to_openai(anthropic_messages),
        "max_tokens": min(body_json.get("max_tokens", 4096), 16384),
        "temperature": body_json.get("temperature", 1.0),
        "stream": False,
    }

    if anthropic_tools:
        openai_request["tools"] = anthropic_tools_to_openai(anthropic_tools)
        openai_request["tool_choice"] = "auto"

    # Optional params
    for key in ("top_p", "top_k"):
        if key in body_json:
            openai_request[key] = body_json[key]

    num_tools = len(anthropic_tools)
    print(f"[proxy] model={model} tools={num_tools} msgs={len(anthropic_messages)} stream={is_streaming}", flush=True)

    url = f"{VLLM_URL}/v1/chat/completions"

    async with ClientSession() as session:
        async with session.post(url, json=openai_request, headers={"Content-Type": "application/json"}) as resp:
            resp_body = await resp.read()

            try:
                data = json.loads(resp_body)
            except json.JSONDecodeError:
                return web.Response(body=resp_body, status=resp.status,
                                    content_type=resp.content_type)

            if resp.status != 200 or "error" in data:
                print(f"[proxy] vLLM error: {resp.status} {resp_body[:200]}", flush=True)
                return web.Response(body=resp_body, status=resp.status,
                                    content_type="application/json")

            # Convert OpenAI response → Anthropic format
            anthropic_resp = openai_response_to_anthropic(data, model)

            if not is_streaming:
                return web.Response(
                    body=json.dumps(anthropic_resp).encode(),
                    status=200,
                    content_type="application/json",
                )

            # Convert to SSE stream
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            await response.prepare(request)

            # message_start
            msg_start = {
                "type": "message_start",
                "message": {
                    **anthropic_resp,
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": anthropic_resp["usage"]["input_tokens"],
                        "output_tokens": 0,
                    },
                },
            }
            await response.write(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode())

            # content blocks
            for idx, block in enumerate(anthropic_resp.get("content", [])):
                block_type = block.get("type", "text")

                if block_type == "text":
                    start_evt = {"type": "content_block_start", "index": idx,
                                 "content_block": {"type": "text", "text": ""}}
                    await response.write(f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n".encode())

                    text = block.get("text", "")
                    if text:
                        delta_evt = {"type": "content_block_delta", "index": idx,
                                     "delta": {"type": "text_delta", "text": text}}
                        await response.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode())

                    stop_evt = {"type": "content_block_stop", "index": idx}
                    await response.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode())

                elif block_type == "tool_use":
                    start_evt = {"type": "content_block_start", "index": idx,
                                 "content_block": {"type": "tool_use", "id": block.get("id", ""),
                                                   "name": block.get("name", ""), "input": {}}}
                    await response.write(f"event: content_block_start\ndata: {json.dumps(start_evt)}\n\n".encode())

                    inp = json.dumps(block.get("input", {}))
                    delta_evt = {"type": "content_block_delta", "index": idx,
                                 "delta": {"type": "input_json_delta", "partial_json": inp}}
                    await response.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode())

                    stop_evt = {"type": "content_block_stop", "index": idx}
                    await response.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode())

            # message_delta
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": anthropic_resp.get("stop_reason", "end_turn"),
                          "stop_sequence": None},
                "usage": {"output_tokens": anthropic_resp["usage"]["output_tokens"]},
            }
            await response.write(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode())

            await response.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode())

            await response.write_eof()
            return response


async def passthrough_handler(request: web.Request) -> web.Response:
    """Pass through non-POST requests."""
    url = f"{VLLM_URL}{request.path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    async with ClientSession() as session:
        method = getattr(session, request.method.lower())
        async with method(url, headers=headers) as resp:
            body = await resp.read()
            return web.Response(body=body, status=resp.status,
                                content_type=resp.content_type or "application/json")


async def handler(request: web.Request) -> web.Response:
    if request.method == "POST":
        return await proxy_handler(request)
    return await passthrough_handler(request)

app = web.Application()
app.router.add_route("*", "/{path:.*}", handler)

if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 4001
    print(f"Anthropic→OpenAI proxy on :{port}, forwarding to {VLLM_URL}")
    web.run_app(app, port=port, print=None)
