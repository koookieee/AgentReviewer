"""
Tokenize multi-turn Harbor chat histories for RL training.

Ported from SkyRL's ``skyrl/train/generators/utils.py``.
Produces ``(response_ids, loss_mask, rollout_logprobs)`` from the chat
history returned by Harbor ``Trial.run()``.
"""

from __future__ import annotations

from typing import Optional


def get_generation_prompt_ids(
    tokenizer,
    chat_template: Optional[str] = None,
) -> list[int]:
    """Return the token IDs for the generation prompt (e.g. ``<|im_start|>assistant\\n``)."""
    empty_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        return_dict=False,
        chat_template=chat_template,
    )
    empty_user_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        chat_template=chat_template,
    )
    return empty_user_gen[len(empty_user):]


def encode_messages_subset(
    messages: list[dict],
    tokenizer,
    chat_template: Optional[str] = None,
) -> list[int]:
    """Tokenize *messages* as if they are a continuation of an existing conversation.

    Uses the "fixed base approach" to prevent the tokenizer from injecting
    default system prompts or merging boundary tokens.
    """
    assert messages, "messages list cannot be empty"
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]
    base_ids = tokenizer.apply_chat_template(
        base, add_generation_prompt=False, tokenize=True, return_dict=False,
        chat_template=chat_template,
    )
    full_ids = tokenizer.apply_chat_template(
        base + messages, add_generation_prompt=False, tokenize=True, return_dict=False,
        chat_template=chat_template,
    )
    return full_ids[len(base_ids):]


def get_response_ids_and_loss_mask(
    messages: list[dict],
    tokenizer,
    assistant_logprobs: Optional[list[list[float]]] = None,
    chat_template: Optional[str] = None,
) -> tuple[list[int], list[int], Optional[list[float]]]:
    """Convert the response portion of a Harbor chat history into training data.

    Parameters
    ----------
    messages
        The response messages (everything after the first user message).
        Each dict must have ``role`` and ``content``.
    tokenizer
        A HuggingFace tokenizer with ``apply_chat_template`` and ``eos_token_id``.
    assistant_logprobs
        Optional per-assistant-message logprobs (from the sampling model).
    chat_template
        Optional custom Jinja chat template string.

    Returns
    -------
    response_ids
        Token IDs for all response messages concatenated.
    loss_mask
        Per-token mask: 1 for assistant-generated tokens, 0 for everything else.
    rollout_logprobs
        Per-token logprobs aligned with *response_ids* (None if *assistant_logprobs* is None).
    """
    assert messages, "messages list cannot be empty"

    gen_prompt_ids = get_generation_prompt_ids(tokenizer, chat_template=chat_template)

    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_logprobs: list[float] | None = None if assistant_logprobs is None else []
    assistant_msg_idx = 0

    for msg in messages:
        cur_ids = encode_messages_subset([msg], tokenizer, chat_template=chat_template)
        response_ids.extend(cur_ids)

        if msg["role"] in ("user", "system", "tool"):
            # Observation / tool-result tokens → loss_mask = 0
            loss_mask.extend([0] * len(cur_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend([0.0] * len(cur_ids))

        elif msg["role"] == "assistant":
            # Split into: generation prompt | generated content | post-EOS
            assert cur_ids[:len(gen_prompt_ids)] == gen_prompt_ids, (
                f"Assistant msg tokens should start with generation prompt. "
                f"Expected {gen_prompt_ids}, got {cur_ids[:len(gen_prompt_ids)]}"
            )

            if tokenizer.eos_token_id in cur_ids:
                last_eos = len(cur_ids) - 1 - cur_ids[::-1].index(tokenizer.eos_token_id)
                generated = cur_ids[len(gen_prompt_ids):last_eos + 1]
                after_eos = cur_ids[last_eos + 1:]
            else:
                generated = cur_ids[len(gen_prompt_ids):]
                after_eos = []

            # Generation prompt → 0
            loss_mask.extend([0] * len(gen_prompt_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend([0.0] * len(gen_prompt_ids))

            # Actually generated content → 1
            loss_mask.extend([1] * len(generated))
            if assistant_logprobs is not None and rollout_logprobs is not None:
                if assistant_msg_idx < len(assistant_logprobs):
                    msg_lp = assistant_logprobs[assistant_msg_idx]
                    if len(msg_lp) == len(generated):
                        rollout_logprobs.extend(msg_lp)
                    else:
                        # Length mismatch — fall back to zeros
                        rollout_logprobs.extend([0.0] * len(generated))
                else:
                    rollout_logprobs.extend([0.0] * len(generated))

            # Post-EOS tokens → 0
            loss_mask.extend([0] * len(after_eos))
            if rollout_logprobs is not None:
                rollout_logprobs.extend([0.0] * len(after_eos))

            assistant_msg_idx += 1
        else:
            # Unknown role — treat as observation
            loss_mask.extend([0] * len(cur_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend([0.0] * len(cur_ids))

    assert len(loss_mask) == len(response_ids)
    if rollout_logprobs is not None:
        assert len(rollout_logprobs) == len(response_ids)
    return response_ids, loss_mask, rollout_logprobs
