# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert responses across the LangChain and libsy boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain.messages import AIMessage

from ._conversion_support import _langchain_blocks, _mapping, _sequence, _text_blocks

_FINISH_TO_NEUTRAL = {
    "stop": "end_turn",
    "end_turn": "end_turn",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "content_filter": "content_filter",
    "error": "error",
}
_NEUTRAL_TO_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "error": "error",
    "unknown": "unknown",
}


def _usage_to_neutral(message: AIMessage) -> dict[str, int]:
    usage = message.usage_metadata
    if usage is None:
        return {}
    result = {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    if input_details.get("cache_read") is not None:
        result["cached_input_tokens"] = input_details["cache_read"]
    if input_details.get("cache_creation") is not None:
        result["cache_creation_input_tokens"] = input_details["cache_creation"]
    if output_details.get("reasoning") is not None:
        result["reasoning_tokens"] = output_details["reasoning"]
    return result


def response_from_ai_message(message: AIMessage, *, model_name: str) -> dict[str, object]:
    """Convert a target LangChain aggregate response into libsy's neutral response."""
    content: list[dict[str, object]] = list(_text_blocks(message.content, "response.content"))
    for index, tool_call in enumerate(message.tool_calls):
        call_id = tool_call.get("id")
        name = tool_call.get("name")
        arguments = tool_call.get("args")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ValueError(f"response.tool_calls[{index}] must have string id and name")
        if not isinstance(arguments, Mapping):
            raise ValueError(f"response.tool_calls[{index}].args must be a mapping")
        content.append({
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "arguments": dict(arguments),
        })
    if not content:
        raise ValueError("target response has no assistant text or tool calls")
    raw_finish = message.response_metadata.get("finish_reason")
    if isinstance(raw_finish, str):
        stop_reason = _FINISH_TO_NEUTRAL.get(raw_finish, "unknown")
    else:
        stop_reason = "tool_use" if message.tool_calls else "end_turn"
    return {
        "id": message.id,
        "model": model_name,
        "outputs": [
            {
                "role": "assistant",
                "content": content,
                "stop_reason": stop_reason,
            }
        ],
        "usage": _usage_to_neutral(message),
    }


def _usage_from_neutral(raw_usage: object) -> dict[str, Any] | None:
    usage = _mapping(raw_usage, "usage")
    required = {
        key: usage.get(key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    if not any(isinstance(value, int) for value in required.values()):
        return None
    result: dict[str, Any] = {
        key: value if isinstance(value, int) else 0
        for key, value in required.items()
    }
    input_details: dict[str, int] = {}
    output_details: dict[str, int] = {}
    if usage.get("cached_input_tokens") is not None:
        input_details["cache_read"] = cast(int, usage["cached_input_tokens"])
    if usage.get("cache_creation_input_tokens") is not None:
        input_details["cache_creation"] = cast(int, usage["cache_creation_input_tokens"])
    if usage.get("reasoning_tokens") is not None:
        output_details["reasoning"] = cast(int, usage["reasoning_tokens"])
    if input_details:
        result["input_token_details"] = input_details
    if output_details:
        result["output_token_details"] = output_details
    return result


def ai_message_from_response(
    response: Mapping[str, object],
    decisions: Sequence[Mapping[str, object]],
) -> AIMessage:
    """Build the routed LangChain response and attach the complete decision trace."""
    outputs = _sequence(response.get("outputs"), "outputs")
    if not outputs:
        raise ValueError("neutral response has no outputs")
    output = _mapping(outputs[0], "outputs[0]")
    if output.get("role") != "assistant":
        raise ValueError("outputs[0].role must be assistant")
    blocks = _sequence(output.get("content"), "outputs[0].content")
    text_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for index, raw_block in enumerate(blocks):
        path = f"outputs[0].content[{index}]"
        block = _mapping(raw_block, path)
        if block.get("type") == "text":
            text_blocks.extend(_text_blocks([block], "outputs[0].content"))
        elif block.get("type") == "tool_call":
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise ValueError(f"{path} must have string id and name")
            if not isinstance(arguments, Mapping):
                raise ValueError(f"{path}.arguments must be a mapping")
            tool_calls.append({
                "id": call_id,
                "name": name,
                "args": dict(arguments),
                "type": "tool_call",
            })
        else:
            raise ValueError(f"{path} is not supported")
    if not text_blocks and not tool_calls:
        raise ValueError("neutral response has no assistant text or tool calls")

    decision_trace = [dict(decision) for decision in decisions]
    switchyard: dict[str, object] = {"decisions": decision_trace}
    if decision_trace:
        selected_model = decision_trace[-1].get("selected_model")
        if selected_model is not None:
            switchyard["selected_model"] = selected_model
    metadata: dict[str, object] = {"switchyard": switchyard}
    model = response.get("model")
    if isinstance(model, str):
        metadata["model_name"] = model
    stop_reason = output.get("stop_reason")
    if isinstance(stop_reason, str):
        metadata["finish_reason"] = _NEUTRAL_TO_FINISH.get(stop_reason, "unknown")

    message_id = response.get("id")
    assistant_content: str | list[str | dict[Any, Any]] = (
        _langchain_blocks(text_blocks) if text_blocks else ""
    )
    return AIMessage(
        content=assistant_content,
        tool_calls=tool_calls,
        id=message_id if isinstance(message_id, str) else None,
        response_metadata=metadata,
        usage_metadata=_usage_from_neutral(response.get("usage", {})),
    )
