# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert requests across the LangChain and libsy boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from ._conversion_support import _langchain_blocks, _mapping, _sequence, _text_blocks
from ._target_invocation import _TargetInvocation


def _message_to_neutral(
    message: BaseMessage,
    index: int,
) -> tuple[str, dict[str, object]]:
    path = f"messages[{index}]"
    if isinstance(message, SystemMessage):
        role = message.additional_kwargs.get("__openai_role__", "system")
        if role not in {"system", "developer"}:
            raise ValueError(f"{path} has unsupported system role {role!r}")
        return "instruction", {
            "role": role,
            "content": _text_blocks(message.content, f"{path}.content"),
        }
    if isinstance(message, HumanMessage):
        return "message", {
            "role": "user",
            "content": _text_blocks(message.content, f"{path}.content"),
        }
    if isinstance(message, AIMessage):
        content: list[dict[str, object]] = list(
            _text_blocks(message.content, f"{path}.content")
        )
        for call_index, tool_call in enumerate(message.tool_calls):
            call_path = f"{path}.tool_calls[{call_index}]"
            call_id = tool_call.get("id")
            name = tool_call.get("name")
            arguments = tool_call.get("args")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise ValueError(f"{call_path} must have string id and name")
            if not isinstance(arguments, Mapping):
                raise ValueError(f"{call_path}.args must be a mapping")
            content.append({
                "type": "tool_call",
                "id": call_id,
                "name": name,
                "arguments": dict(arguments),
            })
        if not content:
            raise ValueError(f"{path} has no assistant text or tool calls")
        return "message", {"role": "assistant", "content": content}
    if isinstance(message, ToolMessage):
        return "message", {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": message.tool_call_id,
                    "content": _text_blocks(message.content, f"{path}.content"),
                    "is_error": message.status == "error",
                }
            ],
        }
    raise ValueError(f"{path} has unsupported LangChain message type {message.type!r}")


def _neutral_tool(raw_tool: object, index: int) -> dict[str, object]:
    converted = convert_to_openai_tool(cast(Any, raw_tool))
    function = _mapping(converted.get("function"), f"tools[{index}].function")
    name = function.get("name")
    parameters = function.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, Mapping):
        raise ValueError(f"tools[{index}] must define a function name and parameters")
    result: dict[str, object] = {
        "name": name,
        "description": function.get("description"),
        "parameters": dict(parameters),
    }
    if "strict" in function:
        result["strict"] = function["strict"]
    return result


def _neutral_tool_choice(choice: object) -> dict[str, object] | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice == "any":
            return {"type": "required"}
        if choice in {"auto", "required", "none"}:
            return {"type": choice}
        return {"type": "tool", "data": {"name": choice}}
    value = _mapping(choice, "tool_choice")
    choice_type = value.get("type")
    if choice_type == "function":
        function = _mapping(value.get("function"), "tool_choice.function")
        name = function.get("name")
        if not isinstance(name, str):
            raise ValueError("tool_choice.function.name must be a string")
        return {"type": "tool", "data": {"name": name}}
    if choice_type in {"auto", "required", "none"}:
        return {"type": choice_type}
    raise ValueError("tool_choice is not supported")


def request_from_langchain(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[object],
    tool_choice: object | None,
    model_settings: Mapping[str, object],
    stop: list[str] | None,
) -> dict[str, object]:
    """Build a buffered provider-neutral request from a LangChain model call."""
    if not messages:
        raise ValueError("messages must not be empty")

    instructions: list[dict[str, object]] = []
    turns: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        kind, converted = _message_to_neutral(message, index)
        (instructions if kind == "instruction" else turns).append(converted)

    unsupported = set(model_settings) - {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "response_format",
    }
    if unsupported:
        field = sorted(unsupported)[0]
        raise ValueError(f"model setting {field} is not supported")

    sampling = {
        key: model_settings[key]
        for key in ("temperature", "top_p")
        if model_settings.get(key) is not None
    }
    max_tokens = model_settings.get("max_completion_tokens", model_settings.get("max_tokens"))
    output = {"max_output_tokens": max_tokens} if max_tokens is not None else {}
    response_format = model_settings.get("response_format")
    if response_format is not None:
        output["response_format"] = dict(
            _mapping(response_format, "model_settings.response_format")
        )
    reasoning_value = model_settings.get("reasoning")
    if reasoning_value is not None:
        reasoning_mapping = _mapping(reasoning_value, "model_settings.reasoning")
        effort = reasoning_mapping.get("effort")
    else:
        effort = model_settings.get("reasoning_effort")
    reasoning = {"effort": effort} if effort is not None else {}

    request: dict[str, object] = {
        "model": "auto",
        "instructions": instructions,
        "messages": turns,
        "tools": [_neutral_tool(tool, index) for index, tool in enumerate(tools)],
        "sampling": sampling,
        "output": output,
        "reasoning": reasoning,
        "stream": False,
    }
    normalized_choice = _neutral_tool_choice(tool_choice)
    if normalized_choice is not None:
        request["tool_choice"] = normalized_choice
    if stop is not None:
        request["extensions"] = {"fields": {"stop": stop}}
    return request


def _instruction_message(raw_instruction: object, index: int) -> SystemMessage:
    path = f"instructions[{index}]"
    instruction = _mapping(raw_instruction, path)
    role = instruction.get("role")
    if role not in {"system", "developer"}:
        raise ValueError(f"{path}.role must be system or developer")
    blocks = _text_blocks(instruction.get("content"), f"{path}.content")
    text = "\n".join(block["text"] for block in blocks)
    kwargs = {"__openai_role__": "developer"} if role == "developer" else {}
    return SystemMessage(content=text, additional_kwargs=kwargs)


def messages_from_request(request: Mapping[str, object]) -> list[BaseMessage]:
    """Convert a neutral libsy request into target LangChain messages."""
    instructions = _sequence(request.get("instructions", []), "instructions")
    messages: list[BaseMessage] = [
        _instruction_message(raw, index) for index, raw in enumerate(instructions)
    ]
    turns = _sequence(request.get("messages"), "messages")
    for index, raw_turn in enumerate(turns):
        path = f"messages[{index}]"
        turn = _mapping(raw_turn, path)
        role = turn.get("role")
        content = _sequence(turn.get("content"), f"{path}.content")
        if role in {"system", "developer"}:
            messages.append(_instruction_message(turn, index))
        elif role == "user":
            messages.append(
                HumanMessage(
                    content=_langchain_blocks(
                        _text_blocks(content, f"{path}.content")
                    )
                )
            )
        elif role == "assistant":
            text_blocks: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            for block_index, raw_block in enumerate(content):
                block_path = f"{path}.content[{block_index}]"
                block = _mapping(raw_block, block_path)
                if block.get("type") == "text":
                    text_blocks.extend(_text_blocks([block], f"{path}.content"))
                elif block.get("type") == "tool_call":
                    call_id = block.get("id")
                    name = block.get("name")
                    arguments = block.get("arguments")
                    if not isinstance(call_id, str) or not isinstance(name, str):
                        raise ValueError(f"{block_path} must have string id and name")
                    if not isinstance(arguments, Mapping):
                        raise ValueError(f"{block_path}.arguments must be a mapping")
                    tool_calls.append({
                        "id": call_id,
                        "name": name,
                        "args": dict(arguments),
                        "type": "tool_call",
                    })
                else:
                    raise ValueError(f"{block_path} is not supported")
            assistant_content: str | list[str | dict[Any, Any]] = (
                _langchain_blocks(text_blocks) if text_blocks else ""
            )
            messages.append(AIMessage(content=assistant_content, tool_calls=tool_calls))
        elif role == "tool":
            for block_index, raw_block in enumerate(content):
                block_path = f"{path}.content[{block_index}]"
                block = _mapping(raw_block, block_path)
                if block.get("type") != "tool_result":
                    raise ValueError(f"{block_path} must be a tool_result")
                call_id = block.get("tool_call_id")
                if not isinstance(call_id, str):
                    raise ValueError(f"{block_path}.tool_call_id must be a string")
                result_content = _text_blocks(block.get("content"), f"{block_path}.content")
                messages.append(
                    ToolMessage(
                        content=_langchain_blocks(result_content),
                        tool_call_id=call_id,
                        status="error" if block.get("is_error") is True else "success",
                    )
                )
        else:
            raise ValueError(f"{path}.role {role!r} is not supported")
    if not messages:
        raise ValueError("messages must not be empty")
    return messages


def model_options_from_request(
    request: Mapping[str, object],
) -> tuple[list[dict[str, object]], object | None, dict[str, object]]:
    """Extract LangChain tools, tool choice, and invocation options from a neutral request."""
    tools: list[dict[str, object]] = []
    for index, raw_tool in enumerate(_sequence(request.get("tools", []), "tools")):
        path = f"tools[{index}]"
        tool = _mapping(raw_tool, path)
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            raise ValueError(f"{path} must define a function name and parameters")
        function: dict[str, object] = {
            "name": name,
            "parameters": dict(parameters),
        }
        if tool.get("description") is not None:
            function["description"] = tool["description"]
        if tool.get("strict") is not None:
            function["strict"] = tool["strict"]
        tools.append({"type": "function", "function": function})

    raw_choice = request.get("tool_choice")
    tool_choice: object | None = None
    if raw_choice is not None:
        choice = _mapping(raw_choice, "tool_choice")
        choice_type = choice.get("type")
        if choice_type in {"auto", "required", "none"}:
            tool_choice = choice_type
        elif choice_type == "tool":
            data = _mapping(choice.get("data"), "tool_choice.data")
            name = data.get("name")
            if not isinstance(name, str):
                raise ValueError("tool_choice.data.name must be a string")
            tool_choice = {"type": "function", "function": {"name": name}}
        else:
            raise ValueError("tool_choice is not supported")

    options: dict[str, object] = {}
    sampling = _mapping(request.get("sampling", {}), "sampling")
    if sampling.get("top_k") is not None:
        raise ValueError("sampling.top_k is not supported")
    for key in ("temperature", "top_p"):
        if sampling.get(key) is not None:
            options[key] = sampling[key]
    output = _mapping(request.get("output", {}), "output")
    if output.get("max_output_tokens") is not None:
        options["max_tokens"] = output["max_output_tokens"]
    if output.get("response_format") is not None:
        options["response_format"] = output["response_format"]
    reasoning = _mapping(request.get("reasoning", {}), "reasoning")
    if reasoning.get("raw") is not None:
        raise ValueError("reasoning.raw is not supported")
    if reasoning.get("effort") is not None:
        options["reasoning"] = {"effort": reasoning["effort"]}
    extensions = _mapping(request.get("extensions", {}), "extensions")
    fields = _mapping(extensions.get("fields", {}), "extensions.fields")
    if fields.get("stop") is not None:
        options["stop"] = fields["stop"]
    return tools, tool_choice, options


def target_invocation_from_request(
    request: Mapping[str, object],
) -> _TargetInvocation:
    """Build one validated LangChain invocation from a neutral libsy request."""
    messages = messages_from_request(request)
    tools, tool_choice, options = model_options_from_request(request)
    return _TargetInvocation(
        messages=tuple(messages),
        tools=tuple(tools),
        tool_choice=tool_choice,
        options=dict(options),
    )
