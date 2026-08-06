# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate supported LangChain content blocks to and from Switchyard dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage

from .utils.type_validation import require_mapping, require_sequence


class _ContentMapper:
    """Translate the ordered content blocks shared by requests and responses.

    Switchyard currently represents text and parsed tool calls. Keeping this
    policy here prevents request history and model responses from drifting apart.
    """

    @classmethod
    def to_switchyard(
        cls,
        message: BaseMessage,
        *,
        path: str,
        allow_tool_calls: bool = False,
    ) -> list[dict[str, object]]:
        """Translate supported LangChain blocks into Switchyard blocks."""
        if isinstance(message, AIMessage) and message.invalid_tool_calls:
            raise ValueError(
                f"{path}.invalid_tool_calls is not supported; tool arguments must be valid JSON"
            )

        blocks: list[dict[str, object]] = []
        for index, block in enumerate(message.content_blocks):
            block_path = f"{path}.content[{index}]"
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{block_path}.text must be a string")
                blocks.append({"type": "text", "text": text})
                continue

            if block_type == "tool_call" and allow_tool_calls:
                call_id = block.get("id")
                name = block.get("name")
                arguments = block.get("args")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise ValueError(f"{block_path} must have string id and name")
                if not isinstance(arguments, Mapping):
                    raise ValueError(f"{block_path}.args must be a mapping")
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": call_id,
                        "name": name,
                        "arguments": dict(arguments),
                    }
                )
                continue

            expected = "text or tool_call" if allow_tool_calls else "text"
            raise ValueError(f"{block_path} is not supported; expected {expected}")

        if allow_tool_calls and not blocks:
            raise ValueError(f"{path} has no assistant text or tool calls")
        return blocks

    @classmethod
    def from_switchyard(
        cls,
        content: object,
        *,
        path: str,
        allow_tool_calls: bool = False,
    ) -> list[dict[str, Any]]:
        """Translate supported Switchyard blocks into LangChain blocks."""
        blocks: list[dict[str, Any]] = []
        for index, raw_block in enumerate(require_sequence(content, path)):
            block_path = f"{path}[{index}]"
            block = require_mapping(raw_block, block_path)
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{block_path}.text must be a string")
                blocks.append({"type": "text", "text": text})
                continue

            if block_type == "tool_call" and allow_tool_calls:
                call_id = block.get("id")
                name = block.get("name")
                arguments = block.get("arguments")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise ValueError(f"{block_path} must have string id and name")
                if not isinstance(arguments, Mapping):
                    raise ValueError(f"{block_path}.arguments must be a mapping")
                blocks.append(
                    {
                        "type": "tool_call",
                        "id": call_id,
                        "name": name,
                        "args": dict(arguments),
                    }
                )
                continue

            expected = "text or tool_call" if allow_tool_calls else "text"
            raise ValueError(f"{block_path} is not supported; expected {expected}")

        if allow_tool_calls and not blocks:
            raise ValueError(f"{path} has no assistant text or tool calls")
        return blocks
