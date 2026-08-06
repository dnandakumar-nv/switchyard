# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation and content helpers for the conversion boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def _text_blocks(content: object, path: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [] if not content else [{"type": "text", "text": content}]
    blocks = _sequence(content, path)
    converted: list[dict[str, Any]] = []
    for index, raw_block in enumerate(blocks):
        block_path = f"{path}[{index}]"
        block = _mapping(raw_block, block_path)
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ValueError(f"{block_path} is not supported; expected a text block")
        converted.append({"type": "text", "text": cast(str, block["text"])})
    return converted


def _langchain_blocks(blocks: list[dict[str, Any]]) -> list[str | dict[Any, Any]]:
    """Widen validated text blocks to LangChain's invariant content-list type."""
    return cast(list[str | dict[Any, Any]], blocks)
