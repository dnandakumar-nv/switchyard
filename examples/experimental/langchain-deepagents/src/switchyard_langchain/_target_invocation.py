# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohesive target invocation produced by the conversion boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class _TargetInvocation:
    """Validated LangChain inputs derived from one neutral libsy request."""

    messages: tuple[BaseMessage, ...]
    tools: tuple[dict[str, object], ...]
    tool_choice: object | None
    options: Mapping[str, object]
