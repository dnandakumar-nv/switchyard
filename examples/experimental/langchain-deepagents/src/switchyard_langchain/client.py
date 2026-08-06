# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt LangChain chat models to the Python-hosted libsy client contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from .conversion import response_from_ai_message, target_invocation_from_request


class LangChainLlmClient:
    """Use a LangChain chat model as a target for a libsy algorithm."""

    def __init__(self, model: BaseChatModel) -> None:
        """Store the target model without opening or owning its transport."""
        self.model = model

    def _model_name(self) -> str:
        for attribute in ("model_name", "model"):
            value = getattr(self.model, attribute, None)
            if isinstance(value, str) and value:
                return value
        return self.model._llm_type

    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Invoke the target with a buffered neutral request and return a neutral response."""
        invocation = target_invocation_from_request(request)
        options = dict(invocation.options)
        stop_value = options.pop("stop", None)
        if stop_value is not None:
            if not isinstance(stop_value, list) or not all(
                isinstance(item, str) for item in stop_value
            ):
                raise ValueError("extensions.fields.stop must be a list of strings")
            stop = cast(list[str], stop_value)
        else:
            stop = None

        model: Any = self.model
        if invocation.tools:
            model = self.model.bind_tools(
                list(invocation.tools),
                tool_choice=cast(Any, invocation.tool_choice),
            )
        response = await model.ainvoke(list(invocation.messages), stop=stop, **options)
        if not isinstance(response, AIMessage):
            raise ValueError(
                f"target returned {type(response).__name__} instead of AIMessage"
            )
        return response_from_ai_message(response, model_name=self._model_name())
