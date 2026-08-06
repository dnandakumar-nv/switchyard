#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run two paid Deep Agent turns through Switchyard Stage routing and OpenRouter."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from switchyard_langchain import LangChainLlmClient, SwitchyardRoutingMiddleware

from switchyard.libsy import LlmTarget, algorithms

EFFICIENT_MODEL_DEFAULT = "openai/gpt-5-mini"
CAPABLE_MODEL_DEFAULT = "anthropic/claude-sonnet-4.6"


@dataclass(frozen=True)
class DemoResult:
    """One routed Deep Agent result displayed by the example."""

    case: str
    selected_model: str
    text: str


def repository_env_path() -> Path:
    """Return the repository-root environment file used by this example."""
    return Path(__file__).resolve().parents[3] / ".env"


def load_repository_environment() -> None:
    """Load repository credentials without overriding caller-provided values."""
    load_dotenv(repository_env_path(), override=False)


def _require_paid_environment() -> None:
    if os.environ.get("SWITCHYARD_LANGCHAIN_E2E") != "1":
        raise RuntimeError(
            "set SWITCHYARD_LANGCHAIN_E2E=1 to acknowledge that this example makes paid calls"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the repository .env or environment")


def _create_models() -> tuple[ChatOpenRouter, ChatOpenRouter]:
    efficient_name = os.environ.get(
        "OPENROUTER_EFFICIENT_MODEL", EFFICIENT_MODEL_DEFAULT
    )
    capable_name = os.environ.get("OPENROUTER_CAPABLE_MODEL", CAPABLE_MODEL_DEFAULT)
    return (
        ChatOpenRouter(model=efficient_name, max_tokens=96),
        ChatOpenRouter(model=capable_name, max_tokens=96),
    )


def _agent(efficient_model: ChatOpenRouter, capable_model: ChatOpenRouter) -> Any:
    router = algorithms.stage_router(
        LlmTarget("capable", LangChainLlmClient(capable_model)),
        LlmTarget("efficient", LangChainLlmClient(efficient_model)),
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    return create_deep_agent(
        model=efficient_model,
        middleware=[SwitchyardRoutingMiddleware(router)],
    )


def _last_ai_message(result: dict[str, object]) -> AIMessage:
    messages = result.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Deep Agent result has no message list")
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise ValueError("Deep Agent result has no AIMessage")


def _demo_result(case: str, result: dict[str, object]) -> DemoResult:
    message = _last_ai_message(result)
    switchyard = message.response_metadata.get("switchyard")
    if not isinstance(switchyard, dict):
        raise ValueError("Deep Agent response has no Switchyard routing metadata")
    selected = switchyard.get("selected_model")
    if not isinstance(selected, str):
        raise ValueError("Deep Agent response has no selected Switchyard model")
    if not message.text.strip():
        raise ValueError("Deep Agent response has no text")
    return DemoResult(case=case, selected_model=selected, text=message.text)


async def run_demo() -> list[DemoResult]:
    """Run the deterministic efficient and capable routing demonstrations."""
    load_repository_environment()
    _require_paid_environment()
    efficient_model, capable_model = _create_models()
    agent = _agent(efficient_model, capable_model)

    simple = await agent.ainvoke({
        "messages": [
            HumanMessage(
                "Reply with one short sentence confirming the simple routing check. "
                "Do not call any tools."
            )
        ]
    })
    failed_tool = await agent.ainvoke({
        "messages": [
            HumanMessage("Help recover from the failed test inspection."),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "call-stage-e2e",
                        "name": "read_file",
                        "args": {"file_path": "tests/failing_test.py"},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                "fatal runtime error: out of memory",
                tool_call_id="call-stage-e2e",
                status="error",
            ),
            HumanMessage(
                "This is a synthetic routing check. Reply with one short sentence "
                "confirming the capable routing check. Do not call any tools."
            ),
        ]
    })
    return [
        _demo_result("simple", cast(dict[str, object], simple)),
        _demo_result("failed-tool", cast(dict[str, object], failed_tool)),
    ]


async def main() -> None:
    """Run the paid demo and print only routing outcomes and response text."""
    results = await run_demo()
    for result in results:
        print(f"{result.case}: selected={result.selected_model}")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
