# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contract tests for the experimental LangChain integration."""

from __future__ import annotations

import tomllib
from pathlib import Path

from conftest import PACKAGE_ROOT, REPOSITORY_ROOT


def _project() -> dict[str, object]:
    return tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())["project"]


def test_package_metadata_declares_installable_public_contract() -> None:
    project = _project()

    assert project["name"] == "switchyard-langchain"
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.12"
    assert project["dependencies"] == [
        "nemo-switchyard>=0.2.0",
        "langchain>=1.3.14,<2",
    ]
    assert project["optional-dependencies"] == {
        "deepagents": ["deepagents>=0.7.4,<0.8"],
        "openrouter": [
            "deepagents>=0.7.4,<0.8",
            "langchain-openrouter>=0.2.7,<0.3",
            "python-dotenv>=1,<2",
        ],
    }


def test_pytest_defaults_never_select_paid_e2e() -> None:
    config = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())

    assert config["tool"]["pytest"]["ini_options"]["addopts"] == '-m "not e2e"'
    assert config["tool"]["pytest"]["ini_options"]["markers"] == [
        "e2e: makes paid calls to real OpenRouter models",
    ]


def test_env_example_contains_names_but_no_secret() -> None:
    assert (PACKAGE_ROOT / ".env.example").read_text() == (
        "OPENROUTER_API_KEY=\n"
        "OPENROUTER_EFFICIENT_MODEL=openai/gpt-5-mini\n"
        "OPENROUTER_CAPABLE_MODEL=anthropic/claude-sonnet-4.6\n"
    )


def test_all_python_files_have_spdx_header() -> None:
    for path in PACKAGE_ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        first_lines = path.read_text().splitlines()[:4]
        assert any("SPDX-FileCopyrightText:" in line for line in first_lines), path
        assert any(line == "# SPDX-License-Identifier: Apache-2.0" for line in first_lines), path


def test_public_package_exports_only_the_two_adapters() -> None:
    import switchyard_langchain

    assert switchyard_langchain.__all__ == [
        "LangChainLlmClient",
        "SwitchyardRoutingMiddleware",
    ]


def test_package_root_is_the_expected_directory() -> None:
    assert Path(PACKAGE_ROOT).name == "langchain-deepagents"


def test_lockfile_uses_the_current_editable_switchyard_version() -> None:
    root_version = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())[
        "project"
    ]["version"]
    lock = tomllib.loads((PACKAGE_ROOT / "uv.lock").read_text())
    switchyard = next(
        package for package in lock["package"] if package["name"] == "nemo-switchyard"
    )

    assert switchyard["version"] == root_version
    assert switchyard["source"] == {"editable": "../../../"}
