# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for the experimental LangChain package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src"

sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))
