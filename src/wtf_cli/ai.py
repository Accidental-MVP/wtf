"""
ai.py — Tier 2 LLM-powered diagnosis via litellm. (Day 7 implementation)
"""

from __future__ import annotations

from wtf_cli.context import LocalContext
from wtf_cli.parser import ErrorInfo
from wtf_cli.rules import Diagnosis


def diagnose(
    error_info: ErrorInfo,
    context: LocalContext,
    model: str | None = None,
) -> Diagnosis:
    """Call LLM for diagnosis. Full implementation in Day 7."""
    raise NotImplementedError("AI diagnosis will be implemented in Day 7")
