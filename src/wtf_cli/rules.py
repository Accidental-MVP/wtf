"""
rules.py — Tier 1 rule-based diagnosis (no LLM). (Day 5-6 implementation)
"""

from __future__ import annotations

from dataclasses import dataclass

from wtf_cli.context import LocalContext
from wtf_cli.parser import ErrorInfo


@dataclass
class Diagnosis:
    summary: str
    explanation: str
    fix_command: str | None
    confidence: str  # "high" or "medium"
    source: str      # "rule" or "ai"


def diagnose(error_info: ErrorInfo, context: LocalContext) -> Diagnosis | None:
    """Run rules engine. Returns None if no rule matches. Full implementation in Day 5-6."""
    return None
