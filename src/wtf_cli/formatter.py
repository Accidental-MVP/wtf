"""
formatter.py — Rich terminal output formatting. (Day 8-9 implementation)
"""

from __future__ import annotations

from wtf_cli.context import LocalContext
from wtf_cli.parser import ErrorInfo
from wtf_cli.rules import Diagnosis
from wtf_cli.runner import CommandResult


def print_diagnosis(
    command_result: CommandResult,
    error_info: ErrorInfo,
    context: LocalContext,
    diagnosis: Diagnosis,
    show_context: bool = False,
) -> None:
    """Format and print the diagnosis panel. Full implementation in Day 8-9."""
    pass
