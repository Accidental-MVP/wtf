"""
parser.py — Error/traceback parsing. (Day 2 implementation)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileReference:
    file_path: str
    line_number: int
    function_name: str | None = None


@dataclass
class ErrorInfo:
    error_type: str
    error_message: str
    traceback_lines: list[str]
    file_references: list[FileReference]
    language: str  # "python", "node", "generic"
    raw_error: str


def parse_error(stderr: str, stdout: str, command: str) -> ErrorInfo:
    """Parse raw stderr/stdout into structured ErrorInfo. Full implementation in Day 2."""
    raw = stderr or stdout or ""
    return ErrorInfo(
        error_type="UnknownError",
        error_message=raw.strip().splitlines()[-1] if raw.strip() else "Unknown error",
        traceback_lines=raw.splitlines(),
        file_references=[],
        language="generic",
        raw_error=raw,
    )
