"""
context.py — Local context gathering. (Day 3-4 implementation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wtf_cli.parser import ErrorInfo


@dataclass
class SourceSnippet:
    file_path: str
    line_number: int
    context_lines: list[str]
    error_line: str


@dataclass
class LocalContext:
    python_version: str | None = None
    python_path: str | None = None
    venv_active: bool = False
    venv_path: str | None = None
    installed_packages: dict[str, str] = field(default_factory=dict)
    requirements_packages: dict[str, str] | None = None
    pyproject_packages: list[str] | None = None
    os_info: str = ""
    env_var_names: list[str] = field(default_factory=list)
    cwd: str = ""
    git_branch: str | None = None
    git_dirty: bool | None = None
    source_snippets: list[SourceSnippet] = field(default_factory=list)
    docker_running: bool | None = None


def gather_context(error_info: ErrorInfo) -> LocalContext:
    """Gather local machine context. Full implementation in Day 3-4."""
    return LocalContext()
