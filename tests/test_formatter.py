"""
test_formatter.py — Tests for formatter.py.

Uses Rich's Console(file=StringIO()) to capture rendered output without
ANSI codes, then asserts on the plain text content.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from wtf_cli.context import LocalContext, SourceSnippet
from wtf_cli.formatter import (
    _find_contradictions,
    _normalize_pkg,
    print_context,
    print_diagnosis,
    print_dry_run,
    print_no_diagnosis,
)
from wtf_cli.parser import ErrorInfo, FileReference
from wtf_cli.rules import Diagnosis
from wtf_cli.runner import CommandResult


# ── Helpers ───────────────────────────────────────────────────────────────────────

def make_console() -> tuple[Console, StringIO]:
    """Return a (console, buffer) pair that captures plain text output."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, highlight=False, markup=False)
    return con, buf


def captured(con: Console, buf: StringIO) -> str:
    con.file.flush()
    return buf.getvalue()


# ── Builder helpers ───────────────────────────────────────────────────────────────

def ei(
    error_type: str = "ModuleNotFoundError",
    error_message: str = "No module named 'pydantic'",
    *,
    traceback_lines: list[str] | None = None,
    file_references: list[FileReference] | None = None,
    language: str = "python",
    raw_error: str = "",
) -> ErrorInfo:
    return ErrorInfo(
        error_type=error_type,
        error_message=error_message,
        traceback_lines=traceback_lines or [],
        file_references=file_references or [],
        language=language,
        raw_error=raw_error or error_message,
    )


def ctx(
    *,
    python_version: str = "3.11.4",
    python_path: str = "/usr/bin/python3",
    venv_active: bool = False,
    venv_path: str | None = None,
    installed_packages: dict[str, str] | None = None,
    requirements_packages: dict[str, str] | None = None,
    os_info: str = "macOS 14.2 arm64",
    env_var_names: list[str] | None = None,
    cwd: str = "/home/user/project",
    git_branch: str | None = "main",
    git_dirty: bool | None = False,
    source_snippets: list[SourceSnippet] | None = None,
) -> LocalContext:
    return LocalContext(
        python_version=python_version,
        python_path=python_path,
        venv_active=venv_active,
        venv_path=venv_path,
        installed_packages=installed_packages or {},
        requirements_packages=requirements_packages,
        os_info=os_info,
        env_var_names=env_var_names or [],
        cwd=cwd,
        git_branch=git_branch,
        git_dirty=git_dirty,
        source_snippets=source_snippets or [],
    )


def diag(
    summary: str = "Module not installed",
    explanation: str = "Run pip install to fix it.",
    fix_command: str | None = "pip install pydantic",
    confidence: str = "high",
    source: str = "rule",
    tokens: int | None = None,
    cost_usd: float | None = None,
) -> Diagnosis:
    return Diagnosis(
        summary=summary,
        explanation=explanation,
        fix_command=fix_command,
        confidence=confidence,
        source=source,
        tokens=tokens,
        cost_usd=cost_usd,
    )


def cmd_result(
    command: str = "python app.py",
    exit_code: int = 1,
    stdout: str = "",
    stderr: str = "",
    duration: float = 0.42,
) -> CommandResult:
    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
    )


# ── _normalize_pkg ────────────────────────────────────────────────────────────────

class TestNormalizePkg:
    def test_lowercases(self):
        assert _normalize_pkg("Flask") == "flask"

    def test_replaces_underscores(self):
        assert _normalize_pkg("some_package") == "some-package"

    def test_replaces_dots(self):
        assert _normalize_pkg("zope.interface") == "zope-interface"

    def test_collapses_multiple_separators(self):
        assert _normalize_pkg("A--B__C") == "a-b-c"


# ── _find_contradictions ──────────────────────────────────────────────────────────

class TestFindContradictions:
    def test_missing_package_flagged(self):
        context = ctx(
            installed_packages={"requests": "2.31.0"},
            requirements_packages={"pydantic": ">=2.0", "requests": ""},
        )
        lines = _find_contradictions(context)
        texts = [line.plain for line in lines]
        assert any("pydantic" in t for t in texts)
        assert any("NOT installed" in t for t in texts)

    def test_installed_package_not_flagged(self):
        context = ctx(
            installed_packages={"requests": "2.31.0"},
            requirements_packages={"requests": ">=2.0"},
        )
        lines = _find_contradictions(context)
        assert not lines

    def test_version_mismatch_flagged(self):
        context = ctx(
            installed_packages={"django": "3.2.0"},
            requirements_packages={"django": ">=4.0"},
        )
        lines = _find_contradictions(context)
        texts = [line.plain for line in lines]
        assert any("django" in t for t in texts)

    def test_version_satisfied_not_flagged(self):
        context = ctx(
            installed_packages={"django": "4.2.0"},
            requirements_packages={"django": ">=4.0"},
        )
        lines = _find_contradictions(context)
        assert not lines

    def test_empty_installed_flags_all_requirements(self):
        """If nothing is installed but requirements.txt exists, everything is missing."""
        context = ctx(
            installed_packages={},
            requirements_packages={"pydantic": ">=2.0"},
        )
        lines = _find_contradictions(context)
        texts = [line.plain for line in lines]
        assert any("pydantic" in t for t in texts)

    def test_no_requirements_returns_empty(self):
        """No requirements.txt means no contradictions to detect."""
        context = ctx(installed_packages={"pydantic": "2.5.0"}, requirements_packages=None)
        lines = _find_contradictions(context)
        assert lines == []

    def test_empty_requirements_returns_empty(self):
        context = ctx(installed_packages={"pydantic": "2.5.0"})
        lines = _find_contradictions(context)
        assert lines == []

    def test_multiple_missing_packages(self):
        context = ctx(
            installed_packages={},
            requirements_packages={"pydantic": ">=2.0", "celery": ">=5.0", "redis": ""},
        )
        lines = _find_contradictions(context)
        texts = [line.plain for line in lines]
        assert any("pydantic" in t for t in texts)
        assert any("celery" in t for t in texts)
        assert any("redis" in t for t in texts)


# ── print_diagnosis ───────────────────────────────────────────────────────────────

class TestPrintDiagnosis:
    def test_renders_without_error(self):
        """Smoke test — should not raise."""
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(), console=con)
        out = captured(con, buf)
        assert len(out) > 0

    def test_contains_error_type(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(error_type="ValueError"), ctx(), diag(), console=con)
        assert "ValueError" in captured(con, buf)

    def test_contains_error_message(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(error_message="invalid value"), ctx(), diag(), console=con)
        assert "invalid value" in captured(con, buf)

    def test_contains_summary(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(summary="pydantic missing"), console=con)
        assert "pydantic missing" in captured(con, buf)

    def test_contains_explanation(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(explanation="Install the package with pip."), console=con)
        assert "Install the package with pip." in captured(con, buf)

    def test_contains_fix_command(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(fix_command="pip install pydantic"), console=con)
        assert "pip install pydantic" in captured(con, buf)

    def test_no_fix_command_absent(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(fix_command=None), console=con)
        out = captured(con, buf)
        assert "Fix:" not in out

    def test_contains_source_badge(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(source="rule"), console=con)
        assert "rule" in captured(con, buf)

    def test_contains_confidence(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(confidence="high"), console=con)
        assert "high" in captured(con, buf)

    def test_contains_duration(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(duration=1.23), ei(), ctx(), diag(), console=con)
        assert "1.23s" in captured(con, buf)

    def test_token_count_shown_when_set(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(tokens=340), console=con)
        assert "340" in captured(con, buf)

    def test_token_count_absent_when_none(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(tokens=None), console=con)
        assert "tokens" not in captured(con, buf)

    def test_cost_shown_when_set(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(cost_usd=0.0004), console=con)
        assert "0.0004" in captured(con, buf)

    def test_wtf_panel_title(self):
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), ctx(), diag(), console=con)
        assert "wtf" in captured(con, buf)

    def test_multiline_explanation_rendered(self):
        con, buf = make_console()
        explanation = "Line one.\nLine two.\nLine three."
        print_diagnosis(cmd_result(), ei(), ctx(), diag(explanation=explanation), console=con)
        out = captured(con, buf)
        assert "Line one." in out
        assert "Line two." in out
        assert "Line three." in out

    def test_show_context_false_hides_context_panel(self):
        con, buf = make_console()
        print_diagnosis(
            cmd_result(), ei(), ctx(python_version="3.11.4"),
            diag(), show_context=False, console=con,
        )
        out = captured(con, buf)
        # "Context detected" header should NOT appear
        assert "Context detected" not in out

    def test_show_context_true_shows_context_panel(self):
        con, buf = make_console()
        print_diagnosis(
            cmd_result(), ei(), ctx(python_version="3.11.4"),
            diag(), show_context=True, console=con,
        )
        out = captured(con, buf)
        assert "Context detected" in out
        assert "3.11.4" in out

    def test_show_context_includes_contradictions(self):
        """The 'aha' moment: package in requirements but not installed."""
        context = ctx(
            installed_packages={"requests": "2.31.0"},
            requirements_packages={"pydantic": ">=2.0", "requests": ""},
        )
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), context, diag(), show_context=True, console=con)
        out = captured(con, buf)
        assert "pydantic" in out
        assert "NOT installed" in out

    def test_show_context_highlights_inactive_venv(self):
        context = ctx(venv_active=False, venv_path="/project/.venv")
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), context, diag(), show_context=True, console=con)
        out = captured(con, buf)
        assert "NOT ACTIVE" in out
        assert "/project/.venv" in out

    def test_show_context_active_venv(self):
        context = ctx(venv_active=True, venv_path="/project/.venv")
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), context, diag(), show_context=True, console=con)
        out = captured(con, buf)
        assert "active" in out

    def test_show_context_renders_source_snippet(self):
        snippet = SourceSnippet(
            file_path="/project/app.py",
            line_number=5,
            context_lines=["import os", "import sys", "import pydantic", "x = 1", "y = 2"],
            error_line="import pydantic",
        )
        context = ctx(source_snippets=[snippet])
        con, buf = make_console()
        print_diagnosis(cmd_result(), ei(), context, diag(), show_context=True, console=con)
        out = captured(con, buf)
        assert "app.py" in out

    def test_ai_source_in_metadata(self):
        con, buf = make_console()
        print_diagnosis(
            cmd_result(), ei(), ctx(),
            diag(source="ai", tokens=200, cost_usd=0.0002),
            console=con,
        )
        out = captured(con, buf)
        assert "ai" in out
        assert "200" in out


# ── print_context ─────────────────────────────────────────────────────────────────

class TestPrintContext:
    def test_renders_python_version(self):
        con, buf = make_console()
        print_context(ctx(python_version="3.12.0"), ei(), console=con)
        assert "3.12.0" in captured(con, buf)

    def test_renders_os_info(self):
        con, buf = make_console()
        print_context(ctx(os_info="Ubuntu 22.04 x86_64"), ei(), console=con)
        assert "Ubuntu 22.04 x86_64" in captured(con, buf)

    def test_renders_git_branch(self):
        con, buf = make_console()
        print_context(ctx(git_branch="feature/auth"), ei(), console=con)
        assert "feature/auth" in captured(con, buf)

    def test_dirty_git_indicated(self):
        con, buf = make_console()
        print_context(ctx(git_branch="main", git_dirty=True), ei(), console=con)
        assert "dirty" in captured(con, buf)

    def test_env_vars_shown(self):
        con, buf = make_console()
        print_context(ctx(env_var_names=["SECRET_KEY", "DATABASE_URL"]), ei(), console=con)
        out = captured(con, buf)
        assert "SECRET_KEY" in out
        assert "DATABASE_URL" in out

    def test_env_vars_capped_at_8(self):
        """More than 8 env vars should show a +N more indicator."""
        many_vars = [f"VAR_{i}" for i in range(12)]
        con, buf = make_console()
        print_context(ctx(env_var_names=many_vars), ei(), console=con)
        out = captured(con, buf)
        assert "+4 more" in out

    def test_missing_package_contradiction(self):
        context = ctx(
            installed_packages={"requests": "2.31.0"},
            requirements_packages={"pydantic": ">=2.0", "requests": ""},
        )
        con, buf = make_console()
        print_context(context, ei(), console=con)
        out = captured(con, buf)
        assert "pydantic" in out
        assert "NOT installed" in out

    def test_file_reference_shown(self):
        error = ei(file_references=[FileReference("app.py", 42, "main")])
        con, buf = make_console()
        print_context(ctx(), error, console=con)
        assert "app.py" in captured(con, buf)

    def test_no_git_when_none(self):
        con, buf = make_console()
        print_context(ctx(git_branch=None), ei(), console=con)
        out = captured(con, buf)
        assert "Git" not in out

    def test_renders_without_error(self):
        """Smoke test — should not raise even with minimal context."""
        con, buf = make_console()
        print_context(ctx(), ei(), console=con)
        assert len(captured(con, buf)) > 0


# ── print_dry_run ─────────────────────────────────────────────────────────────────

class TestPrintDryRun:
    def test_renders_without_error(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(), "claude-haiku-3-5-20241022", console=con)
        assert len(captured(con, buf)) > 0

    def test_shows_model(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(), "gpt-4o-mini", console=con)
        assert "gpt-4o-mini" in captured(con, buf)

    def test_shows_error_type(self):
        con, buf = make_console()
        print_dry_run(ei(error_type="ImportError"), ctx(), "claude-haiku-3-5-20241022", console=con)
        assert "ImportError" in captured(con, buf)

    def test_shows_python_version(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(python_version="3.11.4"), "claude-haiku-3-5-20241022", console=con)
        assert "3.11.4" in captured(con, buf)

    def test_shows_package_count(self):
        context = ctx(
            installed_packages={"pydantic": "2.5.0", "requests": "2.31.0"},
            requirements_packages={"pydantic": ">=2.0"},
        )
        con, buf = make_console()
        # pydantic is in both error message (error_message="No module named 'pydantic'") and requirements
        print_dry_run(
            ei(error_message="No module named 'pydantic'"), context,
            "claude-haiku-3-5-20241022", console=con,
        )
        out = captured(con, buf)
        assert "package" in out

    def test_shows_estimated_tokens(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(), "claude-haiku-3-5-20241022", console=con)
        assert "tokens" in captured(con, buf).lower()

    def test_shows_api_panel_label(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(), "claude-haiku-3-5-20241022", console=con)
        assert "Would send to API" in captured(con, buf)

    def test_shows_os_info(self):
        con, buf = make_console()
        print_dry_run(ei(), ctx(os_info="Windows 11 AMD64"), "gpt-4o-mini", console=con)
        assert "Windows 11" in captured(con, buf)

    def test_shows_file_reference_when_present(self):
        error = ei(file_references=[FileReference("app.py", 10, None)])
        con, buf = make_console()
        print_dry_run(error, ctx(), "gpt-4o-mini", console=con)
        assert "app.py" in captured(con, buf)

    def test_shows_env_vars(self):
        context = ctx(env_var_names=["DATABASE_URL", "SECRET_KEY"])
        con, buf = make_console()
        print_dry_run(ei(), context, "gpt-4o-mini", console=con)
        out = captured(con, buf)
        assert "DATABASE_URL" in out


# ── print_no_diagnosis ────────────────────────────────────────────────────────────

class TestPrintNoDiagnosis:
    def test_renders_without_error(self):
        con, buf = make_console()
        print_no_diagnosis(ei(), console=con)
        assert len(captured(con, buf)) > 0

    def test_contains_error_type(self):
        con, buf = make_console()
        print_no_diagnosis(ei(error_type="RuntimeError"), console=con)
        assert "RuntimeError" in captured(con, buf)

    def test_contains_error_message(self):
        con, buf = make_console()
        print_no_diagnosis(ei(error_message="something went wrong"), console=con)
        assert "something went wrong" in captured(con, buf)

    def test_contains_wtf_title(self):
        con, buf = make_console()
        print_no_diagnosis(ei(), console=con)
        assert "wtf" in captured(con, buf)

    def test_hint_to_enable_ai(self):
        con, buf = make_console()
        print_no_diagnosis(ei(), console=con)
        assert "--no-ai" in captured(con, buf)

    def test_no_error_message_case(self):
        """Should not crash when error_message is empty."""
        con, buf = make_console()
        print_no_diagnosis(ei(error_message=""), console=con)
        assert len(captured(con, buf)) > 0
