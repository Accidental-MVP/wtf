"""
test_ai.py — Tests for ai.py (Tier 2 LLM diagnosis).

All litellm calls are mocked so no real API traffic is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wtf_cli.ai import (
    _parse_response,
    _resolve_model,
    _select_relevant_packages,
    build_prompt,
    diagnose,
)
from wtf_cli.context import LocalContext, SourceSnippet
from wtf_cli.parser import ErrorInfo, FileReference
from wtf_cli.rules import Diagnosis


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


def _mock_litellm_response(
    content: str,
    total_tokens: int = 200,
    model: str = "claude-haiku-3-5-20241022",
) -> MagicMock:
    """Build a minimal litellm-response-shaped mock."""
    usage = SimpleNamespace(total_tokens=total_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


# ── _parse_response ───────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_well_formed(self):
        text = (
            "SUMMARY: pydantic is not installed\n"
            "EXPLANATION: The package pydantic is missing from the active environment.\n"
            "FIX: pip install pydantic"
        )
        summary, explanation, fix = _parse_response(text)
        assert summary == "pydantic is not installed"
        assert "pydantic" in explanation
        assert fix == "pip install pydantic"

    def test_multiline_explanation(self):
        text = (
            "SUMMARY: Something is wrong\n"
            "EXPLANATION: Line one.\n"
            "Line two continues here.\n"
            "FIX: npm install"
        )
        summary, explanation, fix = _parse_response(text)
        assert summary == "Something is wrong"
        assert "Line one" in explanation
        assert "Line two" in explanation
        assert fix == "npm install"

    def test_no_single_command_fix(self):
        text = (
            "SUMMARY: Complex error\n"
            "EXPLANATION: Needs manual investigation.\n"
            "FIX: no single command fix"
        )
        _, _, fix = _parse_response(text)
        assert fix is None

    def test_fix_none_variants(self):
        for phrase in ("none", "n/a", "no fix", "no command"):
            text = f"SUMMARY: x\nEXPLANATION: y\nFIX: {phrase}"
            _, _, fix = _parse_response(text)
            assert fix is None, f"Expected None for FIX: {phrase}"

    def test_fallback_no_format(self):
        text = "Something went completely wrong in the pipeline."
        summary, explanation, fix = _parse_response(text)
        assert summary  # non-empty
        assert explanation == text.strip()
        assert fix is None

    def test_case_insensitive_keys(self):
        text = "summary: lowercase keys\nexplanation: works fine\nfix: make test"
        summary, explanation, fix = _parse_response(text)
        assert summary == "lowercase keys"
        assert fix == "make test"

    def test_empty_string(self):
        summary, explanation, fix = _parse_response("")
        assert summary == "AI diagnosis"
        assert fix is None

    def test_summary_only(self):
        text = "SUMMARY: Just a summary here"
        summary, explanation, fix = _parse_response(text)
        assert summary == "Just a summary here"
        # explanation falls back to summary
        assert explanation == summary


# ── _select_relevant_packages ─────────────────────────────────────────────────────

class TestSelectRelevantPackages:
    def test_picks_package_mentioned_in_error(self):
        error = ei(error_message="No module named 'pydantic'")
        context = ctx(installed_packages={"pydantic": "2.5.0", "requests": "2.31.0"})
        result = _select_relevant_packages(error, context)
        assert "pydantic" in result
        assert "requests" not in result

    def test_picks_package_from_traceback(self):
        error = ei(
            error_message="validation error",
            traceback_lines=["File app.py", "import pydantic"],
        )
        context = ctx(installed_packages={"pydantic": "2.5.0", "flask": "3.0.0"})
        result = _select_relevant_packages(error, context)
        assert "pydantic" in result

    def test_fills_from_requirements_if_space(self):
        error = ei(error_message="generic error with no packages mentioned")
        context = ctx(
            installed_packages={"django": "4.2", "celery": "5.3"},
            requirements_packages={"django": ">=4.0", "celery": ""},
        )
        result = _select_relevant_packages(error, context)
        # Should pull django and celery from requirements
        assert "django" in result or "celery" in result

    def test_max_packages_limit(self):
        installed = {f"pkg{i}": f"1.{i}" for i in range(50)}
        error = ei(
            error_message=" ".join(installed.keys()),  # all mentioned
        )
        context = ctx(installed_packages=installed)
        result = _select_relevant_packages(error, context, max_packages=20)
        assert len(result) <= 20

    def test_empty_installed(self):
        result = _select_relevant_packages(ei(), ctx())
        assert result == {}

    def test_picks_from_snippet_imports(self):
        snippet = SourceSnippet(
            file_path="app.py",
            line_number=5,
            context_lines=["import flask", "from sqlalchemy import Column"],
            error_line="from sqlalchemy import Column",
        )
        error = ei(error_message="some error")
        context = ctx(
            installed_packages={"flask": "3.0.0", "sqlalchemy": "2.0.0", "boto3": "1.34.0"},
            source_snippets=[snippet],
        )
        result = _select_relevant_packages(error, context)
        assert "flask" in result
        assert "sqlalchemy" in result
        assert "boto3" not in result


# ── build_prompt ──────────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_error_info(self):
        error = ei(
            error_type="ImportError",
            error_message="cannot import name 'v2' from 'pydantic'",
            traceback_lines=["Traceback...", "File app.py"],
        )
        prompt = build_prompt(error, ctx(), command="python app.py")
        assert "ImportError" in prompt
        assert "pydantic" in prompt
        assert "python app.py" in prompt

    def test_contains_python_context(self):
        context = ctx(python_version="3.11.4", python_path="/usr/bin/python3")
        prompt = build_prompt(ei(), context)
        assert "3.11.4" in prompt
        assert "/usr/bin/python3" in prompt

    def test_venv_not_active_shown(self):
        context = ctx(venv_active=False, venv_path="/project/.venv")
        prompt = build_prompt(ei(), context)
        assert "NOT ACTIVE" in prompt
        assert "/project/.venv" in prompt

    def test_venv_active_shown(self):
        context = ctx(venv_active=True, venv_path="/project/.venv")
        prompt = build_prompt(ei(), context)
        assert "active" in prompt
        assert "NOT ACTIVE" not in prompt

    def test_traceback_capped_at_30_lines(self):
        long_tb = [f"frame {i}" for i in range(50)]
        error = ei(traceback_lines=long_tb)
        prompt = build_prompt(error, ctx())
        # Only first 30 should appear
        assert "frame 29" in prompt
        assert "frame 30" not in prompt

    def test_git_info_included(self):
        context = ctx(git_branch="feature/auth", git_dirty=True)
        prompt = build_prompt(ei(), context)
        assert "feature/auth" in prompt
        assert "dirty" in prompt

    def test_no_git_when_none(self):
        context = ctx(git_branch=None)
        prompt = build_prompt(ei(), context)
        assert "not a git repo" in prompt

    def test_env_var_names_included(self):
        context = ctx(env_var_names=["DATABASE_URL", "SECRET_KEY"])
        prompt = build_prompt(ei(), context)
        assert "DATABASE_URL" in prompt
        assert "SECRET_KEY" in prompt

    def test_source_snippets_included(self):
        snippet = SourceSnippet(
            file_path="/project/app.py",
            line_number=10,
            context_lines=["line8", "line9", "line10_error", "line11"],
            error_line="line10_error",
        )
        context = ctx(source_snippets=[snippet])
        prompt = build_prompt(ei(), context)
        assert "/project/app.py" in prompt
        assert "line10_error" in prompt

    def test_no_command_when_empty(self):
        prompt = build_prompt(ei(), ctx(), command="")
        assert "Command:" not in prompt

    def test_requirements_shown(self):
        context = ctx(requirements_packages={"django": ">=4.0", "celery": ""})
        prompt = build_prompt(ei(), context)
        assert "django" in prompt


# ── _resolve_model ────────────────────────────────────────────────────────────────

class TestResolveModel:
    def test_local_flag_returns_ollama(self):
        model = _resolve_model(None, local=True, api_key=None)
        assert model.startswith("ollama/")

    def test_local_flag_with_custom_model(self):
        model = _resolve_model("ollama/mistral", local=True, api_key=None)
        assert model == "ollama/mistral"

    def test_explicit_model_wins(self):
        model = _resolve_model("gpt-4o", local=False, api_key=None)
        assert model == "gpt-4o"

    def test_anthropic_key_selects_haiku(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": ""}):
            os.environ.pop("OPENAI_API_KEY", None)
            model = _resolve_model(None, local=False, api_key=None)
        assert "haiku" in model or "claude" in model

    def test_openai_key_selects_gpt4o_mini(self):
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
        with patch.dict(os.environ, {**clean_env, "OPENAI_API_KEY": "sk-test"}, clear=True):
            model = _resolve_model(None, local=False, api_key=None)
        assert "gpt" in model

    def test_api_key_prefix_sk_ant(self):
        with patch.dict(os.environ, {}, clear=True):
            model = _resolve_model(None, local=False, api_key="sk-ant-abc")
        assert "claude" in model or "haiku" in model


# ── diagnose — dry_run ────────────────────────────────────────────────────────────

class TestDiagnoseDryRun:
    def test_dry_run_returns_without_api_call(self):
        error = ei()
        context = ctx()
        diagnosis = diagnose(error, context, dry_run=True, api_key="fake-key")
        assert diagnosis.source == "dry-run"
        assert "DRY RUN" in diagnosis.explanation
        assert diagnosis.fix_command is None

    def test_dry_run_contains_model_name(self):
        diagnosis = diagnose(
            ei(), ctx(),
            model="gpt-4o",
            dry_run=True,
            api_key="fake-key",
        )
        assert "gpt-4o" in diagnosis.explanation

    def test_dry_run_contains_prompt_sections(self):
        diagnosis = diagnose(
            ei(error_type="ValueError", error_message="invalid literal"),
            ctx(python_version="3.11.4"),
            dry_run=True,
            api_key="fake-key",
        )
        assert "ValueError" in diagnosis.explanation
        assert "3.11.4" in diagnosis.explanation
        assert "System prompt" in diagnosis.explanation


# ── diagnose — missing API key ────────────────────────────────────────────────────

class TestDiagnoseNoApiKey:
    def test_returns_helpful_message(self):
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}
        with patch.dict(os.environ, clean_env, clear=True):
            diagnosis = diagnose(ei(), ctx())
        assert diagnosis.source == "ai"
        assert "No API key" in diagnosis.summary or "No API key" in diagnosis.explanation
        assert "ANTHROPIC_API_KEY" in diagnosis.explanation or "OPENAI_API_KEY" in diagnosis.explanation

    def test_local_flag_bypasses_key_check(self):
        """With --local, no API key check should happen — the call goes to Ollama."""
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}
        mock_response = _mock_litellm_response(
            "SUMMARY: local model\nEXPLANATION: ok\nFIX: no single command fix"
        )
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("wtf_cli.ai.litellm") as mock_litellm:
                mock_litellm.completion.return_value = mock_response
                mock_litellm.completion_cost.return_value = 0.0
                mock_litellm.suppress_debug_info = False
                diagnosis = diagnose(ei(), ctx(), local=True)
        assert diagnosis.source == "ai"
        assert diagnosis.summary == "local model"


# ── diagnose — successful API call ───────────────────────────────────────────────

class TestDiagnoseSuccess:
    def _run(self, content: str, **kwargs) -> Diagnosis:
        mock_response = _mock_litellm_response(content)
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            mock_litellm.completion_cost.return_value = 0.0004
            mock_litellm.suppress_debug_info = False
            return diagnose(ei(), ctx(), api_key="sk-ant-test", **kwargs)

    def test_basic_structured_response(self):
        content = (
            "SUMMARY: pydantic not installed\n"
            "EXPLANATION: Package is missing.\n"
            "FIX: pip install pydantic"
        )
        diagnosis = self._run(content)
        assert diagnosis.summary == "pydantic not installed"
        assert diagnosis.fix_command == "pip install pydantic"
        assert diagnosis.source == "ai"
        assert diagnosis.confidence == "medium"

    def test_token_count_captured(self):
        content = "SUMMARY: test\nEXPLANATION: ok\nFIX: no single command fix"
        diagnosis = self._run(content)
        assert diagnosis.tokens == 200

    def test_cost_captured(self):
        content = "SUMMARY: test\nEXPLANATION: ok\nFIX: no single command fix"
        diagnosis = self._run(content)
        assert diagnosis.cost_usd == pytest.approx(0.0004)

    def test_no_fix_sets_none(self):
        content = "SUMMARY: complex issue\nEXPLANATION: needs investigation\nFIX: no single command fix"
        diagnosis = self._run(content)
        assert diagnosis.fix_command is None

    def test_litellm_called_with_correct_messages(self):
        mock_response = _mock_litellm_response(
            "SUMMARY: x\nEXPLANATION: y\nFIX: no fix"
        )
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            mock_litellm.completion_cost.return_value = 0.0
            mock_litellm.suppress_debug_info = False
            diagnose(ei(), ctx(), api_key="sk-ant-test")

        call_kwargs = mock_litellm.completion.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "senior developer" in messages[0]["content"]

    def test_explicit_model_forwarded(self):
        mock_response = _mock_litellm_response(
            "SUMMARY: x\nEXPLANATION: y\nFIX: no fix"
        )
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            mock_litellm.completion_cost.return_value = 0.0
            mock_litellm.suppress_debug_info = False
            diagnose(ei(), ctx(), model="gpt-4o", api_key="sk-test")

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_api_key_forwarded(self):
        mock_response = _mock_litellm_response(
            "SUMMARY: x\nEXPLANATION: y\nFIX: no fix"
        )
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            mock_litellm.completion_cost.return_value = 0.0
            mock_litellm.suppress_debug_info = False
            diagnose(ei(), ctx(), api_key="sk-ant-explicit")

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs.get("api_key") == "sk-ant-explicit"


# ── diagnose — API error handling ────────────────────────────────────────────────

class TestDiagnoseErrorHandling:
    def _diagnose_with_exc(self, exc: Exception) -> Diagnosis:
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = exc
            mock_litellm.suppress_debug_info = False
            return diagnose(ei(), ctx(), api_key="sk-ant-test")

    def test_auth_error_handled(self):
        class AuthenticationError(Exception):
            pass
        diagnosis = self._diagnose_with_exc(AuthenticationError("invalid key"))
        assert "authentication" in diagnosis.summary.lower()
        assert diagnosis.source == "ai"

    def test_rate_limit_handled(self):
        class RateLimitError(Exception):
            pass
        diagnosis = self._diagnose_with_exc(RateLimitError("too many requests"))
        assert "rate limit" in diagnosis.summary.lower()

    def test_generic_exception_handled(self):
        diagnosis = self._diagnose_with_exc(ConnectionError("network unreachable"))
        assert "unavailable" in diagnosis.summary.lower() or "ConnectionError" in diagnosis.summary
        assert diagnosis.source == "ai"

    def test_never_raises(self):
        """diagnose() must never propagate exceptions to the caller."""
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = RuntimeError("catastrophic failure")
            mock_litellm.suppress_debug_info = False
            result = diagnose(ei(), ctx(), api_key="sk-ant-test")
        assert isinstance(result, Diagnosis)

    def test_litellm_import_error(self):
        """If litellm isn't installed (module-level None), returns a helpful message."""
        with patch("wtf_cli.ai.litellm", None):
            diagnosis = diagnose(ei(), ctx(), api_key="sk-ant-test")

        assert "litellm" in diagnosis.explanation
        assert diagnosis.fix_command == "pip install litellm"

    def test_cost_calculation_failure_is_silent(self):
        """If completion_cost() raises, tokens are still captured."""
        mock_response = _mock_litellm_response(
            "SUMMARY: x\nEXPLANATION: y\nFIX: no fix",
            total_tokens=150,
        )
        with patch("wtf_cli.ai.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            mock_litellm.completion_cost.side_effect = Exception("cost calc failed")
            mock_litellm.suppress_debug_info = False
            diagnosis = diagnose(ei(), ctx(), api_key="sk-ant-test")

        assert diagnosis.tokens == 150
        assert diagnosis.cost_usd is None


# ── Integration: build_prompt round-trip ─────────────────────────────────────────

class TestBuildPromptIntegration:
    def test_realistic_module_not_found(self):
        error = ei(
            error_type="ModuleNotFoundError",
            error_message="No module named 'pydantic'",
            traceback_lines=[
                "Traceback (most recent call last):",
                '  File "app.py", line 3, in <module>',
                "    import pydantic",
                "ModuleNotFoundError: No module named 'pydantic'",
            ],
        )
        context = ctx(
            python_version="3.11.4",
            venv_active=False,
            venv_path="/project/.venv",
            installed_packages={"requests": "2.31.0"},
            requirements_packages={"pydantic": ">=2.0", "requests": ""},
            git_branch="main",
            git_dirty=False,
        )
        prompt = build_prompt(error, context, command="python app.py")
        # All key facts present
        assert "ModuleNotFoundError" in prompt
        assert "pydantic" in prompt
        assert "3.11.4" in prompt
        assert "NOT ACTIVE" in prompt
        assert "/project/.venv" in prompt
        assert "main" in prompt

    def test_prompt_does_not_include_env_values(self):
        """Privacy: env var VALUES must never appear in the prompt."""
        context = ctx(env_var_names=["SECRET_KEY", "DATABASE_URL"])
        prompt = build_prompt(ei(), context)
        # Names appear, but no "=" signs suggesting values
        assert "SECRET_KEY" in prompt
        assert "DATABASE_URL" in prompt
        # The prompt should not have "SECRET_KEY=somevalue" style content
        assert "SECRET_KEY=" not in prompt
