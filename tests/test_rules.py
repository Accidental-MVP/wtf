"""
test_rules.py — Tests for rules.py.

Each rule is tested with a matching error_info + context combination,
plus negative cases to confirm rules don't fire on wrong error types.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pytest

from wtf_cli.context import LocalContext, SourceSnippet
from wtf_cli.parser import ErrorInfo, FileReference
from wtf_cli.rules import (
    Diagnosis,
    diagnose,
    rule_connection_refused,
    rule_file_not_found,
    rule_generic_python_exception,
    rule_module_not_found,
    rule_permission_error,
    rule_pytest_failures,
    rule_syntax_error,
    rule_version_conflict,
)


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
    python_path: str = "/usr/bin/python3",
    venv_active: bool = False,
    venv_path: str | None = None,
    installed_packages: dict[str, str] | None = None,
    requirements_packages: dict[str, str] | None = None,
    cwd: str = "/home/user/project",
    source_snippets: list[SourceSnippet] | None = None,
) -> LocalContext:
    return LocalContext(
        python_version="3.11.4",
        python_path=python_path,
        venv_active=venv_active,
        venv_path=venv_path,
        installed_packages=installed_packages or {},
        requirements_packages=requirements_packages,
        cwd=cwd,
        source_snippets=source_snippets or [],
    )


def _is_diag(result: Diagnosis | None) -> Diagnosis:
    assert result is not None, "Expected a Diagnosis, got None"
    assert result.source == "rule"
    return result


# ── rule_module_not_found ────────────────────────────────────────────────────────

class TestRuleModuleNotFound:

    # ── positive: module not installed ─────────────────────────────────────────

    def test_module_not_installed_summary(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={}, requirements_packages=None),
        ))
        assert "pydantic" in d.summary

    def test_module_not_installed_fix_command(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={}),
        ))
        assert d.fix_command is not None
        assert "pydantic" in d.fix_command

    def test_module_not_installed_confidence_high(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={}),
        ))
        assert d.confidence == "high"

    # ── positive: in requirements.txt but not installed ─────────────────────────

    def test_in_requirements_but_not_installed(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(
                installed_packages={},
                requirements_packages={"pydantic": ">=2.0"},
            ),
        ))
        assert "requirements" in d.summary.lower() or "requirements" in d.explanation.lower()
        assert d.fix_command == "pip install -r requirements.txt"

    def test_in_requirements_explanation_mentions_version(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'flask'"),
            ctx(
                installed_packages={},
                requirements_packages={"flask": ">=3.0"},
            ),
        ))
        assert ">=3.0" in d.explanation or "flask" in d.explanation.lower()

    # ── positive: venv not activated ────────────────────────────────────────────

    def test_venv_not_active_highest_priority(self):
        """Venv check must fire before 'module not installed' check."""
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(
                venv_active=False,
                venv_path="/home/user/project/.venv",
                installed_packages={},
                requirements_packages={"pydantic": ">=2.0"},
            ),
        ))
        assert "venv" in d.summary.lower() or "virtualenv" in d.explanation.lower()

    def test_venv_not_active_fix_includes_activate(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(
                venv_active=False,
                venv_path="/project/.venv",
                installed_packages={},
            ),
        ))
        assert "activate" in (d.fix_command or "").lower()

    def test_venv_active_does_not_trigger_venv_rule(self):
        """With an active venv, should fall through to module-not-installed."""
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(
                venv_active=True,
                venv_path="/project/.venv",
                installed_packages={},
            ),
        ))
        assert "pip install" in (d.fix_command or "")

    # ── positive: cannot import name ─────────────────────────────────────────────

    def test_cannot_import_name(self):
        d = _is_diag(rule_module_not_found(
            ei("ImportError", "cannot import name 'verify_token' from 'auth'"),
            ctx(installed_packages={"auth": "1.0.0"}),
        ))
        assert "verify_token" in d.summary or "verify_token" in d.explanation
        assert d.confidence == "medium"

    def test_cannot_import_name_suggests_upgrade(self):
        d = _is_diag(rule_module_not_found(
            ei("ImportError", "cannot import name 'BaseSettings' from 'pydantic'"),
            ctx(installed_packages={"pydantic": "1.10.0"}),
        ))
        assert "upgrade" in (d.fix_command or "").lower() or "pydantic" in (d.fix_command or "")

    # ── submodule: extracts top-level package ─────────────────────────────────────

    def test_submodule_extracts_top_level(self):
        d = _is_diag(rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic.v1'"),
            ctx(installed_packages={}),
        ))
        assert "pydantic" in d.summary
        # Fix should target the top-level package
        assert "pydantic" in (d.fix_command or "")
        assert "pydantic.v1" not in (d.fix_command or "")

    # ── negative ──────────────────────────────────────────────────────────────────

    def test_wrong_error_type_returns_none(self):
        assert rule_module_not_found(
            ei("FileNotFoundError", "No such file or directory: 'app.py'"),
            ctx(),
        ) is None

    def test_module_already_installed_returns_none(self):
        """If the module IS installed, no rule should fire."""
        assert rule_module_not_found(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={"pydantic": "2.5.0"}),
        ) is None


# ── rule_file_not_found ───────────────────────────────────────────────────────────

class TestRuleFileNotFound:

    def test_basic_file_not_found(self):
        d = _is_diag(rule_file_not_found(
            ei("FileNotFoundError", "[Errno 2] No such file or directory: 'config.json'"),
            ctx(cwd="/home/user/project"),
        ))
        assert "config.json" in d.summary
        assert "config.json" in d.explanation

    def test_explanation_includes_cwd(self):
        d = _is_diag(rule_file_not_found(
            ei("FileNotFoundError", "[Errno 2] No such file or directory: 'app.yaml'"),
            ctx(cwd="/home/user/project"),
        ))
        assert "/home/user/project" in d.explanation

    def test_confidence_high(self):
        d = _is_diag(rule_file_not_found(
            ei("FileNotFoundError", "[Errno 2] No such file or directory: 'x.txt'"),
            ctx(),
        ))
        assert d.confidence == "high"

    def test_wrong_error_type_returns_none(self):
        assert rule_file_not_found(
            ei("PermissionError", "Permission denied: '/etc/passwd'"),
            ctx(),
        ) is None

    def test_no_path_in_message_returns_none(self):
        assert rule_file_not_found(
            ei("FileNotFoundError", ""),
            ctx(),
        ) is None


# ── rule_syntax_error ─────────────────────────────────────────────────────────────

class TestRuleSyntaxError:

    def test_basic_syntax_error(self):
        d = _is_diag(rule_syntax_error(
            ei("SyntaxError", "'(' was never closed",
               file_references=[FileReference("app.py", 12)]),
            ctx(),
        ))
        assert "SyntaxError" in d.summary
        assert "app.py" in d.summary
        assert "12" in d.summary

    def test_no_fix_command(self):
        d = _is_diag(rule_syntax_error(
            ei("SyntaxError", "invalid syntax",
               file_references=[FileReference("app.py", 5)]),
            ctx(),
        ))
        assert d.fix_command is None

    def test_indentation_error(self):
        d = _is_diag(rule_syntax_error(
            ei("IndentationError", "unexpected indent",
               file_references=[FileReference("main.py", 8)]),
            ctx(),
        ))
        assert "IndentationError" in d.summary

    def test_includes_offending_line_from_snippet(self):
        snippet = SourceSnippet(
            file_path="/project/app.py",
            line_number=12,
            context_lines=["def foo(x,", "        y"],
            error_line="def foo(x,",
        )
        d = _is_diag(rule_syntax_error(
            ei("SyntaxError", "'(' was never closed",
               file_references=[FileReference("app.py", 12)]),
            ctx(source_snippets=[snippet]),
        ))
        assert "def foo(x," in d.explanation

    def test_without_file_reference(self):
        d = _is_diag(rule_syntax_error(
            ei("SyntaxError", "invalid syntax"),
            ctx(),
        ))
        assert "SyntaxError" in d.summary

    def test_wrong_error_type_returns_none(self):
        assert rule_syntax_error(
            ei("TypeError", "unsupported operand"),
            ctx(),
        ) is None


# ── rule_permission_error ─────────────────────────────────────────────────────────

class TestRulePermissionError:

    def test_basic_permission_error(self):
        d = _is_diag(rule_permission_error(
            ei("PermissionError", "Permission denied: '/etc/passwd'"),
            ctx(cwd="/tmp"),
        ))
        assert "Permission" in d.summary or "permission" in d.explanation.lower()

    def test_wrong_error_type_returns_none(self):
        assert rule_permission_error(
            ei("FileNotFoundError", "no file"),
            ctx(),
        ) is None

    def test_no_path_returns_none(self):
        assert rule_permission_error(
            ei("PermissionError", ""),
            ctx(),
        ) is None


# ── rule_connection_refused ────────────────────────────────────────────────────────

class TestRuleConnectionRefused:

    def test_postgres_port_5432(self):
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError",
               "[Errno 111] Connection refused",
               raw_error="...connect(host='localhost', port=5432)...\nConnectionRefusedError"),
            ctx(),
        ))
        assert "PostgreSQL" in d.summary or "PostgreSQL" in d.explanation
        assert d.fix_command is not None
        assert "postgres" in d.fix_command.lower()

    def test_redis_port_6379(self):
        raw = "ConnectionRefusedError: [Errno 111] Connection refused\n...localhost:6379..."
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError",
               "[Errno 111] Connection refused",
               raw_error=raw),
            ctx(),
        ))
        assert "Redis" in d.summary or "Redis" in d.explanation

    def test_dev_port_8000(self):
        raw = "ConnectionRefusedError\nlocalhost:8000"
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError", "[Errno 111] Connection refused", raw_error=raw),
            ctx(),
        ))
        assert d is not None
        assert "8000" in d.summary or "8000" in d.explanation

    def test_unknown_port_returns_medium_confidence(self):
        raw = "ConnectionRefusedError\nlocalhost:12345"
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError", "[Errno 111] Connection refused", raw_error=raw),
            ctx(),
        ))
        assert d.confidence == "medium"

    def test_no_port_returns_generic_diagnosis(self):
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError", "Connection refused"),
            ctx(),
        ))
        assert d is not None

    def test_wrong_error_type_returns_none(self):
        assert rule_connection_refused(
            ei("FileNotFoundError", "no file"),
            ctx(),
        ) is None

    def test_high_confidence_for_known_service(self):
        raw = "ConnectionRefusedError\nlocalhost:5432"
        d = _is_diag(rule_connection_refused(
            ei("ConnectionRefusedError", "[Errno 111]", raw_error=raw),
            ctx(),
        ))
        assert d.confidence == "high"


# ── rule_pytest_failures ───────────────────────────────────────────────────────────

PYTEST_SINGLE = """\
============================= test session starts ==============================
tests/test_auth.py::test_login FAILED
================================== FAILURES ===================================
FAILED tests/test_auth.py::test_login - AssertionError: assert 302 == 200
========================= 1 failed, 4 passed in 0.34s ==========================
"""

PYTEST_MULTI = """\
============================= test session starts ==============================
FAILED tests/test_api.py::test_get_user - KeyError: 'name'
FAILED tests/test_api.py::test_post_user - AssertionError: assert None is not None
========================= 2 failed, 3 passed in 0.21s ==========================
"""


class TestRulePytestFailures:

    def test_single_failure_summary(self):
        d = _is_diag(rule_pytest_failures(
            ei("AssertionError", "assert 302 == 200", raw_error=PYTEST_SINGLE),
            ctx(),
        ))
        assert "1 test" in d.summary

    def test_single_failure_mentions_test_name(self):
        d = _is_diag(rule_pytest_failures(
            ei("AssertionError", "assert 302 == 200", raw_error=PYTEST_SINGLE),
            ctx(),
        ))
        assert "test_login" in d.explanation

    def test_multi_failure_count(self):
        d = _is_diag(rule_pytest_failures(
            ei("KeyError", "'name'", raw_error=PYTEST_MULTI),
            ctx(),
        ))
        assert "2 test" in d.summary

    def test_multi_failure_mentions_primary(self):
        d = _is_diag(rule_pytest_failures(
            ei("KeyError", "'name'", raw_error=PYTEST_MULTI),
            ctx(),
        ))
        assert "test_get_user" in d.explanation

    def test_multi_failure_mentions_other_failures(self):
        d = _is_diag(rule_pytest_failures(
            ei("KeyError", "'name'", raw_error=PYTEST_MULTI),
            ctx(),
        ))
        assert "test_post_user" in d.explanation

    def test_no_pytest_output_returns_none(self):
        assert rule_pytest_failures(
            ei("KeyError", "'name'", raw_error="KeyError: 'name'"),
            ctx(),
        ) is None

    def test_medium_confidence(self):
        d = _is_diag(rule_pytest_failures(
            ei("AssertionError", "", raw_error=PYTEST_SINGLE),
            ctx(),
        ))
        assert d.confidence == "medium"

    def test_passed_count_in_summary(self):
        d = _is_diag(rule_pytest_failures(
            ei("AssertionError", "", raw_error=PYTEST_SINGLE),
            ctx(),
        ))
        assert "out of" in d.summary or "5" in d.summary


# ── rule_version_conflict ──────────────────────────────────────────────────────────

class TestRuleVersionConflict:

    def test_detects_version_conflict(self):
        raw = (
            "ImportError: requires pydantic >=2.0 but you have 1.10.0\n"
            "pydantic requires >=2.0"
        )
        d = _is_diag(rule_version_conflict(
            ei("ImportError", "requires pydantic >=2.0", raw_error=raw),
            ctx(installed_packages={"pydantic": "1.10.0"}),
        ))
        assert "pydantic" in d.summary.lower()
        assert "1.10.0" in d.summary or "1.10.0" in d.explanation

    def test_fix_command_upgrades_package(self):
        raw = "requires pydantic >=2.0"
        d = _is_diag(rule_version_conflict(
            ei("ImportError", "version conflict", raw_error=raw),
            ctx(installed_packages={"pydantic": "1.9.0"}),
        ))
        assert "pydantic" in (d.fix_command or "")

    def test_no_version_keywords_returns_none(self):
        assert rule_version_conflict(
            ei("ImportError", "cannot import name 'X' from 'y'",
               raw_error="cannot import name 'X' from 'y'"),
            ctx(installed_packages={"y": "1.0"}),
        ) is None

    def test_package_not_installed_returns_none(self):
        raw = "requires pydantic >=2.0"
        assert rule_version_conflict(
            ei("ImportError", "version conflict", raw_error=raw),
            ctx(installed_packages={}),  # pydantic not installed
        ) is None

    def test_medium_confidence(self):
        raw = "requires pydantic >=2.0"
        d = _is_diag(rule_version_conflict(
            ei("ImportError", "version conflict", raw_error=raw),
            ctx(installed_packages={"pydantic": "1.9.0"}),
        ))
        assert d.confidence == "medium"


# ── rule_generic_python_exception ──────────────────────────────────────────────────

class TestRuleGenericPythonException:

    def test_attribute_error_explanation(self):
        d = _is_diag(rule_generic_python_exception(
            ei("AttributeError", "'list' object has no attribute 'items'"),
            ctx(),
        ))
        assert "AttributeError" in d.summary
        assert "attribute" in d.explanation.lower()

    def test_key_error_explanation(self):
        d = _is_diag(rule_generic_python_exception(
            ei("KeyError", "'user_id'"),
            ctx(),
        ))
        assert "dict" in d.explanation.lower() or "key" in d.explanation.lower()

    def test_zero_division_explanation(self):
        d = _is_diag(rule_generic_python_exception(
            ei("ZeroDivisionError", "division by zero"),
            ctx(),
        ))
        assert "zero" in d.explanation.lower() or "division" in d.explanation.lower()

    def test_medium_confidence(self):
        d = _is_diag(rule_generic_python_exception(
            ei("KeyError", "'x'"),
            ctx(),
        ))
        assert d.confidence == "medium"

    def test_no_fix_command(self):
        d = _is_diag(rule_generic_python_exception(
            ei("TypeError", "unsupported operand"),
            ctx(),
        ))
        assert d.fix_command is None

    def test_unknown_exception_returns_none(self):
        assert rule_generic_python_exception(
            ei("WeirdCustomException", "something broke"),
            ctx(),
        ) is None

    def test_non_python_returns_none(self):
        assert rule_generic_python_exception(
            ei("Error", "generic error", language="node"),
            ctx(),
        ) is None

    def test_error_message_included_in_explanation(self):
        d = _is_diag(rule_generic_python_exception(
            ei("KeyError", "'session_token'"),
            ctx(),
        ))
        assert "session_token" in d.explanation


# ── diagnose() pipeline ────────────────────────────────────────────────────────────

class TestDiagnosePipeline:

    def test_returns_none_when_nothing_matches(self):
        assert diagnose(
            ei("WeirdError", "something unknown", language="generic"),
            ctx(),
        ) is None

    def test_returns_diagnosis_for_module_not_found(self):
        result = diagnose(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={}),
        )
        assert result is not None
        assert result.source == "rule"

    def test_module_not_found_takes_priority_over_generic(self):
        """ModuleNotFoundError should be caught by rule_module_not_found, not the fallback."""
        result = diagnose(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(installed_packages={}),
        )
        assert result is not None
        assert "pydantic" in result.fix_command or "pydantic" in result.summary

    def test_venv_rule_takes_priority_over_install_rule(self):
        result = diagnose(
            ei("ModuleNotFoundError", "No module named 'pydantic'"),
            ctx(
                venv_active=False,
                venv_path="/project/.venv",
                installed_packages={},
                requirements_packages={"pydantic": ">=2.0"},
            ),
        )
        assert result is not None
        assert "venv" in result.summary.lower() or "virtualenv" in result.explanation.lower()

    def test_rule_exception_does_not_propagate(self):
        """A crashing rule must not prevent other rules from running."""
        from unittest.mock import patch
        from wtf_cli import rules as rules_mod

        boom_rule = lambda ei, ctx: (_ for _ in ()).throw(RuntimeError("kaboom"))

        with patch.object(rules_mod, "_RULES", [boom_rule, rule_generic_python_exception]):
            result = diagnose(
                ei("KeyError", "'x'"),
                ctx(),
            )
        assert result is not None  # generic rule should still fire

    def test_source_is_always_rule(self):
        result = diagnose(
            ei("SyntaxError", "invalid syntax",
               file_references=[FileReference("x.py", 1)]),
            ctx(),
        )
        assert result is not None
        assert result.source == "rule"
