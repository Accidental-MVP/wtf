"""
test_runner.py — Tests for runner.py command execution.
"""

from __future__ import annotations

import sys

import pytest

from wtf_cli.runner import CommandResult, run_command


class TestCommandResult:
    def test_dataclass_fields(self):
        result = CommandResult(
            command="echo hi",
            exit_code=0,
            stdout="hi\n",
            stderr="",
            duration=0.01,
        )
        assert result.command == "echo hi"
        assert result.exit_code == 0
        assert result.stdout == "hi\n"
        assert result.stderr == ""
        assert result.duration == pytest.approx(0.01)


class TestRunCommand:
    def test_successful_command(self):
        """A zero-exit command should return exit_code=0 and capture stdout."""
        if sys.platform == "win32":
            cmd = ["echo", "hello"]
        else:
            cmd = ["echo", "hello"]

        result = run_command(cmd)
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.duration > 0
        assert result.command == " ".join(cmd)

    def test_failing_command(self):
        """A non-zero-exit command should return its exit code."""
        if sys.platform == "win32":
            cmd = ["cmd", "/c", "exit 1"]
        else:
            cmd = ["bash", "-c", "exit 1"]

        result = run_command(cmd)
        assert result.exit_code == 1

    def test_stderr_captured(self):
        """stderr output should be captured separately from stdout."""
        if sys.platform == "win32":
            cmd = ["cmd", "/c", "echo err 1>&2 && exit 1"]
        else:
            cmd = ["bash", "-c", "echo err >&2; exit 1"]

        result = run_command(cmd)
        assert result.exit_code == 1
        assert "err" in result.stderr

    def test_duration_is_measured(self):
        """Duration should be greater than zero for any real command."""
        result = run_command(["python", "--version"])
        assert result.duration > 0

    def test_python_error_output_captured(self):
        """Python errors go to stderr; run_command should capture them."""
        cmd = ["python", "-c", "import nonexistent_module_xyz"]
        result = run_command(cmd)
        assert result.exit_code != 0
        assert "nonexistent_module_xyz" in result.stderr

    def test_multiline_stdout(self):
        """Multiline output should be fully captured."""
        if sys.platform == "win32":
            cmd = ["python", "-c", "for i in range(5): print(i)"]
        else:
            cmd = ["python", "-c", "for i in range(5): print(i)"]

        result = run_command(cmd)
        assert result.exit_code == 0
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) == 5
        assert "0" in result.stdout
        assert "4" in result.stdout

    def test_command_str_is_joined(self):
        """The command field should be the space-joined tokens."""
        cmd = ["python", "--version"]
        result = run_command(cmd)
        assert result.command == "python --version"
