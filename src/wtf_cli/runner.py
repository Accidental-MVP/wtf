"""
runner.py — Runs the user's command and captures stdout, stderr, and exit code.

Key design: real-time streaming to the terminal AND buffered capture.
The user sees normal command output as it happens; diagnosis appears after.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float


def run_command(command: list[str], timeout: int = 300) -> CommandResult:
    """
    Run a shell command, stream output in real-time, and capture the result.

    Uses subprocess.Popen with two reader threads so stdout is printed line-by-line
    as it arrives while also being buffered. stderr is captured separately and
    printed after the process exits (it typically arrives all at once at the end).

    Args:
        command: List of command tokens, e.g. ["pytest", "tests/"]
        timeout: Seconds before forcibly killing the process (default 300 / 5 min)

    Returns:
        CommandResult with captured stdout, stderr, exit code, and wall-clock duration.
    """
    # Build a properly-quoted shell string so arguments with spaces/special
    # characters survive the round-trip through the platform shell.
    if sys.platform == "win32":
        command_str = subprocess.list2cmdline(command)
    else:
        command_str = shlex.join(command)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    start = time.monotonic()

    process = subprocess.Popen(
        command_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    def _stream_stdout() -> None:
        """Read stdout line by line, printing each line and buffering it."""
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stdout_lines.append(line)
        except ValueError:
            # File closed mid-read (process killed); that's fine.
            pass

    def _stream_stderr() -> None:
        """Read stderr line by line, printing each line and buffering it."""
        assert process.stderr is not None
        try:
            for line in process.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)
        except ValueError:
            pass

    t_out = threading.Thread(target=_stream_stdout, daemon=True)
    t_err = threading.Thread(target=_stream_stderr, daemon=True)
    t_out.start()
    t_err.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        sys.stderr.write(f"\n[wtf] Command timed out after {timeout}s\n")
        sys.stderr.flush()
    finally:
        # Ensure reader threads finish draining before we measure duration.
        t_out.join(timeout=5)
        t_err.join(timeout=5)

    duration = time.monotonic() - start

    return CommandResult(
        command=command_str,
        exit_code=process.returncode if process.returncode is not None else 1,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        duration=duration,
    )
