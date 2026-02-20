"""
parser.py — Parse raw stderr/stdout into structured error information.

Python is the hero use case; Node.js and generic are best-effort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Standard Python traceback header
_TB_START = re.compile(r"Traceback \(most recent call last\):", re.MULTILINE)

# File reference inside a traceback: File "path", line N, in func
_TB_FILE = re.compile(r'^\s+File "(.+?)", line (\d+), in (.+)', re.MULTILINE)

# Python exception final line: "SomeError: message" or bare "SomeError"
# Covers every built-in exception plus custom ones that follow convention.
_PY_ERROR_LINE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_.]+"
    r"(?:Error|Exception|Warning|Interrupt|Exit|StopIteration|GeneratorExit))"
    r"(?::\s*(.*))?$"
)

# pytest location-style final line: "path/file.py:N: SomeError"
_PYTEST_LOCATION_ERROR = re.compile(
    r"^.+\.py:\d+: ([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception|Warning))$"
)

# pytest file-location reference: "tests/foo.py:18: AssertionError"  (no File/line syntax)
_PYTEST_FILE_LOC = re.compile(r"^(.+\.py):(\d+):\s*([A-Za-z].*)$", re.MULTILINE)

# pytest FAILED summary line: "FAILED tests/foo.py::test_bar - ErrorType: msg"
_PYTEST_FAILED = re.compile(r"^FAILED\s+(.+?)\s+-\s+(.+)$", re.MULTILINE)

# pytest short summary: "N failed, M passed in Xs"
_PYTEST_SUMMARY = re.compile(
    r"(\d+) failed(?:,\s*(\d+) passed)?(?:,\s*(\d+) error(?:s)?)?"
)

# Node.js error line (at top of output): "ErrorType: message"
_NODE_ERROR_LINE = re.compile(r"^([A-Za-z]*Error): (.+)$", re.MULTILINE)

# Node.js stack frame: "    at FuncName (file.js:line:col)"
_NODE_FRAME = re.compile(r"^\s+at .+ \(.+:\d+:\d+\)", re.MULTILINE)
_NODE_FRAME_DETAIL = re.compile(r"^\s+at (?:(.+?) )?\((.+?):(\d+):\d+\)")


# ── Dataclasses ─────────────────────────────────────────────────────────────────

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


# ── Language detection ──────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Identify whether the error is Python, Node.js, or generic."""
    if _TB_START.search(text):
        return "python"
    # SyntaxError / IndentationError can appear without a "Traceback" header
    if re.search(r"^(SyntaxError|IndentationError|TabError): ", text, re.MULTILINE):
        return "python"
    # Any recognised Python exception line after a blank/indent context
    if _PY_ERROR_LINE.search(text):
        return "python"
    # pytest output markers
    if re.search(r"^(FAILED|PASSED|ERROR)\s", text, re.MULTILINE):
        return "python"
    if _PYTEST_FAILED.search(text):
        return "python"
    # Node.js stack frames
    if _NODE_FRAME.search(text):
        return "node"
    return "generic"


# ── Error-type extraction ───────────────────────────────────────────────────────

def _find_python_error(lines: list[str]) -> tuple[str, str]:
    """
    Scan lines bottom-up for the Python exception type and message.

    Handles:
    - Standard: "ModuleNotFoundError: No module named 'foo'"
    - Bare:     "AssertionError"
    - pytest location: "tests/foo.py:5: AssertionError"
    """
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Standard exception line
        m = _PY_ERROR_LINE.match(stripped)
        if m:
            return m.group(1), (m.group(2) or "").strip()

        # pytest location line "path.py:N: ExceptionType"
        m = _PYTEST_LOCATION_ERROR.match(stripped)
        if m:
            return m.group(1), ""

    return "UnknownError", ""


# ── File reference extraction ───────────────────────────────────────────────────

def _extract_file_refs(text: str) -> list[FileReference]:
    refs: list[FileReference] = []

    # Standard Python traceback format: File "path", line N, in func
    for m in _TB_FILE.finditer(text):
        refs.append(FileReference(
            file_path=m.group(1),
            line_number=int(m.group(2)),
            function_name=m.group(3).strip() or None,
        ))

    # pytest location format: "tests/foo.py:18: AssertionError"
    # Only add if not already covered by a standard traceback ref.
    existing = {(r.file_path, r.line_number) for r in refs}
    for m in _PYTEST_FILE_LOC.finditer(text):
        path, lineno_str = m.group(1), m.group(2)
        lineno = int(lineno_str)
        if (path, lineno) not in existing:
            refs.append(FileReference(
                file_path=path,
                line_number=lineno,
                function_name=None,
            ))
            existing.add((path, lineno))

    return refs


# ── Python parser ───────────────────────────────────────────────────────────────

def _parse_python(text: str, command: str) -> ErrorInfo:
    lines = text.splitlines()

    # If there are multiple traceback blocks (chained exceptions, pytest multi-fail),
    # use the LAST one as the primary — it's the most specific failure.
    tb_starts = [i for i, l in enumerate(lines) if _TB_START.match(l)]
    if tb_starts:
        tb_lines = lines[tb_starts[-1]:]
    else:
        # SyntaxError / pytest output without a full traceback header
        tb_lines = lines

    file_refs = _extract_file_refs("\n".join(tb_lines))
    error_type, error_message = _find_python_error(tb_lines)

    # Fallback: check pytest FAILED summary for error type
    if error_type == "UnknownError":
        failed_matches = _PYTEST_FAILED.findall(text)
        if failed_matches:
            _, err_part = failed_matches[0]
            parts = err_part.split(":", 1)
            error_type = parts[0].strip()
            error_message = parts[1].strip() if len(parts) == 2 else err_part

    # Clean up traceback lines (keep non-empty)
    clean_tb = [l for l in tb_lines if l.strip()]

    return ErrorInfo(
        error_type=error_type,
        error_message=error_message,
        traceback_lines=clean_tb,
        file_references=file_refs,
        language="python",
        raw_error=text,
    )


# ── Node.js parser ──────────────────────────────────────────────────────────────

def _parse_node(text: str, command: str) -> ErrorInfo:
    lines = text.splitlines()
    error_type = "Error"
    error_message = ""

    m = _NODE_ERROR_LINE.search(text)
    if m:
        error_type = m.group(1)
        error_message = m.group(2).strip()

    file_refs: list[FileReference] = []
    for line in lines:
        m2 = _NODE_FRAME_DETAIL.match(line)
        if m2:
            func = m2.group(1)
            path = m2.group(2)
            # Skip Node internals
            if not path.startswith("node:") and not path.startswith("internal/"):
                file_refs.append(FileReference(
                    file_path=path,
                    line_number=int(m2.group(3)),
                    function_name=func or None,
                ))

    return ErrorInfo(
        error_type=error_type,
        error_message=error_message,
        traceback_lines=[l for l in lines if l.strip()],
        file_references=file_refs,
        language="node",
        raw_error=text,
    )


# ── Generic parser ──────────────────────────────────────────────────────────────

def _parse_generic(text: str, command: str) -> ErrorInfo:
    lines = [l for l in text.splitlines() if l.strip()]
    last_lines = lines[-20:] if len(lines) > 20 else lines
    error_message = last_lines[-1] if last_lines else "Unknown error"

    return ErrorInfo(
        error_type="Error",
        error_message=error_message,
        traceback_lines=last_lines,
        file_references=[],
        language="generic",
        raw_error=text,
    )


# ── Text selection ──────────────────────────────────────────────────────────────

def _pick_error_text(stderr: str, stdout: str) -> str:
    """
    Choose the most informative text from stderr and stdout.

    Python tracebacks land in stderr when Python crashes directly.
    pytest writes everything (including tracebacks) to stdout by default.

    The original strings are returned unmodified so that raw_error is faithful.
    """
    se_empty = not stderr.strip()
    so_empty = not stdout.strip()

    if se_empty and so_empty:
        return ""
    if se_empty:
        return stdout
    if so_empty:
        return stderr

    # Prefer whichever contains a traceback header
    se_has_tb = bool(_TB_START.search(stderr))
    so_has_tb = bool(_TB_START.search(stdout))

    if se_has_tb and not so_has_tb:
        return stderr
    if so_has_tb and not se_has_tb:
        return stdout

    # Both (or neither) have a traceback — combine stderr first (it's higher fidelity)
    return stderr + "\n" + stdout


# ── Public API ──────────────────────────────────────────────────────────────────

def parse_error(stderr: str, stdout: str, command: str) -> ErrorInfo:
    """
    Parse raw stderr/stdout into structured ErrorInfo.

    Args:
        stderr: Captured stderr from the failed command.
        stdout: Captured stdout from the failed command.
        command: The command string that was run (for context, not parsed).

    Returns:
        Structured ErrorInfo with error type, message, file references, etc.
    """
    text = _pick_error_text(stderr, stdout)

    if not text:
        return ErrorInfo(
            error_type="UnknownError",
            error_message="Command failed with no output",
            traceback_lines=[],
            file_references=[],
            language="generic",
            raw_error="",
        )

    lang = _detect_language(text)

    if lang == "python":
        return _parse_python(text, command)
    if lang == "node":
        return _parse_node(text, command)
    return _parse_generic(text, command)
