"""
rules.py — Tier 1 rule-based diagnosis (no LLM needed).

Rules are tried in priority order; the first match wins.
Each rule function returns a Diagnosis or None. Rules must never raise —
any exception is caught by the pipeline so one bad rule never blocks others.

This is what makes the tool useful with ZERO config on first install.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from wtf_cli.context import LocalContext
from wtf_cli.parser import ErrorInfo


# ── Diagnosis dataclass ──────────────────────────────────────────────────────────

@dataclass
class Diagnosis:
    summary: str
    explanation: str
    fix_command: str | None
    confidence: str        # "high" | "medium"
    source: str            # "rule" | "ai" | "dry-run"
    tokens: int | None = None
    cost_usd: float | None = None


# ── Internal helpers ─────────────────────────────────────────────────────────────

def _diag(
    summary: str,
    explanation: str,
    fix_command: str | None = None,
    confidence: str = "high",
) -> Diagnosis:
    return Diagnosis(
        summary=summary,
        explanation=explanation,
        fix_command=fix_command,
        confidence=confidence,
        source="rule",
    )


def _normalize(name: str) -> str:
    """Lowercase and canonicalise package name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_missing_module(error_info: ErrorInfo) -> str | None:
    """
    Pull the missing module name from ModuleNotFoundError / ImportError messages.

    'No module named 'pydantic''          → 'pydantic'
    'No module named 'pydantic.v1''        → 'pydantic'  (top-level only)
    'cannot import name 'X' from 'auth''   → None (handled separately)
    """
    m = re.search(r"No module named '([^']+)'", error_info.error_message)
    if m:
        return m.group(1).split(".")[0]
    return None


def _extract_cannot_import(error_info: ErrorInfo) -> tuple[str, str] | None:
    """Return (name, from_module) for 'cannot import name X from Y' errors."""
    m = re.search(
        r"cannot import name '([^']+)' from '([^']+)'",
        error_info.error_message,
    )
    return (m.group(1), m.group(2)) if m else None


def _extract_file_path(error_info: ErrorInfo) -> str | None:
    """Extract the file path from FileNotFoundError / PermissionError messages."""
    # '[Errno 2] No such file or directory: 'config.json''
    m = re.search(r": '([^']+)'$", error_info.error_message)
    if m:
        return m.group(1)
    m = re.search(r': "([^"]+)"$', error_info.error_message)
    if m:
        return m.group(1)
    # Bare path without quotes
    m = re.search(r"(?:No such file or directory|Permission denied)[:\s]+(.+)$",
                  error_info.error_message)
    if m:
        return m.group(1).strip().strip("'\"")
    return None


def _find_case_variant(path_str: str, cwd: str) -> str | None:
    """
    On case-sensitive filesystems, look for a file whose name matches
    case-insensitively. Returns the found path string or None.
    """
    p = Path(path_str) if Path(path_str).is_absolute() else Path(cwd) / path_str
    target = p.name.lower()
    try:
        for entry in p.parent.iterdir():
            if entry.name.lower() == target and entry.name != p.name:
                return str(entry)
    except OSError:
        pass
    return None


def _activate_cmd(venv_path: str) -> str:
    """Platform-appropriate virtualenv activation command."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "activate")
    return f"source {venv_path}/bin/activate"


# ── Port → service mapping ────────────────────────────────────────────────────────

_KNOWN_PORTS: dict[int, tuple[str, str]] = {
    5432:  ("PostgreSQL",      "docker compose up -d postgres"),
    3306:  ("MySQL",           "docker compose up -d mysql"),
    27017: ("MongoDB",         "docker compose up -d mongo"),
    6379:  ("Redis",           "docker compose up -d redis"),
    5672:  ("RabbitMQ",        "docker compose up -d rabbitmq"),
    9200:  ("Elasticsearch",   "docker compose up -d elasticsearch"),
    2181:  ("ZooKeeper",       "docker compose up -d zookeeper"),
}
_DEV_PORTS: frozenset[int] = frozenset({3000, 4000, 5000, 8000, 8080, 8888, 9000})


# ── Well-known exception explanations (fallback rule) ────────────────────────────

_EXCEPTION_EXPLANATIONS: dict[str, str] = {
    "AttributeError": (
        "You tried to access an attribute or method that doesn't exist on this object. "
        "Check the object's type and its available attributes with dir()."
    ),
    "TypeError": (
        "An operation was applied to the wrong type. "
        "Check that you're passing arguments of the right type to the function."
    ),
    "ValueError": (
        "A function received the right type but an invalid value. "
        "Verify the value you're passing is within the expected range or format."
    ),
    "KeyError": (
        "You tried to access a dictionary key that doesn't exist. "
        "Use .get(key) to access safely, or check with `key in d` first."
    ),
    "IndexError": (
        "You tried to access a list index that's out of range. "
        "Check that the index is less than len(sequence)."
    ),
    "NameError": (
        "A variable or function name was referenced before it was defined. "
        "Check for typos or a missing import."
    ),
    "RuntimeError": (
        "A general runtime error occurred. "
        "The full traceback above should point to the specific cause."
    ),
    "RecursionError": (
        "The maximum recursion depth was exceeded. "
        "Your recursive function is missing a base case or the input triggers too many levels."
    ),
    "NotImplementedError": (
        "An abstract method or feature stub hasn't been implemented yet. "
        "Check if you need to subclass and override this method."
    ),
    "AssertionError": (
        "An assert statement failed — the condition evaluated to False. "
        "Check the value being asserted."
    ),
    "ZeroDivisionError": (
        "Division or modulo by zero. "
        "Guard the denominator: `if denominator != 0: ...`."
    ),
    "UnicodeDecodeError": (
        "The bytes couldn't be decoded with the specified encoding. "
        "Try opening the file with encoding='utf-8' or encoding='latin-1'."
    ),
    "UnicodeEncodeError": (
        "The string couldn't be encoded with the specified encoding. "
        "Use errors='ignore' or errors='replace' to skip problem characters."
    ),
    "OverflowError": (
        "A numeric result was too large to represent. "
        "Consider using Python's arbitrary-precision integers instead of floats."
    ),
    "MemoryError": (
        "The program ran out of memory. "
        "Consider streaming or processing data in smaller batches."
    ),
    "TimeoutError": (
        "An operation exceeded its allowed time limit. "
        "Increase the timeout or investigate why the operation is slow."
    ),
    "StopIteration": (
        "StopIteration was raised outside a generator or for-loop. "
        "When using next() directly, always provide a default: next(it, None)."
    ),
    "OSError": (
        "A system-level error with file I/O or OS calls. "
        "Check file paths, permissions, and disk space."
    ),
}


# ── Individual rules ──────────────────────────────────────────────────────────────

def rule_module_not_found(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """
    Rules 1 + 8: ModuleNotFoundError / ImportError.

    Priority within this rule:
      1. Venv exists but not activated  → activate it
      2. Module in requirements.txt but not installed  → pip install -r
      3. Module not installed at all    → pip install <module>
      4. ImportError: cannot import name X from Y (installed but wrong version)
    """
    if error_info.error_type not in ("ModuleNotFoundError", "ImportError"):
        return None

    # ── Handle 'cannot import name X from Y' separately ──────────────────────────
    cannot = _extract_cannot_import(error_info)
    if cannot and error_info.error_type == "ImportError":
        name, from_module = cannot
        norm = _normalize(from_module)
        installed_ver = context.installed_packages.get(norm, "unknown version")
        return _diag(
            summary=f"Cannot import '{name}' from '{from_module}'",
            explanation=(
                f"'{from_module}' ({installed_ver}) is installed but does not export '{name}'. "
                f"This is usually a version mismatch — '{name}' may have been added, renamed, "
                f"or removed in a different release. Check the {from_module} changelog."
            ),
            fix_command=f"pip install --upgrade {from_module}",
            confidence="medium",
        )

    module = _extract_missing_module(error_info)
    if not module:
        return None

    norm_module = _normalize(module)
    installed_names = {_normalize(k) for k in context.installed_packages}
    req_names = {_normalize(k) for k in (context.requirements_packages or {})}

    is_installed = norm_module in installed_names
    in_requirements = norm_module in req_names

    # ── Priority 1: venv exists but is not activated ──────────────────────────────
    if not context.venv_active and context.venv_path:
        act = _activate_cmd(context.venv_path)
        req_suffix = " && pip install -r requirements.txt" if context.requirements_packages else ""
        return _diag(
            summary=f"Module '{module}' not found — virtualenv is not activated",
            explanation=(
                f"A virtualenv exists at '{context.venv_path}' but is not currently active. "
                f"Your command ran with system Python ({context.python_path}), which may not "
                f"have '{module}' installed. Activate the virtualenv to use its packages."
            ),
            fix_command=act + req_suffix,
            confidence="high",
        )

    # ── Priority 2: in requirements but not installed ─────────────────────────────
    if in_requirements and not is_installed:
        req_ver = (context.requirements_packages or {}).get(norm_module, "")
        return _diag(
            summary=f"Module '{module}' is in requirements.txt but not installed",
            explanation=(
                f"'{module}' is listed in requirements.txt"
                + (f" ({req_ver})" if req_ver else "")
                + f" but is not installed in the current environment ({context.python_path}). "
                f"Run pip install to sync your environment with requirements.txt."
            ),
            fix_command="pip install -r requirements.txt",
            confidence="high",
        )

    # ── Priority 3: not installed at all ─────────────────────────────────────────
    if not is_installed:
        return _diag(
            summary=f"Module '{module}' is not installed",
            explanation=(
                f"'{module}' is not installed in the current Python environment "
                f"({context.python_path}). "
                f"Install it with pip. If this is a project dependency, also add it to "
                f"requirements.txt or pyproject.toml."
            ),
            fix_command=f"pip install {module}",
            confidence="high",
        )

    return None


def rule_file_not_found(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """Rule 2: FileNotFoundError."""
    if error_info.error_type != "FileNotFoundError":
        return None

    file_path = _extract_file_path(error_info)
    if not file_path:
        return None

    variant = _find_case_variant(file_path, context.cwd)
    if variant:
        return _diag(
            summary=f"File '{file_path}' not found — did you mean '{Path(variant).name}'?",
            explanation=(
                f"'{file_path}' does not exist, but '{variant}' was found nearby. "
                f"This is often a case-sensitivity issue on Linux filesystems where "
                f"'config.json' and 'Config.json' are different files."
            ),
            confidence="medium",
        )

    return _diag(
        summary=f"File '{file_path}' not found",
        explanation=(
            f"'{file_path}' does not exist. "
            f"Your current working directory is '{context.cwd}'. "
            f"Check that the path is correct relative to where you're running the command."
        ),
        confidence="high",
    )


def rule_syntax_error(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """Rule 3: SyntaxError / IndentationError / TabError."""
    if error_info.error_type not in ("SyntaxError", "IndentationError", "TabError"):
        return None

    ref = error_info.file_references[-1] if error_info.file_references else None

    if ref:
        location = f"{ref.file_path}:{ref.line_number}"
        summary = f"{error_info.error_type} at {location}"

        # Fetch the offending line from the source snippet if gathered
        error_line = ""
        for snippet in context.source_snippets:
            if Path(snippet.file_path).name == Path(ref.file_path).name:
                error_line = snippet.error_line.strip()
                break

        explanation = (
            f"{error_info.error_type} in '{ref.file_path}' at line {ref.line_number}: "
            f"{error_info.error_message}."
        )
        if error_line:
            explanation += f"\nOffending line: {error_line!r}"
    else:
        summary = f"{error_info.error_type}: {error_info.error_message}"
        explanation = (
            f"{error_info.error_type}: {error_info.error_message}. "
            f"Check the file and line number shown in the traceback above."
        )

    # Syntax errors don't have a single shell command fix
    return _diag(summary, explanation, fix_command=None, confidence="high")


def rule_permission_error(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """Rule 4: PermissionError."""
    if error_info.error_type != "PermissionError":
        return None

    file_path = _extract_file_path(error_info)
    if not file_path:
        return None

    p = Path(file_path) if Path(file_path).is_absolute() else Path(context.cwd) / file_path

    if p.exists():
        if not os.access(p, os.R_OK):
            access_type, chmod_flag = "read", "+r"
        elif not os.access(p, os.W_OK):
            access_type, chmod_flag = "write", "+w"
        else:
            access_type, chmod_flag = "access", "+r"
        return _diag(
            summary=f"Permission denied: no {access_type} access to '{file_path}'",
            explanation=(
                f"'{file_path}' exists but your user does not have {access_type} permission. "
                f"This happens when a file is owned by root or another user. "
                f"You may need to use sudo or fix the file ownership."
            ),
            fix_command=f"chmod {chmod_flag} {file_path}",
            confidence="high",
        )

    return _diag(
        summary=f"Permission denied accessing '{file_path}'",
        explanation=(
            f"Permission was denied when trying to access '{file_path}'. "
            f"Check the file's ownership and permissions with `ls -la`."
        ),
        confidence="medium",
    )


def rule_connection_refused(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """Rule 5: ConnectionRefusedError — maps ports to known services."""
    if error_info.error_type not in ("ConnectionRefusedError", "ConnectionError"):
        return None

    # Extract port from error message or raw error
    port: int | None = None
    for text in (error_info.error_message, error_info.raw_error):
        m = re.search(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)", text)
        if m:
            port = int(m.group(1))
            break
    if port is None:
        for text in (error_info.error_message, error_info.raw_error):
            m = re.search(r"port[=\s:]+(\d+)", text, re.IGNORECASE)
            if m:
                port = int(m.group(1))
                break

    if port is None:
        return _diag(
            summary="Connection refused",
            explanation=(
                "A connection was refused. "
                "The target service is likely not running. "
                "Check that the service is started and verify the host and port."
            ),
            confidence="medium",
        )

    if port in _KNOWN_PORTS:
        service, fix = _KNOWN_PORTS[port]
        return _diag(
            summary=f"{service} is not running on port {port}",
            explanation=(
                f"Connection was refused on port {port} — the default {service} port. "
                f"{service} doesn't appear to be running. "
                f"Start it with the command below, or check if it's on a different port."
            ),
            fix_command=fix,
            confidence="high",
        )

    if port in _DEV_PORTS:
        return _diag(
            summary=f"No server running on port {port}",
            explanation=(
                f"Connection was refused on port {port}, a common dev-server port. "
                f"Make sure your development server is started before running this command."
            ),
            confidence="high",
        )

    return _diag(
        summary=f"Connection refused on port {port}",
        explanation=(
            f"The connection to port {port} was refused. "
            f"The service is likely not running or is bound to a different interface. "
            f"Check your process list and service configuration."
        ),
        confidence="medium",
    )


def rule_pytest_failures(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """
    Rule 9: pytest failure summary.

    Produces a human-readable summary of how many tests failed and what
    the primary failures were. Only fires on pytest output with FAILED lines.
    """
    raw = error_info.raw_error
    if "test session starts" not in raw and not re.search(r"^FAILED\s+", raw, re.MULTILINE):
        return None

    # Count from short summary line: "2 failed, 5 passed in 0.34s"
    count_m = re.search(r"(\d+) failed(?:,\s*(\d+) passed)?", raw)
    n_failed = int(count_m.group(1)) if count_m else 0
    n_passed = int(count_m.group(2)) if count_m and count_m.group(2) else 0

    # All FAILED lines (with reasons)
    failed_lines = re.findall(r"^FAILED\s+(.+?)\s+-\s+(.+)$", raw, re.MULTILINE)
    if not failed_lines and not n_failed:
        return None
    if n_failed == 0:
        n_failed = len(failed_lines)

    total = n_failed + n_passed
    total_str = f" out of {total}" if total > n_failed else ""
    plural = "s" if n_failed > 1 else ""
    summary = f"{n_failed} test{plural} failed{total_str}"

    if failed_lines:
        test_id, reason = failed_lines[0]
        short_name = test_id.split("::")[-1] if "::" in test_id else test_id
        explanation = f"Primary failure: {short_name} — {reason}."
        if len(failed_lines) > 1:
            others = ", ".join(
                f[0].split("::")[-1] if "::" in f[0] else f[0]
                for f in failed_lines[1:3]
            )
            explanation += f" Also failing: {others}."
    else:
        explanation = "Check the test output above for individual failure details."

    return _diag(summary, explanation, fix_command=None, confidence="medium")


def rule_version_conflict(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """Rule 7: Version mismatch / incompatibility detected in the error text."""
    version_keywords = {
        "version", "incompatible", "requires", "expected", "got version",
        "but found", "conflict", "incompatibility",
    }
    raw_lower = error_info.raw_error.lower()
    if not any(kw in raw_lower for kw in version_keywords):
        return None

    # Pattern: "requires SomePackage >=X.Y" or "SomePackage requires >=X.Y"
    m = re.search(
        r"requires?\s+([A-Za-z][A-Za-z0-9_-]+)\s*([><=!]+\s*[\d.]+)",
        error_info.raw_error,
        re.IGNORECASE,
    )
    if not m:
        return None

    req_pkg = _normalize(m.group(1))
    req_ver_spec = m.group(2).strip()
    installed_ver = context.installed_packages.get(req_pkg)
    if not installed_ver:
        return None

    ver_m = re.search(r"([\d.]+)", req_ver_spec)
    if ">=" in req_ver_spec and ver_m:
        fix = f"pip install '{req_pkg}>={ver_m.group(1)}'"
    else:
        fix = f"pip install --upgrade {req_pkg}"

    return _diag(
        summary=f"Version conflict: {req_pkg} {req_ver_spec} required, {installed_ver} installed",
        explanation=(
            f"A dependency requires {req_pkg} {req_ver_spec} but you have {installed_ver} installed. "
            f"Upgrade the package to resolve the conflict."
        ),
        fix_command=fix,
        confidence="medium",
    )


def rule_generic_python_exception(
    error_info: ErrorInfo, context: LocalContext
) -> Diagnosis | None:
    """
    Rule 10: Catch-all for recognised Python exception types.

    Provides a plain-English explanation of what the exception means.
    Confidence is always 'medium' — no specific fix is known.
    """
    if error_info.language != "python":
        return None

    explanation_text = _EXCEPTION_EXPLANATIONS.get(error_info.error_type)
    if not explanation_text:
        return None

    msg = error_info.error_message
    summary = (
        f"{error_info.error_type}: {msg[:80]}{'…' if len(msg) > 80 else ''}"
        if msg
        else error_info.error_type
    )
    explanation = explanation_text
    if msg:
        explanation += f"\nError detail: {msg}"

    return _diag(summary, explanation, fix_command=None, confidence="medium")


# ── Rule pipeline ─────────────────────────────────────────────────────────────────

_RULES = [
    rule_module_not_found,       # ModuleNotFoundError / ImportError (incl. venv check)
    rule_file_not_found,         # FileNotFoundError
    rule_syntax_error,           # SyntaxError / IndentationError / TabError
    rule_permission_error,       # PermissionError
    rule_connection_refused,     # ConnectionRefusedError
    rule_version_conflict,       # version incompatibility
    rule_pytest_failures,        # pytest FAILED summary
    rule_generic_python_exception,  # catch-all for known exceptions
]


def diagnose(error_info: ErrorInfo, context: LocalContext) -> Diagnosis | None:
    """
    Try each rule in priority order and return the first match.
    Returns None if no rule matches — caller falls through to LLM tier.
    """
    for rule in _RULES:
        try:
            result = rule(error_info, context)
            if result is not None:
                return result
        except Exception:
            # A rule must never crash the pipeline
            continue
    return None
