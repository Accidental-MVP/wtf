"""
ai.py — Tier 2 LLM-powered diagnosis via litellm.

Used when no rule matches the error. Builds a minimal, privacy-respecting
context prompt and calls the configured LLM. Falls back gracefully on
missing API keys or network failures — never crashes the pipeline.

Public API:
  diagnose(error_info, context, model, dry_run, local, api_key, command) -> Diagnosis
  build_prompt(error_info, context, command) -> str
"""

from __future__ import annotations

import os
import re

from wtf_cli.context import LocalContext
from wtf_cli.parser import ErrorInfo
from wtf_cli.rules import Diagnosis

# litellm is an optional runtime dependency — imported at module level so tests
# can patch `wtf_cli.ai.litellm` cleanly. Set to None when not installed.
try:
    import litellm
except ImportError:  # pragma: no cover
    litellm = None  # type: ignore[assignment]


# ── Constants ────────────────────────────────────────────────────────────────────

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-3-5-20241022"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_LOCAL_MODEL = "ollama/llama3"

_MAX_TRACEBACK_LINES = 30
_MAX_PACKAGES = 20

_SYSTEM_PROMPT = """\
You are a senior developer diagnosing a terminal error. You have access to \
the user's local machine context. Give a specific diagnosis, not generic advice.

Rules:
- Be concise. Max 3 sentences for explanation.
- If you can suggest a fix command, do so.
- Reference specific file paths, package versions, and env vars from the context.
- Never say "usually" or "typically" — be specific to THIS context.
- If you're not sure, say so.

Respond ONLY in this exact format:
SUMMARY: <one-line summary of what is wrong>
EXPLANATION: <2-3 sentences specific to this context>
FIX: <shell command that fixes the issue, or "no single command fix">"""


# ── Package filtering ─────────────────────────────────────────────────────────────

def _select_relevant_packages(
    error_info: ErrorInfo,
    context: LocalContext,
    max_packages: int = _MAX_PACKAGES,
) -> dict[str, str]:
    """
    Return only the installed packages most relevant to this error.

    Priority:
      1. Packages mentioned in the error message or traceback
      2. Packages found in import statements in source snippets
      3. Packages from requirements.txt (fill remaining slots)
    """
    installed = context.installed_packages
    if not installed:
        return {}

    relevant: dict[str, str] = {}

    # Error message + traceback as a single searchable string
    error_text = (
        error_info.error_message + " " + " ".join(error_info.traceback_lines)
    ).lower()

    for pkg, ver in installed.items():
        if len(relevant) >= max_packages:
            break
        pkg_lower = pkg.lower()
        pkg_under = pkg_lower.replace("-", "_")
        if pkg_lower in error_text or pkg_under in error_text:
            relevant[pkg] = ver

    # Packages imported in source snippets
    for snippet in context.source_snippets:
        if len(relevant) >= max_packages:
            break
        for line in snippet.context_lines:
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            m = re.match(r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped)
            if not m:
                continue
            mod = m.group(1).lower()
            for pkg, ver in installed.items():
                if len(relevant) >= max_packages:
                    break
                if pkg.lower().replace("-", "_") == mod and pkg not in relevant:
                    relevant[pkg] = ver

    # Fill remaining slots with requirements.txt packages that are installed
    req_pkgs = context.requirements_packages or {}
    for req_name in req_pkgs:
        if len(relevant) >= max_packages:
            break
        norm = req_name.lower().replace("_", "-")
        if norm in installed and norm not in relevant:
            relevant[norm] = installed[norm]

    return relevant


# ── Prompt construction ───────────────────────────────────────────────────────────

def _format_source_snippets(context: LocalContext) -> str:
    """Format source snippets for the prompt, with line numbers and error pointer."""
    if not context.source_snippets:
        return "none"

    parts: list[str] = []
    for snippet in context.source_snippets:
        header = f"# {snippet.file_path} (error at line {snippet.line_number})"
        start_line = max(1, snippet.line_number - 5)
        numbered: list[str] = []
        for i, line in enumerate(snippet.context_lines):
            lineno = start_line + i
            marker = ">>>" if lineno == snippet.line_number else "   "
            numbered.append(f"{marker} {lineno:4d} | {line}")
        parts.append(header + "\n" + "\n".join(numbered))

    return "\n\n".join(parts)


def build_prompt(
    error_info: ErrorInfo,
    context: LocalContext,
    command: str = "",
) -> str:
    """
    Build the user-turn prompt from error_info + context.

    Keeps the prompt minimal: capped traceback, relevant packages only,
    no raw file trees, no full pip freeze.
    """
    # Traceback (capped)
    tb_lines = error_info.traceback_lines[:_MAX_TRACEBACK_LINES]
    traceback_text = "\n".join(tb_lines) if tb_lines else "none"

    # Virtualenv status
    if context.venv_active:
        venv_status = f"active ({context.venv_path})"
    elif context.venv_path:
        venv_status = f"NOT ACTIVE — exists at {context.venv_path}"
    else:
        venv_status = "none detected"

    # Relevant packages only
    relevant_pkgs = _select_relevant_packages(error_info, context)
    if relevant_pkgs:
        pkg_lines = "\n".join(
            f"  {name}=={ver}" for name, ver in sorted(relevant_pkgs.items())
        )
        packages_text = "\n" + pkg_lines
    else:
        packages_text = " none"

    # Requirements packages
    req_pkgs = context.requirements_packages
    if req_pkgs:
        req_lines = "\n".join(
            f"  {name}{(' ' + ver) if ver else ''}" for name, ver in sorted(req_pkgs.items())
        )
        req_text = "\n" + req_lines
    else:
        req_text = " none"

    # Git info
    if context.git_branch is not None:
        git_text = f"{context.git_branch}, {'dirty' if context.git_dirty else 'clean'}"
    else:
        git_text = "not a git repo"

    # Env var names
    env_text = ", ".join(context.env_var_names) if context.env_var_names else "none"

    # Source code snippets
    snippets_text = _format_source_snippets(context)

    lines: list[str] = []
    if command:
        lines.append(f"Command: {command}")
    lines.append(f"Error type: {error_info.error_type}")
    lines.append(f"Error message: {error_info.error_message}")
    lines.append(f"\nTraceback:\n{traceback_text}")
    lines.append(
        "\nLocal context:"
        f"\n- Python: {context.python_version} ({context.python_path})"
        f"\n- Virtualenv: {venv_status}"
        f"\n- OS: {context.os_info}"
        f"\n- Relevant env vars (names only): {env_text}"
        f"\n- Installed packages (relevant):{packages_text}"
        f"\n- Requirements:{req_text}"
        f"\n- Git: {git_text}"
    )
    lines.append(f"\nSource code at error location:\n{snippets_text}")

    return "\n".join(lines)


# ── Response parsing ──────────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[str, str, str | None]:
    """
    Parse the structured LLM response into (summary, explanation, fix_command).

    Expected format:
      SUMMARY: one line
      EXPLANATION: 2-3 sentences
      FIX: command (or "no single command fix")

    Falls back to using the raw text as explanation if the format is absent.
    """
    summary = ""
    explanation = ""
    fix_command: str | None = None

    m = re.search(r"^SUMMARY:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        summary = m.group(1).strip()

    # EXPLANATION may span multiple lines until FIX: or end of string
    m = re.search(
        r"^EXPLANATION:\s*(.+?)(?=\nFIX:|\Z)",
        text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if m:
        explanation = m.group(1).strip()

    m = re.search(r"^FIX:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        fix_raw = m.group(1).strip()
        _no_fix = {"no single command fix", "none", "n/a", "no fix", "no command"}
        if fix_raw.lower() not in _no_fix:
            fix_command = fix_raw

    # Fallback: structured format not found → use raw text
    if not summary and not explanation:
        stripped = [l.strip() for l in text.strip().splitlines() if l.strip()]
        first = stripped[0] if stripped else "AI diagnosis"
        summary = (first[:97] + "…") if len(first) > 100 else first
        explanation = text.strip()
    elif not summary:
        summary = (explanation[:97] + "…") if len(explanation) > 100 else explanation
    elif not explanation:
        explanation = summary

    return summary, explanation, fix_command


# ── Model selection ───────────────────────────────────────────────────────────────

def _resolve_model(
    model: str | None,
    local: bool,
    api_key: str | None,
) -> str:
    """Determine the litellm model string to use based on flags and env vars."""
    if local:
        return model or _DEFAULT_LOCAL_MODEL

    if model:
        return model

    # Auto-select the cheapest model for the available API key
    if os.environ.get("ANTHROPIC_API_KEY") or (
        api_key and api_key.startswith("sk-ant")
    ):
        return _DEFAULT_ANTHROPIC_MODEL
    if os.environ.get("OPENAI_API_KEY") or (
        api_key and api_key.startswith("sk-") and not api_key.startswith("sk-ant")
    ):
        return _DEFAULT_OPENAI_MODEL
    if os.environ.get("OPENROUTER_API_KEY"):
        return f"openrouter/{_DEFAULT_OPENAI_MODEL}"

    # Default — will fail with an auth error if no key is set (caught below)
    return _DEFAULT_ANTHROPIC_MODEL


# ── Public API ────────────────────────────────────────────────────────────────────

def diagnose(
    error_info: ErrorInfo,
    context: LocalContext,
    model: str | None = None,
    dry_run: bool = False,
    local: bool = False,
    api_key: str | None = None,
    command: str = "",
) -> Diagnosis:
    """
    Use an LLM to diagnose the error. Returns a Diagnosis with source="ai".

    Args:
        error_info:  Structured error from parser.parse_error().
        context:     Local machine context from context.gather_context().
        model:       litellm model string override (e.g. "gpt-4o", "claude-sonnet-4-6").
        dry_run:     Return the prompt that WOULD be sent without calling the API.
        local:       Route to Ollama via litellm — nothing leaves the machine.
        api_key:     Explicit API key (falls back to env vars when not provided).
        command:     Original command string (included in prompt for context).

    Returns:
        Diagnosis with source="ai" (or "dry-run"). Never raises.
    """
    prompt = build_prompt(error_info, context, command=command)
    resolved_model = _resolve_model(model, local, api_key)

    # ── Dry-run: return the prompt without calling the API ────────────────────────
    if dry_run:
        dry_content = (
            f"[DRY RUN — would send to {resolved_model}]\n\n"
            f"--- System prompt ---\n{_SYSTEM_PROMPT}\n\n"
            f"--- User prompt ---\n{prompt}"
        )
        return Diagnosis(
            summary=f"Dry run — would call {resolved_model}",
            explanation=dry_content,
            fix_command=None,
            confidence="medium",
            source="dry-run",
        )

    # ── Check for API key (not required for local/Ollama) ────────────────────────
    if not local:
        has_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not has_key:
            return Diagnosis(
                summary="No API key configured — AI diagnosis unavailable",
                explanation=(
                    "No API key was found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                    "to enable AI-powered diagnosis, or use --no-ai for rule-based "
                    "diagnosis only, or use --local with Ollama for local inference."
                ),
                fix_command="export ANTHROPIC_API_KEY=sk-ant-...",
                confidence="high",
                source="ai",
            )

    # ── Call the LLM via litellm ──────────────────────────────────────────────────
    try:
        if litellm is None:
            raise ImportError("No module named 'litellm'")

        litellm.suppress_debug_info = True

        call_kwargs: dict = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        if api_key:
            call_kwargs["api_key"] = api_key

        response = litellm.completion(**call_kwargs)

        raw_text: str = response.choices[0].message.content or ""

        tokens: int | None = None
        if response.usage:
            tokens = response.usage.total_tokens

        cost_usd: float | None = None
        try:
            cost_usd = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        summary, explanation, fix_command = _parse_response(raw_text)

        return Diagnosis(
            summary=summary,
            explanation=explanation,
            fix_command=fix_command,
            confidence="medium",
            source="ai",
            tokens=tokens,
            cost_usd=cost_usd,
        )

    except ImportError:
        return Diagnosis(
            summary="litellm not installed",
            explanation=(
                "The 'litellm' package is required for AI-powered diagnosis. "
                "Install it with: pip install litellm\n"
                "Or use --no-ai for rule-based diagnosis only."
            ),
            fix_command="pip install litellm",
            confidence="high",
            source="ai",
        )

    except Exception as exc:
        error_cls = type(exc).__name__
        error_msg = str(exc)

        if "AuthenticationError" in error_cls or "401" in error_msg or "Unauthorized" in error_msg:
            return Diagnosis(
                summary="API authentication failed",
                explanation=(
                    f"The API key was rejected when calling {resolved_model}. "
                    "Check that your key is valid and has not expired. "
                    "Use --no-ai for rule-based diagnosis while you resolve this."
                ),
                fix_command=None,
                confidence="high",
                source="ai",
            )

        if "RateLimitError" in error_cls or "429" in error_msg:
            return Diagnosis(
                summary="API rate limit reached",
                explanation=(
                    f"The rate limit for {resolved_model} was hit. "
                    "Wait a moment and try again, or use --no-ai for instant "
                    "rule-based diagnosis."
                ),
                fix_command=None,
                confidence="high",
                source="ai",
            )

        return Diagnosis(
            summary=f"AI diagnosis unavailable ({error_cls})",
            explanation=(
                f"Could not reach the AI service ({resolved_model}). "
                f"Error: {error_msg[:200]}. "
                "Falling back to rule-based diagnosis where available."
            ),
            fix_command=None,
            confidence="medium",
            source="ai",
        )
