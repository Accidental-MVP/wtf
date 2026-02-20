"""
formatter.py — Rich terminal output formatting.

THIS IS WHAT MAKES THE TOOL SCREENSHOTTABLE.

Public API:
  print_diagnosis(command_result, error_info, context, diagnosis, ...)
  print_context(context, error_info, console)
  print_dry_run(error_info, context, model, console)
  print_no_diagnosis(error_info, console)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from wtf_cli.context import LocalContext, SourceSnippet
from wtf_cli.parser import ErrorInfo
from wtf_cli.rules import Diagnosis
from wtf_cli.runner import CommandResult

# ── Shared console (overridable per-call for testing) ────────────────────────────

# On Windows, sys.stdout may default to cp1252, which cannot encode Rich's
# Unicode glyphs (✗, ⏱, etc.).  Reconfigure to UTF-8 before creating the
# Console so all output — including runner.py streaming — uses the same
# encoding.  legacy_windows=False switches from the Win32 Console API to
# VT100/ANSI sequences, which modern Windows Terminal and VS Code support.
def _make_console() -> Console:
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return Console(legacy_windows=False)


_console = _make_console()


def _con(console: Console | None) -> Console:
    return console if console is not None else _console


# ── Colour tokens ────────────────────────────────────────────────────────────────

_RED       = "bold red"
_RED_DIM   = "red"
_GREEN     = "bold green"
_YELLOW    = "bold yellow"
_YELLOW_DIM = "yellow"
_CYAN      = "cyan"
_CYAN_DIM  = "dim cyan"
_DIM       = "dim"
_BOLD      = "bold"


# ── Contradiction / context helpers ──────────────────────────────────────────────

def _normalize_pkg(name: str) -> str:
    """PEP 503 normalisation: lowercase, collapse separators to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_contradictions(context: LocalContext) -> list[Text]:
    """
    Return styled Text lines for every package that is in requirements.txt
    but missing from the active environment — the core 'aha' moment.

    Also flags version mismatches where the installed version is clearly
    incompatible with the requirement specifier.
    """
    lines: list[Text] = []
    installed = context.installed_packages
    req = context.requirements_packages or {}

    if not req:
        return lines

    for raw_name, req_spec in sorted(req.items()):
        norm = _normalize_pkg(raw_name)
        installed_ver = installed.get(norm)

        if installed_ver is None:
            t = Text()
            t.append(f"  {raw_name}", style=_YELLOW)
            t.append(" in requirements.txt but ", style=_YELLOW_DIM)
            t.append("NOT installed", style=_YELLOW)
            lines.append(t)
        elif req_spec:
            # Very light version check: flag only obvious mismatches
            # e.g. req ">=2.0" but installed "1.9"
            _flag_version_mismatch(raw_name, req_spec, installed_ver, lines)

    return lines


def _flag_version_mismatch(
    pkg: str, req_spec: str, installed_ver: str, lines: list[Text]
) -> None:
    """Append a yellow line if installed_ver clearly violates req_spec."""
    # Only check simple >= constraints to avoid pulling in packaging dependency
    m = re.search(r">=\s*([\d.]+)", req_spec)
    if not m:
        return
    required = tuple(int(x) for x in m.group(1).split(".") if x.isdigit())
    try:
        actual = tuple(int(x) for x in installed_ver.split(".") if x.isdigit())
    except ValueError:
        return
    if actual < required:
        t = Text()
        t.append(f"  {pkg}", style=_YELLOW)
        t.append(f" {req_spec} required, ", style=_YELLOW_DIM)
        t.append(f"{installed_ver} installed", style=_YELLOW)
        lines.append(t)


def _build_context_table(
    context: LocalContext, error_info: ErrorInfo
) -> Table:
    """
    Build a compact two-column table for the context panel.

    Contradictions (inactive venv, missing packages) are highlighted in yellow.
    """
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style=_CYAN_DIM, no_wrap=True)
    tbl.add_column()

    # Python
    if context.python_version:
        tbl.add_row("Python", f"{context.python_version}  {context.python_path}")

    # Virtualenv
    if context.venv_active:
        tbl.add_row(
            "Virtualenv",
            Text.assemble(("active", "green"), f"  {context.venv_path or ''}"),
        )
    elif context.venv_path:
        tbl.add_row(
            "Virtualenv",
            Text.assemble(
                ("NOT ACTIVE", _YELLOW),
                ("  — exists at ", _YELLOW_DIM),
                (context.venv_path, _YELLOW_DIM),
            ),
        )

    # OS
    if context.os_info:
        tbl.add_row("OS", context.os_info)

    # Git
    if context.git_branch is not None:
        dirty_tag = Text.assemble(
            context.git_branch,
            ("  (dirty)", _YELLOW_DIM) if context.git_dirty else "",
        )
        tbl.add_row("Git", dirty_tag)

    # Packages summary
    n_installed = len(context.installed_packages)
    n_req = len(context.requirements_packages) if context.requirements_packages else 0
    if n_installed:
        pkg_summary = f"{n_installed} installed"
        if n_req:
            pkg_summary += f"  ·  {n_req} in requirements.txt"
        tbl.add_row("Packages", pkg_summary)

    # Env var names
    if context.env_var_names:
        shown = context.env_var_names[:8]
        suffix = f"  +{len(context.env_var_names) - 8} more" if len(context.env_var_names) > 8 else ""
        tbl.add_row("Env", ", ".join(shown) + suffix)

    # File references from error
    if error_info.file_references:
        ref = error_info.file_references[-1]
        tbl.add_row("Error at", f"{ref.file_path}:{ref.line_number}")

    return tbl


# ── Source snippet renderer ───────────────────────────────────────────────────────

def _render_snippet(snippet: SourceSnippet) -> Panel:
    """Render a source snippet with syntax highlighting and error-line pointer."""
    start_line = max(1, snippet.line_number - 5)

    # Build the code string with line numbers already embedded for Syntax
    code_lines: list[str] = []
    for i, line in enumerate(snippet.context_lines):
        lineno = start_line + i
        pointer = ">>>" if lineno == snippet.line_number else "   "
        code_lines.append(f"{pointer} {lineno:4d} │ {line}")

    code = "\n".join(code_lines)

    # Detect language from extension
    ext = Path(snippet.file_path).suffix.lstrip(".")
    lang = {"py": "python", "js": "javascript", "ts": "typescript"}.get(ext, "text")

    syn = Syntax(code, lang, theme="monokai", word_wrap=True)
    short_path = snippet.file_path
    return Panel(
        syn,
        title=f"[dim]{short_path}[/dim]",
        border_style="dim",
        padding=(0, 0),
    )


# ── Dry-run summary panel ─────────────────────────────────────────────────────────

def print_dry_run(
    error_info: ErrorInfo,
    context: LocalContext,
    model: str,
    console: Console | None = None,
) -> None:
    """
    Print a compact summary of what WOULD be sent to the LLM.

    Shown when --dry-run is passed. Does not call the API.
    """
    con = _con(console)

    # Import here to avoid circular import; ai.py also imports from here indirectly
    from wtf_cli.ai import _select_relevant_packages, build_prompt  # noqa: PLC0415

    relevant_pkgs = _select_relevant_packages(error_info, context)

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style=_CYAN_DIM, no_wrap=True, min_width=10)
    tbl.add_column()

    # Error
    err_str = error_info.error_type
    if error_info.error_message:
        msg = error_info.error_message
        err_str += f": {msg[:60]}{'…' if len(msg) > 60 else ''}"
    tbl.add_row("error", err_str)

    # Primary file ref
    if error_info.file_references:
        ref = error_info.file_references[-1]
        tbl.add_row("file", f"{ref.file_path}:{ref.line_number}")

    # Code snippets
    if context.source_snippets:
        total_lines = sum(len(s.context_lines) for s in context.source_snippets)
        names = ", ".join(Path(s.file_path).name for s in context.source_snippets)
        tbl.add_row("code", f"{total_lines} lines from {names}")

    # Python
    if context.python_version:
        tbl.add_row("python", context.python_version)

    # Packages
    n = len(relevant_pkgs)
    tbl.add_row("packages", f"{n} relevant package{'s' if n != 1 else ''}")

    # Env vars
    if context.env_var_names:
        tbl.add_row("env_vars", ", ".join(context.env_var_names[:6]))

    # OS
    if context.os_info:
        tbl.add_row("os", context.os_info)

    # Model
    tbl.add_row("model", model)

    panel = Panel(
        tbl,
        title="[bold cyan]Would send to API[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )
    con.print()
    con.print(panel)

    # Rough token estimate (4 chars ≈ 1 token)
    prompt = build_prompt(error_info, context)
    from wtf_cli.ai import _SYSTEM_PROMPT  # noqa: PLC0415
    full_text = _SYSTEM_PROMPT + "\n" + prompt
    est_tokens = len(full_text) // 4
    con.print(f"  [dim]Estimated tokens: ~{est_tokens}[/dim]")
    con.print()


# ── Context panel (standalone, for --show-context) ───────────────────────────────

def print_context(
    context: LocalContext,
    error_info: ErrorInfo,
    console: Console | None = None,
) -> None:
    """
    Print the 'Context detected' panel.

    Called explicitly for --show-context. Also called internally by
    print_diagnosis when show_context=True.
    """
    con = _con(console)

    tbl = _build_context_table(context, error_info)
    contradictions = _find_contradictions(context)

    group_items: list = [tbl]
    if contradictions:
        group_items.append(Text())  # blank line
        for line in contradictions:
            group_items.append(line)

    # Add source snippets if file references exist
    if context.source_snippets:
        group_items.append(Text())
        for snippet in context.source_snippets:
            group_items.append(_render_snippet(snippet))

    panel = Panel(
        Group(*group_items),
        title="[dim cyan]Context detected[/dim cyan]",
        border_style=_CYAN_DIM,
        padding=(0, 1),
    )
    con.print(panel)


# ── Main diagnosis display ────────────────────────────────────────────────────────

def print_diagnosis(
    command_result: CommandResult,
    error_info: ErrorInfo,
    context: LocalContext,
    diagnosis: Diagnosis,
    show_context: bool = False,
    copy_fix: bool = False,
    console: Console | None = None,
) -> None:
    """
    Print the full diagnosis panel — the main output of the tool.

    Args:
        command_result: Output from runner.run_command().
        error_info:     Structured error from parser.parse_error().
        context:        Local machine context from context.gather_context().
        diagnosis:      Diagnosis from rules.diagnose() or ai.diagnose().
        show_context:   If True, include the Context detected panel.
        copy_fix:       If True, copy the fix command to clipboard after display.
        console:        Optional Rich Console (uses module default if None).
    """
    con = _con(console)

    items: list = []

    # ── Error header ──────────────────────────────────────────────────────────────
    error_text = Text()
    error_text.append("✗  ", style=_RED)
    error_text.append(error_info.error_type, style=_RED)
    if error_info.error_message:
        error_text.append(f": {error_info.error_message}", style=_RED_DIM)
    items.append(error_text)

    # ── Context block ─────────────────────────────────────────────────────────────
    if show_context:
        items.append(Text())
        tbl = _build_context_table(context, error_info)
        contradictions = _find_contradictions(context)
        ctx_items: list = [tbl]
        if contradictions:
            ctx_items.append(Text())
            ctx_items.extend(contradictions)
        ctx_panel = Panel(
            Group(*ctx_items),
            title="[dim cyan]Context detected[/dim cyan]",
            border_style=_CYAN_DIM,
            padding=(0, 1),
        )
        items.append(ctx_panel)

    # ── Diagnosis section ─────────────────────────────────────────────────────────
    items.append(Text())

    diag_header = Text()
    diag_header.append("Diagnosis  ", style=_BOLD)
    diag_header.append(f"[{diagnosis.source} · {diagnosis.confidence} confidence]", style=_DIM)
    items.append(diag_header)

    items.append(Text(f"  {diagnosis.summary}"))
    items.append(Text())

    for line in diagnosis.explanation.splitlines():
        items.append(Text(f"  {line}"))

    # ── Fix command ───────────────────────────────────────────────────────────────
    if diagnosis.fix_command:
        items.append(Text())
        items.append(Text("Fix:", style=_GREEN))
        items.append(Text(f"  $ {diagnosis.fix_command}", style=_GREEN))

        if sys.stdin.isatty() and sys.stdout.isatty():
            items.append(Text())
            items.append(Text("  [dim]Run fix? [Y/n][/dim]", no_wrap=True))

    # ── Source snippets (show_context only) ───────────────────────────────────────
    if show_context and context.source_snippets:
        items.append(Text())
        for snippet in context.source_snippets:
            items.append(_render_snippet(snippet))

    # ── Metadata footer ───────────────────────────────────────────────────────────
    items.append(Text())
    meta_parts = [f"⏱ {command_result.duration:.2f}s", diagnosis.source]
    if diagnosis.tokens is not None:
        meta_parts.append(f"{diagnosis.tokens:,} tokens")
    if diagnosis.cost_usd is not None:
        meta_parts.append(f"${diagnosis.cost_usd:.4f}")
    items.append(Text(" · ".join(meta_parts), style=_DIM))

    # ── Render as a single panel ──────────────────────────────────────────────────
    panel = Panel(
        Group(*items),
        title="[bold]wtf[/bold]",
        border_style=_DIM,
        padding=(0, 1),
    )
    con.print()
    con.print(panel)

    # ── Interactive fix execution (after panel render) ────────────────────────────
    if diagnosis.fix_command and sys.stdin.isatty() and sys.stdout.isatty():
        con.print()
        try:
            answer = con.input("  Run fix? [Y/n] ").strip().lower()
            run_it = answer in ("", "y", "yes")
        except (KeyboardInterrupt, EOFError):
            run_it = False
            con.print()

        if run_it:
            con.print()
            fix_result = subprocess.run(diagnosis.fix_command, shell=True)
            if fix_result.returncode == 0:
                con.print()
                con.print("[bold green]✓ Fix applied![/bold green]")
                # Re-run the original command to confirm it's fixed
                con.print(f"[dim]Re-running: {command_result.command}[/dim]")
                rerun = subprocess.run(command_result.command, shell=True)
                if rerun.returncode == 0:
                    con.print("[bold green]✓ Fixed! Command now succeeds.[/bold green]")
                else:
                    con.print("[yellow]Command still fails. The fix may be incomplete.[/yellow]")
            else:
                con.print(f"[yellow]Fix exited with code {fix_result.returncode}[/yellow]")

    # ── Copy fix to clipboard ─────────────────────────────────────────────────────
    if copy_fix and diagnosis.fix_command:
        try:
            import pyperclip  # noqa: PLC0415
            pyperclip.copy(diagnosis.fix_command)
            con.print()
            con.print("[dim]Fix command copied to clipboard.[/dim]")
        except Exception:
            pass


# ── No-diagnosis fallback ─────────────────────────────────────────────────────────

def print_no_diagnosis(
    error_info: ErrorInfo,
    console: Console | None = None,
) -> None:
    """
    Print a minimal panel when no rule matched and AI is disabled.

    Always shows the error — never shows less than the original command would.
    """
    con = _con(console)

    error_text = Text()
    error_text.append("✗  ", style=_RED)
    error_text.append(error_info.error_type, style=_RED)
    if error_info.error_message:
        error_text.append(f": {error_info.error_message}", style=_RED_DIM)

    hint = Text(
        "  No rule matched. Add an API key and remove --no-ai for AI-powered diagnosis.",
        style=_DIM,
    )

    panel = Panel(
        Group(error_text, Text(), hint),
        title="[bold]wtf[/bold]",
        border_style=_DIM,
        padding=(0, 1),
    )
    con.print()
    con.print(panel)
