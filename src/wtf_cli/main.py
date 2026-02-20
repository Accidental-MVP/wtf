"""
main.py — CLI entry point (Typer app).

Wires together: runner → parser → context → rules → ai → output.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from wtf_cli import __version__
from wtf_cli.runner import run_command

app = typer.Typer(
    name="wtf",
    help="Diagnoses terminal errors using your local code, packages, and environment.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wtf-cli {__version__}")
        raise typer.Exit()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def main(
    ctx: typer.Context,
    # Flags
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show context that would be sent; don't call LLM.")] = False,
    no_ai: Annotated[bool, typer.Option("--no-ai", help="Rule-based diagnosis only (no API key needed).")] = False,
    local: Annotated[bool, typer.Option("--local", help="Use Ollama for local LLM inference.")] = False,
    show_context: Annotated[bool, typer.Option("--show-context", help="Show gathered context alongside diagnosis.")] = False,
    model: Annotated[str | None, typer.Option("--model", help="LLM model to use (e.g. claude-haiku, gpt-4o-mini).")] = None,
    copy: Annotated[bool, typer.Option("--copy", help="Copy the fix command to clipboard after displaying.")] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
) -> None:
    """
    Run COMMAND [ARGS]... and diagnose any errors.

    Example: wtf pytest tests/
    """
    command = ctx.args
    if not command:
        console.print("[red]Error:[/red] No command provided. Example: wtf pytest tests/")
        raise typer.Exit(1)

    # ── Run the command with real-time output streaming ───────────────────────────
    result = run_command(command)

    if result.exit_code == 0:
        console.print(f"\n[bold green]✓[/bold green] Command succeeded in {result.duration:.2f}s")
        raise typer.Exit(0)

    # ── Command failed — run the full diagnosis pipeline ─────────────────────────
    from wtf_cli.parser import parse_error
    from wtf_cli.context import gather_context
    from wtf_cli.rules import diagnose as rules_diagnose
    from wtf_cli.config import get_config
    import wtf_cli.ai as ai_module

    config = get_config(model=model, no_ai=no_ai, local=local, show_context=show_context)

    console.print()  # blank line separator after command output

    # Parse error
    error_info = parse_error(result.stderr, result.stdout, result.command)

    # Gather local context (runs in parallel, 2s budget)
    with console.status("[dim]Gathering local context…[/dim]", spinner="dots"):
        context = gather_context(error_info)

    # ── Optional: show gathered context ──────────────────────────────────────────
    if config.show_context or dry_run:
        _print_context(context, error_info)

    # ── Dry-run: show what would be sent, then exit ───────────────────────────────
    if dry_run:
        diagnosis = ai_module.diagnose(
            error_info,
            context,
            model=config.model,
            dry_run=True,
            local=config.local,
            api_key=config.api_key,
            command=result.command,
        )
        console.print(Panel(diagnosis.explanation, title="[bold cyan]Would send to API[/bold cyan]", border_style="cyan"))
        raise typer.Exit(result.exit_code)

    # ── Tier 1: rule-based diagnosis ──────────────────────────────────────────────
    diagnosis = rules_diagnose(error_info, context)

    # ── Tier 2: AI diagnosis (if no rule matched and AI is enabled) ───────────────
    if diagnosis is None and not config.no_ai:
        with console.status(f"[dim]Asking AI ({config.model or 'auto'})…[/dim]", spinner="dots"):
            diagnosis = ai_module.diagnose(
                error_info,
                context,
                model=config.model,
                dry_run=False,
                local=config.local,
                api_key=config.api_key,
                command=result.command,
            )

    # ── Print diagnosis ───────────────────────────────────────────────────────────
    if diagnosis is not None:
        _print_diagnosis(diagnosis, error_info, result.duration, copy_fix=copy)
    else:
        # No rule matched and AI is disabled — show raw error info
        _print_no_diagnosis(error_info)

    raise typer.Exit(result.exit_code)


# ── Output helpers ────────────────────────────────────────────────────────────────

def _print_context(context, error_info) -> None:
    """Print a compact context summary block."""
    from wtf_cli.context import LocalContext

    lines: list[str] = []

    if context.python_version:
        venv = f" · venv: {context.venv_path}" if context.venv_active else (
            f" · [yellow]venv NOT active[/yellow] ({context.venv_path})" if context.venv_path else ""
        )
        lines.append(f"Python {context.python_version} ({context.python_path}){venv}")

    if context.os_info:
        lines.append(f"OS: {context.os_info}")

    if context.git_branch is not None:
        dirty = " [yellow](dirty)[/yellow]" if context.git_dirty else ""
        lines.append(f"Git: {context.git_branch}{dirty}")

    if context.installed_packages:
        lines.append(f"Packages: {len(context.installed_packages)} installed")

    if context.requirements_packages is not None:
        req_count = len(context.requirements_packages)
        lines.append(f"requirements.txt: {req_count} packages")

    if context.env_var_names:
        lines.append(f"Env vars: {', '.join(context.env_var_names[:8])}")

    if lines:
        console.print(Panel(
            "\n".join(lines),
            title="[dim cyan]Context detected[/dim cyan]",
            border_style="dim cyan",
        ))


def _print_diagnosis(diagnosis, error_info, duration: float, copy_fix: bool = False) -> None:
    """Print the diagnosis as a styled Rich panel."""
    source_badge = f"[dim]{diagnosis.source}[/dim]"
    confidence_badge = f"[dim]{diagnosis.confidence} confidence[/dim]"

    # Error header
    error_header = Text()
    error_header.append("✗ ", style="bold red")
    error_header.append(error_info.error_type, style="bold red")
    if error_info.error_message:
        error_header.append(f": {error_info.error_message}", style="red")

    # Build body
    body_lines: list[str] = [
        error_header.markup if hasattr(error_header, "markup") else str(error_header),
        "",
        f"[bold]Diagnosis[/bold] [{source_badge} · {confidence_badge}]",
        f"  {diagnosis.summary}",
        "",
        f"  {diagnosis.explanation.replace(chr(10), chr(10) + '  ')}",
    ]

    if diagnosis.fix_command:
        body_lines += [
            "",
            "[bold green]Fix:[/bold green]",
            f"  [bold green]$ {diagnosis.fix_command}[/bold green]",
        ]

    # Metadata footer
    meta_parts = [f"{duration:.2f}s", diagnosis.source]
    if diagnosis.tokens is not None:
        meta_parts.append(f"{diagnosis.tokens} tokens")
    if diagnosis.cost_usd is not None:
        meta_parts.append(f"${diagnosis.cost_usd:.4f}")
    footer = " · ".join(meta_parts)

    # Print components
    console.print()
    console.print(
        Text.assemble(
            ("✗ ", "bold red"),
            (error_info.error_type, "bold red"),
            (f": {error_info.error_message}" if error_info.error_message else "", "red"),
        )
    )
    console.print()
    console.print(
        f"[bold]Diagnosis[/bold]  [{diagnosis.source} · {diagnosis.confidence} confidence]"
    )
    console.print(f"  {diagnosis.summary}")
    console.print()
    for line in diagnosis.explanation.splitlines():
        console.print(f"  {line}")

    if diagnosis.fix_command:
        console.print()
        console.print("[bold green]Fix:[/bold green]")
        console.print(f"  [bold green]$ {diagnosis.fix_command}[/bold green]")

        # Interactive prompt if terminal is interactive
        if sys.stdin.isatty() and sys.stdout.isatty():
            console.print()
            try:
                run_it = typer.confirm("  Run fix?", default=True)
            except (KeyboardInterrupt, EOFError):
                run_it = False

            if run_it:
                import subprocess
                console.print()
                fix_result = subprocess.run(diagnosis.fix_command, shell=True)
                if fix_result.returncode == 0:
                    console.print("[bold green]✓ Fix applied![/bold green]")
                else:
                    console.print(f"[yellow]Fix exited with code {fix_result.returncode}[/yellow]")

    if copy_fix and diagnosis.fix_command:
        try:
            import pyperclip
            pyperclip.copy(diagnosis.fix_command)
            console.print()
            console.print("[dim]Fix command copied to clipboard.[/dim]")
        except Exception:
            pass

    console.print()
    console.print(f"[dim]{footer}[/dim]")


def _print_no_diagnosis(error_info) -> None:
    """Fallback when no rule matched and AI is disabled."""
    console.print()
    console.print(
        Text.assemble(
            ("✗ ", "bold red"),
            (error_info.error_type, "bold red"),
            (f": {error_info.error_message}" if error_info.error_message else "", "red"),
        )
    )
    console.print()
    console.print(
        "[dim]No rule matched this error. Run without --no-ai for AI-powered diagnosis.[/dim]"
    )
