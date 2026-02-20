"""
main.py — CLI entry point (Typer app).

Wires together: runner → parser → context → rules → ai → formatter.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from wtf_cli import __version__
from wtf_cli.runner import run_command

app = typer.Typer(
    name="wtf",
    help="Diagnoses terminal errors using your local code, packages, and environment.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wtf-cli {__version__}")
        raise typer.Exit()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def main(
    ctx: typer.Context,
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
    from wtf_cli import formatter
    import wtf_cli.ai as ai_module

    config = get_config(model=model, no_ai=no_ai, local=local, show_context=show_context)

    # Parse the error
    error_info = parse_error(result.stderr, result.stdout, result.command)

    # Gather local context (runs in parallel, 2s budget)
    with console.status("[dim]Gathering local context…[/dim]", spinner="dots"):
        context = gather_context(error_info)

    # ── Dry-run: show what would be sent, then exit ───────────────────────────────
    if dry_run:
        resolved_model = config.model or ai_module._resolve_model(None, config.local, config.api_key)
        formatter.print_dry_run(error_info, context, resolved_model)
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
                local=config.local,
                api_key=config.api_key,
                command=result.command,
            )

    # ── Format and print ──────────────────────────────────────────────────────────
    if diagnosis is not None:
        formatter.print_diagnosis(
            result,
            error_info,
            context,
            diagnosis,
            show_context=config.show_context,
            copy_fix=copy,
        )
    else:
        formatter.print_no_diagnosis(error_info)

    raise typer.Exit(result.exit_code)
