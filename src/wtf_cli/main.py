"""
main.py — CLI entry point (Typer app).

Wires together: runner → parser → context → rules/ai → formatter.
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

    # Run the command with real-time output streaming.
    result = run_command(command)

    if result.exit_code == 0:
        console.print(f"\n[bold green]✓[/bold green] Command succeeded in {result.duration:.2f}s")
        raise typer.Exit(0)

    # Command failed — full diagnosis pipeline will be wired in later days.
    console.print(
        f"\n[dim]wtf: command exited with code {result.exit_code} "
        f"({result.duration:.2f}s) — diagnosis coming in a future day[/dim]"
    )
    raise typer.Exit(result.exit_code)
