"""
config.py — Configuration management. (Day 10 implementation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model: str | None = None
    api_key: str | None = None
    no_ai: bool = False
    local: bool = False
    show_context: bool = False
    timeout: int = 300


def get_config(
    model: str | None = None,
    no_ai: bool = False,
    local: bool = False,
    show_context: bool = False,
) -> Config:
    """Merge CLI flags > env vars > defaults. Full implementation in Day 10."""
    resolved_model = model or os.environ.get("WTF_MODEL")
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    return Config(
        model=resolved_model,
        api_key=api_key,
        no_ai=no_ai or os.environ.get("WTF_NO_AI", "").lower() in ("1", "true"),
        local=local,
        show_context=show_context,
    )
