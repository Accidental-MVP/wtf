# wtf

Diagnoses terminal errors using your local code, packages, and environment.

Not another "paste into ChatGPT" wrapper. `wtf` understands YOUR machine.

<!-- GIF DEMO HERE - 12 seconds showing: failing pytest → wtf diagnosis → fix → passing tests -->

## The Difference

| | ChatGPT | wtf |
|---|---|---|
| Knows your error | Yes | Yes |
| Knows your Python version | No | Yes |
| Knows your installed packages | No | Yes |
| Knows your requirements.txt | No | Yes |
| Detects version mismatches | No | Yes |
| Detects inactive virtualenv | No | Yes |
| Suggests specific fix command | Sometimes | Yes |
| Works in your terminal | No | Yes |
| Works without API key | No | Yes (rule-based mode) |

## Install

```bash
pip install wtf-cli
```

## Usage

```bash
# Wraps any command. If it fails, you get a diagnosis.
wtf pytest tests/
wtf python app.py
wtf npm run build

# Works without an API key (rule-based diagnosis for common errors)
wtf --no-ai pytest tests/

# See what context would be sent before sending
wtf --dry-run pytest tests/

# Use local LLM (Ollama) — nothing leaves your machine
wtf --local pytest tests/
```

## Example

```bash
$ wtf pytest tests/test_auth.py
```

<!-- SCREENSHOT of beautiful output showing context detection + diagnosis + fix -->

## Privacy

`wtf` sends minimal context to the LLM API:
- Error message and traceback
- Relevant source code (5 lines around the error, NOT full files)
- Package names and versions
- Environment variable NAMES only (never values)
- OS and Python version

Use `--dry-run` to see exactly what would be sent.
Use `--no-ai` for fully offline rule-based diagnosis.
Use `--local` for local Ollama inference.

## How It Works

1. Runs your command and captures the error
2. Gathers local context: Python version, packages, virtualenv, env vars, relevant source code
3. For common errors (ModuleNotFoundError, FileNotFoundError, etc.): diagnoses instantly with rules (no API needed)
4. For complex errors: sends minimal context to an LLM for specific diagnosis
5. Suggests a fix command you can run immediately

## Configuration

Set your preferred model:
```bash
export WTF_MODEL=claude-haiku  # or gpt-4o-mini, ollama/llama3, etc.
```

API keys (any one):
```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
```

## License

MIT
