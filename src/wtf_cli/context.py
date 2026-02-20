"""
context.py — Local context gathering.

All gatherers run in parallel via ThreadPoolExecutor with a 2-second total
budget. Each fails silently and returns its default value so slow or broken
system utilities never block the user.

Privacy rules (enforced throughout):
- Env var VALUES are never stored or returned.
- Source snippets are max 11 lines from files in the error traceback only.
- Full file contents are never captured.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from wtf_cli.parser import ErrorInfo


# ── Dataclasses ─────────────────────────────────────────────────────────────────

@dataclass
class SourceSnippet:
    file_path: str
    line_number: int
    context_lines: list[str]   # up to 11 lines: 5 before + error + 5 after
    error_line: str            # the specific line that errored


@dataclass
class LocalContext:
    python_version: str | None = None
    python_path: str | None = None
    venv_active: bool = False
    venv_path: str | None = None        # active venv path, or nearest local venv if inactive
    installed_packages: dict[str, str] = field(default_factory=dict)
    requirements_packages: dict[str, str] | None = None
    pyproject_packages: list[str] | None = None
    os_info: str = ""
    env_var_names: list[str] = field(default_factory=list)
    cwd: str = ""
    git_branch: str | None = None
    git_dirty: bool | None = None
    source_snippets: list[SourceSnippet] = field(default_factory=list)
    docker_running: bool | None = None


# ── Individual gatherers ─────────────────────────────────────────────────────────

def get_python_info() -> dict:
    """
    Return Python version, executable path, and virtualenv status.

    venv_active = True  → a virtualenv is currently activated
    venv_path           → active venv path when active; nearest local .venv
                          directory when inactive (useful for rule #8)
    """
    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    python_path = str(sys.executable)

    # Active venv: VIRTUAL_ENV (virtualenv/venv) or CONDA_PREFIX (conda)
    active_venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    venv_active = bool(active_venv)
    venv_path = active_venv

    # If no active venv, look for an inactive local one so rules can detect it
    if not venv_active:
        cwd = Path.cwd()
        py_bin = "python.exe" if sys.platform == "win32" else "python"
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        for name in (".venv", "venv", "env"):
            candidate = cwd / name
            if (candidate / bin_dir / py_bin).exists():
                venv_path = str(candidate)
                break

    return {
        "python_version": python_version,
        "python_path": python_path,
        "venv_active": venv_active,
        "venv_path": venv_path,
    }


def get_installed_packages() -> dict[str, str]:
    """
    Return {lowercase_package_name: version} for every package in the
    current Python environment. Uses `pip list --format=json` for reliable
    structured output.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        return {p["name"].lower(): p["version"] for p in json.loads(result.stdout)}
    except Exception:
        return {}


def _parse_requirements_txt(path: Path) -> dict[str, str]:
    """
    Parse a requirements.txt into {normalized_name: version_spec}.

    Handles:  package==1.0  package>=1.0,<2  package[extras]  # comments
    Skips:    -r includes  --flags  VCS URLs (git+https://…)
    """
    packages: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-") or "://" in line:
                continue
            line = line.split("#")[0].strip()
            if not line:
                continue
            # "package[extras] specifiers" or bare "package"
            m = re.match(
                r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*([><=!~^,\s].*)?\s*$",
                line,
            )
            if m:
                name = m.group(1).lower().replace("_", "-")
                version = (m.group(2) or "").strip()
                packages[name] = version
    except Exception:
        pass
    return packages


def _parse_pyproject_deps(path: Path) -> list[str] | None:
    """
    Extract the raw dependency strings from [project.dependencies] in
    a pyproject.toml. Returns None if the section is absent or unreadable.

    Uses stdlib tomllib (Python 3.11+) when available; falls back to a
    simple regex extractor for Python 3.10.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")

        # stdlib tomllib — Python 3.11+
        try:
            import tomllib  # type: ignore[import]
            data = tomllib.loads(content)
            deps = data.get("project", {}).get("dependencies", [])
            return list(deps) if deps else None
        except ImportError:
            pass

        # Regex fallback: find `dependencies = [...]` inside the [project] section.
        # Handles both inline and multiline array syntax.
        m = re.search(
            r"^\[project\].*?^dependencies\s*=\s*\[(.*?)\]",
            content,
            re.DOTALL | re.MULTILINE,
        )
        if not m:
            return None
        deps = re.findall(r'"([^"]+)"', m.group(1))
        return deps or None
    except Exception:
        return None


def get_requirements() -> dict[str, str] | None:
    """
    Search cwd and up to 3 parent directories for requirements.txt.
    Returns {package: version_spec} or None if not found.
    """
    cwd = Path.cwd()
    for directory in [cwd, *list(cwd.parents[:3])]:
        req = directory / "requirements.txt"
        if req.exists():
            result = _parse_requirements_txt(req)
            return result if result else None
    return None


def get_pyproject_packages() -> list[str] | None:
    """
    Search cwd and up to 3 parent directories for pyproject.toml.
    Returns the raw [project.dependencies] list or None.
    """
    cwd = Path.cwd()
    for directory in [cwd, *list(cwd.parents[:3])]:
        pyproject = directory / "pyproject.toml"
        if pyproject.exists():
            return _parse_pyproject_deps(pyproject)
    return None


def get_os_info() -> str:
    """Return a human-readable OS string: 'macOS 14.2 arm64', 'Ubuntu 22.04 x86_64', etc."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver} {machine}"

    if system == "Linux":
        # /etc/os-release is the modern standard (systemd distros)
        try:
            info: dict[str, str] = {}
            with open("/etc/os-release") as fh:
                for line in fh:
                    k, _, v = line.strip().partition("=")
                    info[k] = v.strip('"')
            pretty = info.get("PRETTY_NAME") or info.get("NAME", "Linux")
            return f"{pretty} {machine}"
        except Exception:
            return f"Linux {platform.release()} {machine}"

    if system == "Windows":
        return f"Windows {platform.release()} {machine}"

    return f"{system} {platform.release()} {machine}"


# Env var name fragments that suggest configuration/secrets
_ENV_KEYWORDS = frozenset({
    "KEY", "SECRET", "TOKEN", "URL", "HOST", "PORT", "DB", "API",
    "AUTH", "PASSWORD", "PASS", "ENDPOINT", "BUCKET", "REGION",
    "QUEUE", "TOPIC", "EMAIL", "CERT", "OAUTH", "JWT", "DSN",
    "URI", "CONN", "DATABASE", "REDIS", "MONGO", "POSTGRES", "MYSQL",
    "AWS", "GCP", "AZURE", "S3", "GITHUB", "SLACK", "STRIPE",
})

# System/shell vars that are not useful for diagnosis
_ENV_SKIP = frozenset({
    "PATH", "HOME", "SHELL", "TERM", "LANG", "LC_ALL", "LC_CTYPE",
    "LC_MESSAGES", "TMPDIR", "TMP", "TEMP", "PWD", "OLDPWD", "LOGNAME",
    "DISPLAY", "COLORTERM", "TERM_PROGRAM", "COMMAND_MODE", "SHLVL", "_",
    "LS_COLORS", "MANPATH", "INFOPATH", "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS", "XDG_SESSION_TYPE", "EDITOR", "VISUAL",
    "PAGER", "LESS", "GREP_OPTIONS",
})


def get_env_var_names() -> list[str]:
    """
    Return a sorted list of env var NAMES that look like configuration.
    VALUES are never included — this is purely for context awareness.
    """
    relevant: list[str] = []
    for name in os.environ:
        if name in _ENV_SKIP:
            continue
        if any(kw in name.upper() for kw in _ENV_KEYWORDS):
            relevant.append(name)
    return sorted(relevant)


def get_git_info() -> dict | None:
    """
    Return {'branch': str|None, 'dirty': bool|None} if cwd is inside a
    git repository, else None.
    """
    cwd = Path.cwd()

    # Walk up directory tree to find repo root
    repo_root: str | None = None
    for directory in [cwd, *cwd.parents]:
        if (directory / ".git").exists():
            repo_root = str(directory)
            break
    if repo_root is None:
        return None

    try:
        br = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_root,
        )
        branch: str | None = br.stdout.strip() if br.returncode == 0 else None
    except Exception:
        branch = None

    try:
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=repo_root,
        )
        dirty: bool | None = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:
        dirty = None

    return {"branch": branch, "dirty": dirty}


# Path fragments that indicate a file is NOT the user's own code
_SKIP_FRAGMENTS = (
    "site-packages",
    "dist-packages",
    "/lib/python",
    "\\lib\\python",
    "<frozen",
    "<string>",
    "<stdin>",
    "<ipython",
)


def get_source_snippets(error_info: ErrorInfo) -> list[SourceSnippet]:
    """
    Extract source code snippets (5 lines before + error line + 5 after)
    for each file referenced in the traceback that exists locally and
    belongs to the user's project (not stdlib or site-packages).

    Limits: max 3 snippets, max 11 lines each.
    """
    snippets: list[SourceSnippet] = []
    seen: set[str] = set()
    cwd = Path.cwd()

    for ref in error_info.file_references:
        if len(snippets) >= 3:
            break

        # Resolve to an existing file — try as-is, then relative to cwd
        resolved: Path | None = None
        for candidate in (Path(ref.file_path), cwd / ref.file_path):
            try:
                if candidate.exists() and candidate.is_file():
                    resolved = candidate.resolve()
                    break
            except OSError:
                continue

        if resolved is None:
            continue

        path_str = str(resolved)

        if any(frag in path_str for frag in _SKIP_FRAGMENTS):
            continue
        if path_str in seen:
            continue
        seen.add(path_str)

        try:
            all_lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            line_no = ref.line_number

            if not (1 <= line_no <= len(all_lines)):
                continue

            # 0-indexed window: 5 before + error + 5 after → up to 11 lines
            start = max(0, line_no - 6)
            end = min(len(all_lines), line_no + 5)
            context_lines = all_lines[start:end]
            error_line = all_lines[line_no - 1]

            snippets.append(SourceSnippet(
                file_path=path_str,
                line_number=line_no,
                context_lines=context_lines,
                error_line=error_line,
            ))
        except (OSError, UnicodeDecodeError):
            continue

    return snippets


def get_docker_info() -> bool | None:
    """
    Return True if Docker daemon is reachable, False if Docker is installed
    but not running, None if Docker CLI is not on PATH.
    """
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False


# ── Main gather function ─────────────────────────────────────────────────────────

def gather_context(error_info: ErrorInfo) -> LocalContext:
    """
    Gather all local context in parallel with a 2-second total budget.

    All gatherers run concurrently. Any that fail or exceed the budget
    return their safe default silently — the diagnosis is always shown.
    """
    cwd = str(Path.cwd())

    pool = ThreadPoolExecutor(max_workers=9, thread_name_prefix="wtf-ctx")
    try:
        futures = {
            "python":       pool.submit(get_python_info),
            "packages":     pool.submit(get_installed_packages),
            "requirements": pool.submit(get_requirements),
            "pyproject":    pool.submit(get_pyproject_packages),
            "os":           pool.submit(get_os_info),
            "env":          pool.submit(get_env_var_names),
            "git":          pool.submit(get_git_info),
            "docker":       pool.submit(get_docker_info),
            "snippets":     pool.submit(get_source_snippets, error_info),
        }
        done, _ = wait(futures.values(), timeout=2.0)
    finally:
        pool.shutdown(wait=False)

    def _get(key: str, default=None):
        f = futures[key]
        if f in done:
            try:
                return f.result()
            except Exception:
                pass
        return default

    python_info = _get("python", {})
    git_info    = _get("git")

    return LocalContext(
        python_version        = python_info.get("python_version"),
        python_path           = python_info.get("python_path"),
        venv_active           = python_info.get("venv_active", False),
        venv_path             = python_info.get("venv_path"),
        installed_packages    = _get("packages", {}),
        requirements_packages = _get("requirements"),
        pyproject_packages    = _get("pyproject"),
        os_info               = _get("os", ""),
        env_var_names         = _get("env", []),
        cwd                   = cwd,
        git_branch            = git_info.get("branch") if git_info else None,
        git_dirty             = git_info.get("dirty")  if git_info else None,
        source_snippets       = _get("snippets", []),
        docker_running        = _get("docker"),
    )
