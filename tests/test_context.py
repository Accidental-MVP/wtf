"""
test_context.py — Tests for context.py.

Each gatherer is tested independently by mocking subprocess calls, os.environ,
and the filesystem so tests are fast and deterministic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wtf_cli.context import (
    LocalContext,
    SourceSnippet,
    _parse_pyproject_deps,
    _parse_requirements_txt,
    gather_context,
    get_docker_info,
    get_env_var_names,
    get_git_info,
    get_installed_packages,
    get_os_info,
    get_python_info,
    get_requirements,
    get_source_snippets,
)
from wtf_cli.parser import ErrorInfo, FileReference


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _make_error_info(**kwargs) -> ErrorInfo:
    defaults = dict(
        error_type="ModuleNotFoundError",
        error_message="No module named 'foo'",
        traceback_lines=[],
        file_references=[],
        language="python",
        raw_error="",
    )
    defaults.update(kwargs)
    return ErrorInfo(**defaults)


PIP_LIST_JSON = json.dumps([
    {"name": "requests", "version": "2.31.0"},
    {"name": "Flask", "version": "3.0.0"},
    {"name": "pydantic", "version": "2.5.0"},
])


# ── get_python_info ───────────────────────────────────────────────────────────────

class TestGetPythonInfo:
    def test_returns_dict_with_required_keys(self):
        info = get_python_info()
        assert "python_version" in info
        assert "python_path" in info
        assert "venv_active" in info
        assert "venv_path" in info

    def test_version_format(self):
        info = get_python_info()
        parts = info["python_version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_python_path_is_string(self):
        info = get_python_info()
        assert isinstance(info["python_path"], str)
        assert len(info["python_path"]) > 0

    def test_detects_active_venv_from_env_var(self):
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/home/user/.venv"}, clear=False):
            info = get_python_info()
        assert info["venv_active"] is True
        assert info["venv_path"] == "/home/user/.venv"

    def test_detects_active_conda_env(self):
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["CONDA_PREFIX"] = "/opt/conda/envs/myenv"
        with patch.dict(os.environ, env, clear=True):
            info = get_python_info()
        assert info["venv_active"] is True
        assert info["venv_path"] == "/opt/conda/envs/myenv"

    def test_detects_local_venv_directory(self, tmp_path: Path):
        """If .venv/ exists locally but isn't activated, venv_path should point to it."""
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        py_name = "python.exe" if sys.platform == "win32" else "python"
        venv_bin = tmp_path / ".venv" / bin_dir
        venv_bin.mkdir(parents=True)
        (venv_bin / py_name).touch()

        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
        with patch.dict(os.environ, env, clear=True):
            with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
                info = get_python_info()

        assert info["venv_active"] is False
        assert info["venv_path"] is not None
        assert ".venv" in info["venv_path"]

    def test_no_venv_returns_none_path(self, tmp_path: Path):
        """When no venv is active or local, venv_path should be None."""
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
        with patch.dict(os.environ, env, clear=True):
            with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
                info = get_python_info()
        assert info["venv_active"] is False
        assert info["venv_path"] is None


# ── get_installed_packages ────────────────────────────────────────────────────────

class TestGetInstalledPackages:
    def _mock_run(self, stdout: str, returncode: int = 0):
        mock = MagicMock()
        mock.returncode = returncode
        mock.stdout = stdout
        return mock

    def test_parses_pip_list_json(self):
        with patch("subprocess.run", return_value=self._mock_run(PIP_LIST_JSON)):
            packages = get_installed_packages()
        assert packages["requests"] == "2.31.0"
        assert packages["flask"] == "3.0.0"
        assert packages["pydantic"] == "2.5.0"

    def test_names_are_lowercased(self):
        data = json.dumps([{"name": "MyPackage", "version": "1.0"}])
        with patch("subprocess.run", return_value=self._mock_run(data)):
            packages = get_installed_packages()
        assert "mypackage" in packages

    def test_returns_empty_dict_on_subprocess_failure(self):
        with patch("subprocess.run", return_value=self._mock_run("", returncode=1)):
            packages = get_installed_packages()
        assert packages == {}

    def test_returns_empty_dict_on_exception(self):
        with patch("subprocess.run", side_effect=Exception("timeout")):
            packages = get_installed_packages()
        assert packages == {}

    def test_returns_empty_dict_on_invalid_json(self):
        with patch("subprocess.run", return_value=self._mock_run("not-json")):
            packages = get_installed_packages()
        assert packages == {}


# ── _parse_requirements_txt ───────────────────────────────────────────────────────

class TestParseRequirementsTxt:
    def test_simple_pinned(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\nflask==3.0.0\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert result["requests"] == "==2.31.0"
        assert result["flask"] == "==3.0.0"

    def test_version_range(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("pydantic>=2.0.0,<3\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "pydantic" in result

    def test_bare_package_no_version(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "numpy" in result
        assert result["numpy"] == ""

    def test_skips_comments(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("# This is a comment\nrequests==2.31.0\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "requests" in result
        assert len(result) == 1

    def test_skips_flags(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("-r base.txt\n--no-binary :all:\nrequests\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "requests" in result
        assert len(result) == 1

    def test_skips_vcs_urls(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text(
            "git+https://github.com/user/repo.git\nrequests\n"
        )
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "requests" in result
        assert len(result) == 1

    def test_handles_extras(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("pydantic[email]==2.0\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "pydantic" in result

    def test_normalizes_underscores_to_dashes(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("my_package==1.0\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "my-package" in result or "my_package" in result

    def test_strips_inline_comments(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0  # HTTP library\n")
        result = _parse_requirements_txt(tmp_path / "requirements.txt")
        assert "requests" in result


# ── _parse_pyproject_deps ─────────────────────────────────────────────────────────

class TestParsePyprojectDeps:
    def test_extracts_dependencies(self, tmp_path: Path):
        # Standard pyproject.toml format: dependencies is an array under [project]
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "myapp"\ndependencies = [\n'
            '    "requests>=2.0",\n    "flask>=3.0",\n]\n'
        )
        result = _parse_pyproject_deps(pyproject)
        assert result is not None
        assert any("requests" in d for d in result)
        assert any("flask" in d for d in result)

    def test_returns_none_when_section_absent(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[build-system]\nrequires = ["hatchling"]\n')
        result = _parse_pyproject_deps(pyproject)
        assert result is None

    def test_returns_none_on_empty_deps(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "myapp"\n')
        result = _parse_pyproject_deps(pyproject)
        assert result is None


# ── get_requirements ──────────────────────────────────────────────────────────────

class TestGetRequirements:
    def test_finds_requirements_in_cwd(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            result = get_requirements()
        assert result is not None
        assert "requests" in result

    def test_finds_requirements_in_parent(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("flask==3.0.0\n")
        subdir = tmp_path / "src" / "app"
        subdir.mkdir(parents=True)
        with patch("wtf_cli.context.Path.cwd", return_value=subdir):
            result = get_requirements()
        assert result is not None
        assert "flask" in result

    def test_returns_none_when_not_found(self, tmp_path: Path):
        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            result = get_requirements()
        assert result is None


# ── get_os_info ───────────────────────────────────────────────────────────────────

class TestGetOsInfo:
    def test_macos_format(self):
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
                with patch("platform.machine", return_value="arm64"):
                    result = get_os_info()
        assert "macOS" in result
        assert "14.2" in result
        assert "arm64" in result

    def test_windows_format(self):
        with patch("platform.system", return_value="Windows"):
            with patch("platform.release", return_value="11"):
                with patch("platform.machine", return_value="AMD64"):
                    result = get_os_info()
        assert "Windows" in result
        assert "11" in result

    def test_linux_fallback_format(self):
        with patch("platform.system", return_value="Linux"):
            with patch("platform.release", return_value="6.1.0"):
                with patch("platform.machine", return_value="x86_64"):
                    # Patch open to fail so we use platform fallback
                    with patch("builtins.open", side_effect=OSError):
                        result = get_os_info()
        assert "Linux" in result
        assert "x86_64" in result

    def test_returns_non_empty_string(self):
        result = get_os_info()
        assert isinstance(result, str)
        assert len(result) > 0


# ── get_env_var_names ─────────────────────────────────────────────────────────────

class TestGetEnvVarNames:
    def test_includes_api_key(self):
        env = {"OPENAI_API_KEY": "sk-secret", "PATH": "/usr/bin"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        assert "OPENAI_API_KEY" in names

    def test_includes_database_url(self):
        env = {"DATABASE_URL": "postgres://localhost/db", "HOME": "/home/user"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        assert "DATABASE_URL" in names

    def test_excludes_path(self):
        env = {"PATH": "/usr/bin:/usr/local/bin", "SECRET_KEY": "abc"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        assert "PATH" not in names

    def test_excludes_home(self):
        env = {"HOME": "/home/user", "API_TOKEN": "tok"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        assert "HOME" not in names

    def test_never_includes_values(self):
        env = {"SECRET_KEY": "super-secret-value", "DATABASE_URL": "postgres://secret"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        # Result is a list of names — confirm no values leaked
        assert all(v not in names for v in env.values())

    def test_returns_sorted_list(self):
        env = {"Z_API_KEY": "1", "A_SECRET": "2", "M_TOKEN": "3"}
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        relevant = [n for n in names if n in env]
        assert relevant == sorted(relevant)

    def test_includes_common_service_vars(self):
        env = {
            "REDIS_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "AWS_REGION": "us-east-1",
            "STRIPE_SECRET_KEY": "sk_test_...",
            "HOME": "/home/user",
            "SHELL": "/bin/bash",
        }
        with patch.dict(os.environ, env, clear=True):
            names = get_env_var_names()
        assert "REDIS_HOST" in names
        assert "POSTGRES_PORT" in names
        assert "AWS_REGION" in names
        assert "STRIPE_SECRET_KEY" in names
        assert "HOME" not in names
        assert "SHELL" not in names


# ── get_git_info ──────────────────────────────────────────────────────────────────

class TestGetGitInfo:
    def test_returns_none_outside_repo(self, tmp_path: Path):
        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            result = get_git_info()
        assert result is None

    def test_detects_repo_and_branch(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        branch_mock = MagicMock(returncode=0, stdout="main\n")
        status_mock = MagicMock(returncode=0, stdout="")

        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            with patch("subprocess.run", side_effect=[branch_mock, status_mock]):
                result = get_git_info()

        assert result is not None
        assert result["branch"] == "main"

    def test_clean_repo_dirty_is_false(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        branch_mock = MagicMock(returncode=0, stdout="main\n")
        status_mock = MagicMock(returncode=0, stdout="")  # no output = clean

        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            with patch("subprocess.run", side_effect=[branch_mock, status_mock]):
                result = get_git_info()

        assert result["dirty"] is False

    def test_dirty_repo(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        branch_mock = MagicMock(returncode=0, stdout="feature/auth\n")
        status_mock = MagicMock(returncode=0, stdout=" M app.py\n")

        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            with patch("subprocess.run", side_effect=[branch_mock, status_mock]):
                result = get_git_info()

        assert result["dirty"] is True

    def test_git_command_failure_returns_none_fields(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        with patch("wtf_cli.context.Path.cwd", return_value=tmp_path):
            with patch("subprocess.run", side_effect=Exception("git not found")):
                result = get_git_info()

        assert result is not None
        assert result["branch"] is None
        assert result["dirty"] is None

    def test_finds_repo_in_parent_dir(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)

        branch_mock = MagicMock(returncode=0, stdout="develop\n")
        status_mock = MagicMock(returncode=0, stdout="")

        with patch("wtf_cli.context.Path.cwd", return_value=subdir):
            with patch("subprocess.run", side_effect=[branch_mock, status_mock]):
                result = get_git_info()

        assert result is not None
        assert result["branch"] == "develop"


# ── get_source_snippets ───────────────────────────────────────────────────────────

class TestGetSourceSnippets:
    def _make_file(self, tmp_path: Path, name: str, lines: list[str]) -> Path:
        p = tmp_path / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_extracts_snippet_from_local_file(self, tmp_path: Path):
        code = [f"line_{i}" for i in range(1, 21)]
        p = self._make_file(tmp_path, "app.py", code)

        error_info = _make_error_info(
            file_references=[FileReference(file_path=str(p), line_number=10)]
        )
        snippets = get_source_snippets(error_info)

        assert len(snippets) == 1
        assert snippets[0].error_line == "line_10"
        assert snippets[0].line_number == 10

    def test_context_lines_at_most_11(self, tmp_path: Path):
        code = [f"L{i}" for i in range(1, 30)]
        p = self._make_file(tmp_path, "app.py", code)
        error_info = _make_error_info(
            file_references=[FileReference(file_path=str(p), line_number=15)]
        )
        snippets = get_source_snippets(error_info)
        assert len(snippets[0].context_lines) <= 11

    def test_skips_nonexistent_files(self):
        error_info = _make_error_info(
            file_references=[FileReference(file_path="/nonexistent/path.py", line_number=1)]
        )
        snippets = get_source_snippets(error_info)
        assert snippets == []

    def test_skips_site_packages(self, tmp_path: Path):
        # Simulate a site-packages path by patching resolution
        site_pkg = tmp_path / "site-packages" / "pydantic" / "main.py"
        site_pkg.parent.mkdir(parents=True)
        site_pkg.write_text("class BaseModel: pass\n")

        error_info = _make_error_info(
            file_references=[FileReference(file_path=str(site_pkg), line_number=1)]
        )
        snippets = get_source_snippets(error_info)
        assert snippets == []

    def test_limits_to_three_snippets(self, tmp_path: Path):
        refs = []
        for i in range(5):
            p = self._make_file(tmp_path, f"file{i}.py", [f"code_{j}" for j in range(10)])
            refs.append(FileReference(file_path=str(p), line_number=5))

        error_info = _make_error_info(file_references=refs)
        snippets = get_source_snippets(error_info)
        assert len(snippets) <= 3

    def test_deduplicates_same_file(self, tmp_path: Path):
        code = [f"L{i}" for i in range(10)]
        p = self._make_file(tmp_path, "app.py", code)
        refs = [
            FileReference(file_path=str(p), line_number=3),
            FileReference(file_path=str(p), line_number=7),
        ]
        error_info = _make_error_info(file_references=refs)
        snippets = get_source_snippets(error_info)
        # Should only include the file once
        assert len(snippets) == 1

    def test_handles_start_of_file(self, tmp_path: Path):
        code = [f"L{i}" for i in range(1, 10)]
        p = self._make_file(tmp_path, "app.py", code)
        error_info = _make_error_info(
            file_references=[FileReference(file_path=str(p), line_number=1)]
        )
        snippets = get_source_snippets(error_info)
        assert len(snippets) == 1
        assert snippets[0].error_line == "L1"

    def test_handles_end_of_file(self, tmp_path: Path):
        code = [f"L{i}" for i in range(1, 10)]
        p = self._make_file(tmp_path, "app.py", code)
        error_info = _make_error_info(
            file_references=[FileReference(file_path=str(p), line_number=9)]
        )
        snippets = get_source_snippets(error_info)
        assert len(snippets) == 1
        assert snippets[0].error_line == "L9"


# ── get_docker_info ───────────────────────────────────────────────────────────────

class TestGetDockerInfo:
    def test_returns_none_when_docker_not_installed(self):
        with patch("shutil.which", return_value=None):
            result = get_docker_info()
        assert result is None

    def test_returns_true_when_docker_running(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result):
                result = get_docker_info()
        assert result is True

    def test_returns_false_when_docker_not_running(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            mock_result = MagicMock(returncode=1)
            with patch("subprocess.run", return_value=mock_result):
                result = get_docker_info()
        assert result is False

    def test_returns_false_on_timeout(self):
        import subprocess as sp
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", side_effect=sp.TimeoutExpired("docker", 2)):
                result = get_docker_info()
        assert result is False


# ── gather_context (integration) ──────────────────────────────────────────────────

class TestGatherContext:
    def test_returns_local_context_instance(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert isinstance(ctx, LocalContext)

    def test_cwd_is_set(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert ctx.cwd != ""
        assert Path(ctx.cwd).exists()

    def test_python_version_is_populated(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert ctx.python_version is not None
        assert "." in ctx.python_version

    def test_os_info_is_populated(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert isinstance(ctx.os_info, str)
        assert len(ctx.os_info) > 0

    def test_installed_packages_is_dict(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert isinstance(ctx.installed_packages, dict)

    def test_env_var_names_is_list(self):
        error_info = _make_error_info()
        ctx = gather_context(error_info)
        assert isinstance(ctx.env_var_names, list)

    def test_completes_within_reasonable_time(self):
        import time
        error_info = _make_error_info()
        start = time.monotonic()
        gather_context(error_info)
        elapsed = time.monotonic() - start
        # Allow generous margin; the 2-second budget + overhead
        assert elapsed < 10.0

    def test_gatherer_failure_does_not_raise(self):
        """Even if every gatherer throws, gather_context should return a valid context."""
        error_info = _make_error_info()
        with patch("wtf_cli.context.get_python_info", side_effect=RuntimeError("boom")):
            with patch("wtf_cli.context.get_installed_packages", side_effect=RuntimeError):
                with patch("wtf_cli.context.get_os_info", side_effect=RuntimeError):
                    ctx = gather_context(error_info)
        # Should not raise; should return a context with defaults
        assert isinstance(ctx, LocalContext)
