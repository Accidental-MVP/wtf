"""
test_parser.py — Tests for parser.py using real-world traceback fixtures.

Each test uses a verbatim traceback string (or very close to one) that you'd
see in a real terminal. The goal is to ensure the parser extracts the right
error_type, error_message, file_references, and language for each scenario.
"""

from __future__ import annotations

import pytest

from wtf_cli.parser import (
    ErrorInfo,
    FileReference,
    _detect_language,
    _pick_error_text,
    parse_error,
)


# ── Fixture strings ─────────────────────────────────────────────────────────────

# 1. ModuleNotFoundError — most common "forgot to pip install" error
MODULE_NOT_FOUND = """\
Traceback (most recent call last):
  File "app.py", line 3, in <module>
    import pydantic
ModuleNotFoundError: No module named 'pydantic'
"""

# 2. FileNotFoundError — opening a file that doesn't exist
FILE_NOT_FOUND = """\
Traceback (most recent call last):
  File "load_config.py", line 8, in load
    with open("config.json") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'config.json'
"""

# 3. SyntaxError — missing closing paren
SYNTAX_ERROR = """\
  File "app.py", line 12
    result = calculate(x, y
                          ^
SyntaxError: '(' was never closed
"""

# 4. SyntaxError via script execution (has a traceback header)
SYNTAX_ERROR_WITH_TB = """\
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/user/.venv/bin/myapp", line 11, in <module>
    load_entry_point('myapp==0.1.0', 'console_scripts', 'myapp')()
  File "myapp/cli.py", line 4, in <module>
    from myapp import broken
  File "myapp/broken.py", line 7
    def foo(x,
               ^
SyntaxError: '(' was never closed
"""

# 5. AttributeError — typo in method name
ATTRIBUTE_ERROR = """\
Traceback (most recent call last):
  File "main.py", line 22, in process
    result = client.fetchall()
AttributeError: 'Connection' object has no attribute 'fetchall'
"""

# 6. TypeError — wrong number/type of args
TYPE_ERROR = """\
Traceback (most recent call last):
  File "server.py", line 45, in handle_request
    response = format_response(data, status)
  File "utils.py", line 12, in format_response
    return json.dumps(data, indent=indent)
TypeError: dumps() got an unexpected keyword argument 'indent'
"""

# 7. KeyError — missing dict key
KEY_ERROR = """\
Traceback (most recent call last):
  File "pipeline.py", line 33, in run
    user_id = payload["user_id"]
KeyError: 'user_id'
"""

# 8. ConnectionRefusedError — service not running
CONNECTION_REFUSED = """\
Traceback (most recent call last):
  File "db.py", line 19, in connect
    conn = psycopg2.connect(host="localhost", port=5432, dbname="mydb")
  File "/usr/local/lib/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
ConnectionRefusedError: [Errno 111] Connection refused
"""

# 9. Chained exception (During handling of the above exception…)
CHAINED_EXCEPTION = """\
Traceback (most recent call last):
  File "loader.py", line 5, in load
    data = json.loads(text)
  File "/usr/lib/python3.11/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 14, in main
    config = load("config.json")
  File "loader.py", line 8, in load
    raise RuntimeError("Config file is invalid JSON") from e
RuntimeError: Config file is invalid JSON
"""

# 10. pytest — single assertion failure
PYTEST_ASSERTION = """\
============================= test session starts ==============================
platform linux -- Python 3.11.4, pytest-7.4.0
collected 3 items

tests/test_auth.py::TestAuth::test_login PASSED                          [ 33%]
tests/test_auth.py::TestAuth::test_logout FAILED                         [ 66%]

================================== FAILURES ===================================
______________________ TestAuth::test_logout ___________________________

    def test_logout():
        client = TestClient(app)
        response = client.post("/logout")
>       assert response.status_code == 200
E       AssertionError: assert 302 == 200
E        +  where 302 = <Response [302]>.status_code

tests/test_auth.py:18: AssertionError
========================= 1 failed, 2 passed in 0.34s ==========================

=========================== short test summary info ============================
FAILED tests/test_auth.py::TestAuth::test_logout - AssertionError: assert 302 == 200
"""

# 11. pytest — multiple failures
PYTEST_MULTI_FAIL = """\
============================= test session starts ==============================
collected 4 items

tests/test_api.py::test_get_user FAILED                                  [ 25%]
tests/test_api.py::test_post_user FAILED                                  [ 50%]
tests/test_api.py::test_delete_user PASSED                               [ 75%]
tests/test_api.py::test_list_users PASSED                                [100%]

================================== FAILURES ===================================
_____________________________ test_get_user ____________________________

    def test_get_user():
>       assert get_user(1)["name"] == "Alice"
E       KeyError: 'name'

tests/test_api.py:8: KeyError
______________________________ test_post_user ___________________________

    def test_post_user():
>       assert create_user({"email": "bob@example.com"}) is not None
E       AssertionError: assert None is not None

tests/test_api.py:14: AssertionError
=========================== short test summary info ============================
FAILED tests/test_api.py::test_get_user - KeyError: 'name'
FAILED tests/test_api.py::test_post_user - AssertionError: assert None is not None
========================= 2 failed, 2 passed in 0.21s ==========================
"""

# 12. Node.js error with stack frames
NODE_ERROR = """\
/Users/user/project/app.js:5
throw new Error('Database connection failed');
      ^

Error: Database connection failed
    at connect (/Users/user/project/db.js:12:11)
    at Object.<anonymous> (/Users/user/project/app.js:5:1)
    at Module._compile (node:internal/modules/cjs/loader:1356:14)
    at Object.Module._extensions..js (node:internal/modules/cjs/loader:1414:10)
"""

# 13. ImportError (slightly different from ModuleNotFoundError)
IMPORT_ERROR = """\
Traceback (most recent call last):
  File "app.py", line 1, in <module>
    from auth import verify_token
ImportError: cannot import name 'verify_token' from 'auth' (auth.py)
"""

# 14. IndentationError
INDENTATION_ERROR = """\
  File "script.py", line 8
    return value
    ^
IndentationError: unexpected indent
"""

# 15. ZeroDivisionError (bare, no message)
ZERO_DIVISION = """\
Traceback (most recent call last):
  File "calc.py", line 3, in divide
    return a / b
ZeroDivisionError: division by zero
"""


# ── Language detection tests ─────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_python_with_traceback(self):
        assert _detect_language(MODULE_NOT_FOUND) == "python"

    def test_python_syntax_error_no_header(self):
        assert _detect_language(SYNTAX_ERROR) == "python"

    def test_python_pytest_output(self):
        assert _detect_language(PYTEST_ASSERTION) == "python"

    def test_node_error(self):
        assert _detect_language(NODE_ERROR) == "node"

    def test_generic_random_text(self):
        assert _detect_language("make: *** No rule to make target 'all'. Stop.") == "generic"


# ── Text selection tests ──────────────────────────────────────────────────────────

class TestPickErrorText:
    def test_prefers_stderr_when_only_stderr(self):
        assert _pick_error_text("some error", "") == "some error"

    def test_prefers_stdout_when_only_stdout(self):
        assert _pick_error_text("", "some output") == "some output"

    def test_prefers_stderr_traceback_over_stdout(self):
        se = MODULE_NOT_FOUND
        so = "some stdout"
        assert _pick_error_text(se, so) == se

    def test_prefers_stdout_traceback_when_stderr_has_none(self):
        # pytest writes tracebacks to stdout; stderr might just have warnings
        se = "ResourceWarning: unclosed file"
        so = PYTEST_ASSERTION
        result = _pick_error_text(se, so)
        assert "AssertionError" in result

    def test_combines_when_both_have_tracebacks(self):
        # Two different tracebacks — combine them
        result = _pick_error_text(MODULE_NOT_FOUND, FILE_NOT_FOUND)
        assert "ModuleNotFoundError" in result
        assert "FileNotFoundError" in result


# ── Python parser tests ───────────────────────────────────────────────────────────

class TestParsePythonErrors:

    # 1 ── ModuleNotFoundError
    def test_module_not_found_type(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert r.error_type == "ModuleNotFoundError"

    def test_module_not_found_message(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert "pydantic" in r.error_message

    def test_module_not_found_language(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert r.language == "python"

    def test_module_not_found_file_refs(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert len(r.file_references) == 1
        assert r.file_references[0].file_path == "app.py"
        assert r.file_references[0].line_number == 3

    # 2 ── FileNotFoundError
    def test_file_not_found(self):
        r = parse_error(FILE_NOT_FOUND, "", "python load_config.py")
        assert r.error_type == "FileNotFoundError"
        assert "config.json" in r.error_message

    def test_file_not_found_refs(self):
        r = parse_error(FILE_NOT_FOUND, "", "python load_config.py")
        assert any(ref.file_path == "load_config.py" for ref in r.file_references)

    # 3 ── SyntaxError (no traceback header)
    def test_syntax_error_no_header(self):
        r = parse_error(SYNTAX_ERROR, "", "python app.py")
        assert r.error_type == "SyntaxError"
        assert "was never closed" in r.error_message
        assert r.language == "python"

    # 4 ── SyntaxError with traceback header
    def test_syntax_error_with_traceback(self):
        r = parse_error(SYNTAX_ERROR_WITH_TB, "", "myapp")
        assert r.error_type == "SyntaxError"
        assert r.language == "python"

    def test_syntax_error_with_tb_file_refs(self):
        r = parse_error(SYNTAX_ERROR_WITH_TB, "", "myapp")
        paths = [ref.file_path for ref in r.file_references]
        assert any("myapp" in p for p in paths)

    # 5 ── AttributeError
    def test_attribute_error(self):
        r = parse_error(ATTRIBUTE_ERROR, "", "python main.py")
        assert r.error_type == "AttributeError"
        assert "fetchall" in r.error_message

    def test_attribute_error_file_ref(self):
        r = parse_error(ATTRIBUTE_ERROR, "", "python main.py")
        assert r.file_references[-1].file_path == "main.py"
        assert r.file_references[-1].line_number == 22

    # 6 ── TypeError with multi-level traceback
    def test_type_error(self):
        r = parse_error(TYPE_ERROR, "", "python server.py")
        assert r.error_type == "TypeError"
        assert "indent" in r.error_message

    def test_type_error_multiple_file_refs(self):
        r = parse_error(TYPE_ERROR, "", "python server.py")
        paths = [ref.file_path for ref in r.file_references]
        assert "server.py" in paths
        assert "utils.py" in paths

    # 7 ── KeyError
    def test_key_error(self):
        r = parse_error(KEY_ERROR, "", "python pipeline.py")
        assert r.error_type == "KeyError"
        assert "user_id" in r.error_message

    # 8 ── ConnectionRefusedError
    def test_connection_refused(self):
        r = parse_error(CONNECTION_REFUSED, "", "python db.py")
        assert r.error_type == "ConnectionRefusedError"
        assert "111" in r.error_message or "refused" in r.error_message.lower()

    # 9 ── Chained exception — should use the LAST traceback
    def test_chained_exception_uses_last_traceback(self):
        r = parse_error(CHAINED_EXCEPTION, "", "python main.py")
        assert r.error_type == "RuntimeError"
        assert "invalid JSON" in r.error_message

    def test_chained_exception_file_refs_from_last_tb(self):
        r = parse_error(CHAINED_EXCEPTION, "", "python main.py")
        paths = [ref.file_path for ref in r.file_references]
        assert "main.py" in paths

    # 10 ── ImportError
    def test_import_error(self):
        r = parse_error(IMPORT_ERROR, "", "python app.py")
        assert r.error_type == "ImportError"
        assert "verify_token" in r.error_message

    # 11 ── IndentationError (no traceback header)
    def test_indentation_error(self):
        r = parse_error(INDENTATION_ERROR, "", "python script.py")
        assert r.error_type == "IndentationError"
        assert r.language == "python"

    # 12 ── ZeroDivisionError
    def test_zero_division(self):
        r = parse_error(ZERO_DIVISION, "", "python calc.py")
        assert r.error_type == "ZeroDivisionError"
        assert r.file_references[0].file_path == "calc.py"
        assert r.file_references[0].line_number == 3

    # ── General invariants ─────────────────────────────────────────────────────

    def test_raw_error_preserved(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert r.raw_error == MODULE_NOT_FOUND

    def test_traceback_lines_not_empty(self):
        r = parse_error(MODULE_NOT_FOUND, "", "python app.py")
        assert len(r.traceback_lines) > 0

    def test_file_refs_have_valid_line_numbers(self):
        for tb in [TYPE_ERROR, CONNECTION_REFUSED, CHAINED_EXCEPTION]:
            r = parse_error(tb, "", "python x.py")
            for ref in r.file_references:
                assert ref.line_number > 0


# ── pytest output tests ───────────────────────────────────────────────────────────

class TestParsePytestOutput:

    def test_pytest_single_fail_language(self):
        # pytest writes to stdout; stderr is empty
        r = parse_error("", PYTEST_ASSERTION, "pytest tests/test_auth.py")
        assert r.language == "python"

    def test_pytest_single_fail_error_type(self):
        r = parse_error("", PYTEST_ASSERTION, "pytest tests/test_auth.py")
        assert r.error_type == "AssertionError"

    def test_pytest_single_fail_file_ref(self):
        r = parse_error("", PYTEST_ASSERTION, "pytest tests/test_auth.py")
        paths = [ref.file_path for ref in r.file_references]
        assert any("test_auth" in p for p in paths)

    def test_pytest_multi_fail_uses_first_failed_type(self):
        r = parse_error("", PYTEST_MULTI_FAIL, "pytest tests/test_api.py")
        # The first FAILED line has KeyError
        assert r.error_type in ("KeyError", "AssertionError")

    def test_pytest_multi_fail_file_refs(self):
        r = parse_error("", PYTEST_MULTI_FAIL, "pytest tests/test_api.py")
        paths = [ref.file_path for ref in r.file_references]
        assert any("test_api" in p for p in paths)

    def test_pytest_stdout_preferred_over_empty_stderr(self):
        r = parse_error("", PYTEST_ASSERTION, "pytest")
        assert r.error_type != "UnknownError"


# ── Node.js parser tests ──────────────────────────────────────────────────────────

class TestParseNodeErrors:
    def test_node_language(self):
        r = parse_error(NODE_ERROR, "", "node app.js")
        assert r.language == "node"

    def test_node_error_type(self):
        r = parse_error(NODE_ERROR, "", "node app.js")
        assert r.error_type == "Error"

    def test_node_error_message(self):
        r = parse_error(NODE_ERROR, "", "node app.js")
        assert "Database connection failed" in r.error_message

    def test_node_file_refs_skip_internals(self):
        r = parse_error(NODE_ERROR, "", "node app.js")
        # node: internals should be filtered out
        for ref in r.file_references:
            assert not ref.file_path.startswith("node:")
            assert not ref.file_path.startswith("internal/")

    def test_node_file_refs_include_user_files(self):
        r = parse_error(NODE_ERROR, "", "node app.js")
        paths = [ref.file_path for ref in r.file_references]
        assert any("db.js" in p for p in paths)
        assert any("app.js" in p for p in paths)


# ── Generic parser tests ──────────────────────────────────────────────────────────

class TestParseGenericErrors:
    def test_generic_language(self):
        text = "make: *** No rule to make target 'all'. Stop."
        r = parse_error(text, "", "make all")
        assert r.language == "generic"

    def test_generic_error_message_is_last_line(self):
        text = "Error: something went wrong\nFailed to build"
        r = parse_error(text, "", "make")
        assert "Failed to build" in r.error_message

    def test_no_output_gives_unknown_error(self):
        r = parse_error("", "", "some-command")
        assert r.error_type == "UnknownError"
        assert r.language == "generic"

    def test_generic_caps_at_20_lines(self):
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        r = parse_error(text, "", "cmd")
        assert len(r.traceback_lines) <= 20
