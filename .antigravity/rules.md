# 🛸 Antigravity Directives (v1.0)

## STRICT: No Safe Methods - Fail Fast

**CRITICAL**: Never use safe methods (`dict.get()`, `hasattr()`, `getattr()`, conditional checks like `if "key" in dict`) unless you have encountered a SPECIFIC runtime error that requires it.

### Rules:
1. **Always use direct access**: `dict[key]`, `obj.attr` - let it raise KeyError/AttributeError if missing
2. **Never use**: `dict.get(key, default)`, `hasattr(obj, attr)`, `getattr(obj, attr, default)`, `if "key" in dict`
3. **Exception**: Only add safe methods AFTER encountering a specific runtime error, with this exact comment format:
   #SafeMethodExc: <specific error message> - <library/API name> - <date or version>
   value = dict.get(key)  # Only if KeyError occurred at runtime
4. **For external libraries with polymorphism**: Test once at the start, then cast/assume. Or just assume and let it fail - we'll fix when it crashes.
5. **If data comes from DB**: It will be present (might be empty/null, but key exists). Use direct access.
6. **If data comes from external APIs**: Use direct access. If it crashes, we'll add proper error handling or fix the API contract.

### Philosophy:
- **Crashing is GOOD** - it tells us immediately something is wrong
- **Silent failures are BAD** - they hide bugs and produce wrong data
- **Wrong data is worse than crashes** - crashes are fixable, wrong data corrupts everything downstream

## Coding style

- Use snake_case for code and files (e.g., `example_file.py`).
- Install latest version of packages instead of versions from your memory using `uv add <package_name>`.
- We run everything with `uv run python -m ...`. YOU MUST NEVER USE `python -m ...`.
- Use def for pure functions and async def for asynchronous operations.
- All functions that have LLM API requests downstream of them should be async.
- Use type hints for all function signatures. We use dict, |, list, tuple, | None instead of typing.Dict, typing.Union, typing.List, typing.Tuple, typing.Optional.
- Prefer Pydantic models over raw dictionaries or dataclasses for input validation, unless the code is already using certain dictionaries or dataclasses.
  - The `dict` method is deprecated; use `model_dump` instead. Deprecated in Pydantic V2.0 to be removed in V3.0.
-  Do not pass configuration through many functions. If value is configurable globally, collect it from config at place of use, if it needs to be passed as argument, you might also keep it mandatory.  Default parameters for function attributes are source of hidden bugs.

## Project principles, LLM use

- All core code resides in the `src/` directory.
- All data resides in `data/`. This might contain `output` for all output, and `input`. Example data can be in `example_input` (to be provided by me).
- All tests reside in the `tests/` directory.
- For any complex functionality in `src/`, implement tests in `tests/`. Check existing tests to see if it fits in some existing test file, otherwise create a new one. Prefer integration tests using real data over unit tests. If you have to unite test, do not use mocks. Tests using real data should not be expensive to run (<$0.1 for LLM costs) unless approved.
- Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`).
- Prefer iteration and modularization over code duplication.
- Generate only code that is used, remove code that during edits becomes unreachable (unreachable from `src` folder, reachable from tests is useless).
- Import at the top of the file.
- Whenever you generate code that calls into library APIs, use Context7 MCP to fetch latest documentation for the specific version of package. If available, infer library version from pyproject.toml / package.json. Otherwise, default to latest stable
- For single-file experiments, you can hardcode values as global variables. For complex problems, use a centralized `config.yaml`.
- For complex code, use logging. Generate sufficient debug logs. For debug logs, never shorten the data - keep it full.
- Always catch as specific exceptions as possible. If you absolutely have to catch generic `Exception`, show the traceback.