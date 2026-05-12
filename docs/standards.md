# Repository Engineering Standards

## Folder Organization
- Use lowercase directory names for consistency (`round1`, `round2`, `docs`).
- Keep data files colocated with the scripts that consume them when they are round-specific.
- Keep architecture/process documentation under `docs/` and module-specific docs under `docs/modules/`.

## Naming Conventions
- Python files: `snake_case.py`
- Variables/functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Markdown files: concise descriptive lowercase names

## Code Formatting and Structure
- Prefer explicit module docstrings and clear function-level responsibilities.
- Keep validation checks and constraints close to the logic they protect.
- Use deterministic seeds where stochastic experiments are compared.

## Developer Technical Notes
- Always run `make validate` after code changes.
- Use `round2/historical_simulator.py --self-test` as the baseline simulator integrity check.
- Preserve reproducibility by recording parameter values and random seeds in experiment output and docs.
