# Repository Engineering Standards

## Folder Organization
- Use lowercase directory names for consistency (`round1`, `round2`, `docs`, `docs/images`).
- Keep data files colocated with the scripts that consume them when they are round-specific.
- Keep architecture/process documentation under `docs/` and module-specific docs under `docs/modules/`.
- Store visual documentation (diagrams, images) under `docs/images/` with subdirectories by type: `architecture/`, `workflows/`, `round1_round2/`.

## Naming Conventions
- Python files: `snake_case.py`
- Variables/functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Markdown files: concise descriptive lowercase names
- Image files: lowercase with underscores, e.g., `system_architecture.png`, `dev_workflow.png`

## Code Formatting and Structure
- Prefer explicit module docstrings and clear function-level responsibilities.
- Keep validation checks and constraints close to the logic they protect.
- Use deterministic seeds where stochastic experiments are compared.

## Visual Documentation Standards
- Store images in `docs/images/` organized by type (architecture, workflows, round-specific content).
- Use PNG format for diagrams and vector-based visualizations.
- Reference images in markdown files alongside code-based diagrams (e.g., Mermaid).
- Keep Mermaid diagram source code and PNG versions synchronized where both exist.
- Include descriptive alt text when embedding images.
- Document image content and updates in `docs/images/README.md`.

## Developer Technical Notes
- Always run `make validate` after code changes.
- Use `round2/historical_simulator.py --self-test` as the baseline simulator integrity check.
- Preserve reproducibility by recording parameter values and random seeds in experiment output and docs.
- When updating architecture or workflow documentation, update both Mermaid diagrams and corresponding image files.
