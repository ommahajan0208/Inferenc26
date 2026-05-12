# Round 2 Module Documentation

## Purpose
Round 2 builds and evaluates bid/ask quoting strategies in a competitive market-making environment with adverse selection and participant crowding.

## Modules
- `historical_simulator.py`
  - Core simulator and evaluator.
  - Fits side-specific fill models from historical rounds.
  - Scores formula-based and table-based quoting strategies.
  - Provides self-test mode.
- `iteration_experiments.py`
  - Runs iterative strategy improvements.
  - Includes coordinate-ascent table optimization and holdout validation.
- `robust_strategy_search.py`
  - Fresh-round simulation against modeled competitor behavior.
  - Optimizes smooth, robust tables under weighted multi-scenario objectives.
- `advanced_math_experiments.py`
  - Extends robust search with additional objectives and diagnostics:
    - information-theoretic
    - statistical estimation
    - risk metrics (minimax, CVaR-like)
    - polynomial table parameterizations

## Input Data
- `auction_history.csv`

## Output Artifacts
- Strategy comparison tables
- Robust objective metrics
- Recommended 21-point submission tables

## Key Design Notes
- Scripts default to local round data via script-relative paths.
- Constraints (`0 <= bid <= ask <= 1200`) are enforced in validation flows.
