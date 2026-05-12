# Development and Analysis Workflows

## Standard Development Workflow
```mermaid
flowchart TD
  A[Create/update branch] --> B[Install dependencies]
  B --> C[Run make validate]
  C --> D[Run scenario scripts]
  D --> E[Update docs/modules and docs/technologies]
  E --> F[Re-run validation]
  F --> G[Commit and review]
```

## Round 2 Strategy Workflow
```mermaid
flowchart TD
  A[Load auction_history.csv] --> B[Run historical_simulator self-test]
  B --> C[Evaluate baseline formulas/tables]
  C --> D[Run robust_strategy_search]
  D --> E[Run advanced_math_experiments]
  E --> F[Compare objective and risk metrics]
  F --> G[Publish recommended 21-point table]
```

## Operational Commands
- Compile-time checks: `make validate`
- Round 1 strategy run: `make round1`
- Round 2 baseline checks: `make round2-self-test`
