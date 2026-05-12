# System Architecture

## High-Level System Architecture
```mermaid
flowchart TD
  A[Round 1 Data: race_data.csv] --> B[Round 1 Strategy Engine]
  B --> C[Round 1 Reports]

  D[Round 2 Data: auction_history.csv] --> E[Historical Replay Simulator]
  E --> F[Iteration Experiments]
  E --> G[Robust Strategy Search]
  G --> H[Advanced Math Experiments]
  F --> I[Submission Tables]
  G --> I
  H --> I

  I --> J[Recommended Strategy Documentation]
  J --> K[Competition Submission]
```

## Data-Flow Diagram
```mermaid
flowchart LR
  R1CSV[race_data.csv] --> R1S[betting_strategy.py]
  R1S --> R1OUT[Bet allocation output]

  R2CSV[auction_history.csv] --> HS[historical_simulator.py]
  HS --> FM[Fill-model estimation]
  FM --> IT[iteration_experiments.py]
  FM --> RS[robust_strategy_search.py]
  RS --> AM[advanced_math_experiments.py]
  IT --> TBL[21-point bid/ask tables]
  RS --> TBL
  AM --> TBL
```

## Architectural Notes
- Round 1 is a deterministic EV/Kelly workflow.
- Round 2 is a layered simulation-and-optimization workflow:
  - deterministic historical replay baseline
  - robustness-focused synthetic scenario optimization
  - advanced math and risk diagnostics
- Data files are colocated with scripts for portable, script-relative execution.
