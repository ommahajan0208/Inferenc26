# Inferenc26

Enterprise-style research repository for two quantitative strategy tracks:
- **Round 1**: horse-racing expected-value and Kelly sizing analysis
- **Round 2**: market-making strategy design under adverse selection and competitive quoting

## Project Overview
This repository contains reproducible analytical workflows, simulation engines, and optimization experiments used to build and validate trading-style quoting strategies across competition rounds.

## Repository Architecture Summary
```text
Inferenc26/
├── round1/                    # Round 1 strategy models and artifacts
│   ├── race_data.csv
│   ├── betting_strategy.py
│   ├── main.py
│   ├── deep_analysis.py
│   └── *.md / *.html
├── round2/                    # Round 2 simulators and robust optimization workflows
│   ├── auction_history.csv
│   ├── historical_simulator.py
│   ├── iteration_experiments.py
│   ├── robust_strategy_search.py
│   ├── advanced_math_experiments.py
│   └── *.md
├── docs/                      # Architecture, workflow, module, and tech documentation
│   ├── architecture.md
│   ├── workflows.md
│   ├── technologies.md
│   ├── standards.md
│   └── modules/
│       ├── round1.md
│       └── round2.md
├── requirements.txt
├── Makefile
└── README.md
```

## Setup Instructions
1. Create a Python 3.11+ virtual environment.
2. Install dependencies:
   ```bash
   make setup
   ```
3. Validate scripts compile:
   ```bash
   make validate
   ```

## Development Workflow
1. Start with `docs/standards.md` for naming and structure conventions.
2. Use `round2/historical_simulator.py --self-test` before changing simulation logic.
3. Keep experiments reproducible by documenting command arguments and seeds.
4. Update module docs in `docs/modules/` whenever behavior changes.

## Architecture and Workflow Diagrams
- [System architecture and data-flow diagrams](docs/architecture.md)
- [Operational workflows](docs/workflows.md)

## Technical Documentation Index
- [Round 1 module documentation](docs/modules/round1.md)
- [Round 2 module documentation](docs/modules/round2.md)
- [Technology stack, libraries, techniques, and references](docs/technologies.md)
- [Engineering standards and repository conventions](docs/standards.md)

## License
This project is licensed under the terms in [LICENSE](LICENSE).
