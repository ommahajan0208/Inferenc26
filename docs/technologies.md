# Technologies, Libraries, Tools, Techniques, and References

## Languages
- Python 3

## Core Libraries
- `numpy`: vectorized numerical computation and simulation
- `pandas`: tabular data loading and transformation
- `scipy` (`stats`, `qmc`): statistical modeling, distributions, and quasi-random sampling

## Tools and Workflow Support
- `make`: common setup and validation command entrypoints
- `py_compile`: baseline static compile validation for Python scripts

## Techniques Used
- Expected value estimation
- Kelly criterion sizing
- Monte Carlo simulation
- Bootstrap confidence intervals
- Even/odd holdout validation
- Coordinate-ascent optimization on submission tables
- Piecewise and polynomial parameterizations (Chebyshev/Legendre)
- Weighted robust objective optimization
- Minimax and CVaR-style risk objectives
- Adverse-selection diagnostics and fill-model estimation
- Information-theoretic diagnostics (mutual information, KL)

## Referenced Research Concepts and Market Microstructure Foundations
- Kelly criterion
- Glosten-Milgrom model
- Avellaneda-Stoikov-style spread control concepts
- Risk-sensitive optimization (minimax/CVaR families)

## Data Assets
- `round1/race_data.csv`
- `round2/auction_history.csv`
