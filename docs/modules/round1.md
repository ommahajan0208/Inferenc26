# Round 1 Module Documentation

## Purpose
Round 1 implements probability-based horse-race betting allocation using expected-value filtering and Kelly sizing.

## Modules
- `main.py`: thin entrypoint for running the standard Round 1 strategy.
- `betting_strategy.py`: computes true win probabilities from historical winners, compares against market probabilities, selects positive-EV horses, and sizes bets with Kelly.
- `deep_analysis.py`: deep statistical analysis workflow with distribution fitting, Monte Carlo simulations, bootstrap confidence intervals, and sensitivity checks.

## Inputs
- `race_data.csv`

## Outputs
- Console strategy table with win probabilities, EV edge, Kelly fraction, and stake allocation.
- Extended statistical report from deep analysis.

## Key Design Notes
- Data access is script-relative (`Path(__file__).with_name(...)`) for portability.
- Kelly allocation is normalized if total fraction exceeds portfolio capacity.
