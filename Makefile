.RECIPEPREFIX := >
.PHONY: setup validate round1 round2-self-test

setup:
>python -m pip install -r requirements.txt

validate:
>python -m py_compile round1/main.py round1/betting_strategy.py round1/deep_analysis.py round2/historical_simulator.py round2/robust_strategy_search.py round2/advanced_math_experiments.py round2/iteration_experiments.py

round1:
>python round1/betting_strategy.py

round2-self-test:
>python round2/historical_simulator.py --self-test
