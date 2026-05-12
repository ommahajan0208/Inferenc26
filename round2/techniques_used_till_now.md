# Techniques Used Till Now

This document summarizes the full strategy research path used for Round 2 so far. It is meant to be a clean reference for what was tried, why it was tried, and what each technique taught us.

## 1. Historical Replay Simulator

We started with `historical_simulator.py`, which replays the 1,000 historical rounds against the five fixed house bots.

Core idea:

```text
edge_ask = min(bot_asks) - our_ask
edge_bid = our_bid - max(bot_bids)
```

Positive edge means our quote is tighter than the best house bot quote.

The simulator evaluates expected PnL as:

```text
PnL_round =
  fill_count_ask * (our_ask - V)
+ fill_count_bid * (V - our_bid)
```

This gave us a deterministic way to score candidate formulas and 21-point submission tables.

## 2. Historical Fill Model

The simulator fits a side-specific fill model from the reference booth data.

Fills are binned by:

- quote edge size
- whether the trade was adverse

The edge bins are:

```text
[0,5), [5,10), [10,20), [20,40), [40,80), [80, infinity)
```

Important finding:

Adverse trades generate more fills than non-adverse trades. Counters are more aggressive when the quote is bad for us, so quoting too tightly increases fill count exactly when fill quality is worst.

## 3. Bayesian Posterior Centering

The signal model is:

```text
s = V + epsilon
V ~ Uniform[0,1000]
epsilon ~ N(0, 50^2)
```

The theoretically correct posterior center is the truncated-normal posterior mean:

```text
E[V | s] = s + sigma * (phi(a) - phi(b)) / (Phi(b) - Phi(a))

a = (0 - s) / sigma
b = (1000 - s) / sigma
```

This matters near the boundaries:

| Signal | Posterior Mean | Posterior - Signal |
|---:|---:|---:|
| 0 | 39.89 | 39.89 |
| 50 | 64.38 | 14.38 |
| 100 | 102.76 | 2.76 |
| 900 | 897.24 | -2.76 |
| 950 | 935.62 | -14.38 |
| 1000 | 960.11 | -39.89 |

The final robust searches used this posterior mean as the base center instead of raw signal.

## 4. Bot Reverse Engineering

Each bot was modeled by regressing quotes against the true value:

```text
bid_i = alpha_i * V + beta_i + residual
ask_i = alpha_i * V + gamma_i + residual
```

The bot mids are close to true value, with slopes near 1 and residual standard deviations around 50.

Key competitive facts:

- `tight` is most often the best ask, but only around 36% of rounds.
- `noisy` wins many best-ask rounds because its randomness occasionally quotes very tight.
- `skewed` is especially competitive on the bid side.
- The field leaves exploitable gaps, but those gaps are unstable under fresh signals.

## 5. Formula Grid Search

We tested formula quotes of the form:

```text
mid = alpha * s + intercept
bid = mid - h
ask = mid + h
```

The original best historical replay formula was around:

```text
alpha = 0.955
intercept = 22.5
h = 100
```

This improved over the naive Bayesian formula because it tracks the bot field more closely and reduces directional bias.

## 6. Raw 21-Point Coordinate Ascent

We then optimized the 21 bid values and 21 ask values directly.

Starting point:

```text
mid = 0.955 * s + 22.5
h = 100
```

Coordinate ascent tried step sizes:

```text
50, 25, 15, 10, 5, 3, 2, 1
```

Result:

The historical replay score increased to about `73.71` PnL per round, nearly double the best symmetric formula.

Main lesson:

Free-form coordinate ascent found historical gaps, especially around low signals and selected mid/high signal areas. However, some of this edge may come from fitting bin boundaries and historical quirks.

## 7. Even/Odd Holdout Validation

To check overfitting, we split the 1,000 historical rounds into even and odd periods.

Tests:

- train fill model on even, evaluate on odd
- train fill model on odd, evaluate on even
- compare formula, published coordinate table, and newly optimized half-sample tables

Finding:

The published coordinate table held up better than expected, but re-optimizing raw coordinates on only 500 rounds overfit badly.

Example:

```text
raw opt on even: train 79.25, test 32.04
raw opt on odd:  train 73.72, test 45.92
```

Conclusion:

Raw table optimization is powerful but dangerous. It needs robustness checks.

## 8. Annealed Smaller-Step Optimization

We tried coordinate ascent with smaller starting steps:

```text
10, 5, 3, 2, 1
```

This reduced some jumpy bin-boundary behavior but did not beat the published coordinate table on historical replay.

Main lesson:

Smaller steps alone are not enough. The objective itself needs to be more robust.

## 9. Smooth Piecewise Spread Parameterization

Instead of optimizing all 42 raw table values, we constrained the shape:

```text
bid(s) = mid(s) - h_bid(s)
ask(s) = mid(s) + h_ask(s)
```

where `h_bid(s)` and `h_ask(s)` are piecewise-linear functions.

This reduces noise fitting and gives the table a more stable structure.

Main lesson:

Smooth tables score lower on pure historical replay but are more plausible for fresh final scoring.

## 10. Fresh-Round Bot Simulator

Historical replay reuses the exact historical bot quotes. Final scoring will not.

So we built a fresh-round simulator in `robust_strategy_search.py`.

For each bot:

```text
mid = alpha * V + beta + bootstrapped_mid_residual
spread = mean_spread + bootstrapped_spread_residual
bid = mid - spread / 2
ask = mid + spread / 2
```

Then we simulate:

```text
V ~ Uniform[0,1000]
our_signal = V + N(0, 50^2)
bot quotes from fitted bot models
```

Validation:

The simulated bots reproduce historical best-bid and best-ask shares within about 2 percentage points.

## 11. Participant Crowding Stress Tests

Final scoring includes other participant submissions. Since we do not know them, we modeled crowding by adding clone competitors:

- no participant clones
- 1 coordinate-table clone
- 3 coordinate-table clones
- 5 coordinate-table clones
- 3 fresh-formula clones
- mixed coordinate/fresh-formula clone field

This changed the strategic conclusion:

- Historical replay favors the raw coordinate-ascent table.
- Fresh scoring plus crowding favors smoother posterior-centered tables.

## 12. Robust Weighted Objective

We optimized a weighted scenario objective:

```text
objective = weighted_mean_pnl - 0.5 * scenario_pnl_std
```

Scenario weights:

| Scenario | Weight |
|---|---:|
| no clones | 0.35 |
| 1 coord clone | 0.15 |
| 3 coord clones | 0.20 |
| 5 coord clones | 0.10 |
| 3 fresh-formula clones | 0.10 |
| mixed coord/fresh field | 0.10 |

This balances aggressiveness and robustness.

## 13. Minimax and CVaR Objectives

In `advanced_math_experiments.py`, we also tried stricter robust objectives:

```text
minimax = min_scenario PnL
cvar_bottom3 = average PnL of the three worst scenarios
```

Finding:

Minimax and CVaR reduce downside risk but give up too much weighted expected value. The weighted-risk objective produced the best final recommendation.

## 14. Glosten-Milgrom Adverse-Selection Proxy

We estimated a Glosten-Milgrom-style spread requirement:

```text
h >= lambda * E[|V - mid| | trade occurs]
```

Historical proxy estimates:

| Side | Adverse Fill Fraction | Mean Abs Error | Half-Spread Proxy |
|---|---:|---:|---:|
| ask | 54.6% | 74.68 | 40.81 |
| bid | 66.6% | 82.44 | 54.87 |

Main lesson:

The bid side is more adverse in the historical reference data. Robust strategies therefore avoid over-aggressive bid-side tightening.

## 15. Fill-Rate x Edge Decomposition

We decomposed PnL into:

```text
E[PnL] =
  P(ask_win) * fill_rate_ask * E[ask - V | ask_win]
+ P(bid_win) * fill_rate_bid * E[V - bid | bid_win]
```

This helped separate three levers:

- win probability
- expected fills conditional on winning
- average edge per fill

Main lesson:

The robust tables win mostly by improving ask-side edge while keeping enough fill rate. They do not simply maximize fills.

## 16. Asymmetric Spread Skewing

The final robust tables allow separate bid and ask spreads:

```text
h_bid(s) != h_ask(s)
```

This matters because risk is not symmetric:

- near low signals, posterior mean is above raw signal
- near high signals, posterior mean is below raw signal
- adverse-selection pressure differs by side
- competitor gaps differ by side

The final table is therefore not a simple symmetric spread around signal.

## 17. Deep Math Expansion

The latest `advanced_math_experiments.py` pass added concrete versions of the deeper techniques:

- information theory: mutual information, posterior KL, and rate-distortion knot diagnostics
- statistical estimation: Huber bot regressions, KDE local best response, isotonic shape constraints, conformal competitor intervals
- stochastic control: Avellaneda-Stoikov-style spread candidate
- optimization: distributionally robust, chance-constrained, spectral-risk, minimax, and CVaR objectives
- functional approximation: Chebyshev and Legendre low-degree table parameterizations, plus a GP-smoothed variant
- microstructure/risk: PIN, Kyle lambda, Roll spread, Amihud, EVT tail, CVaR, ruin, and LIL diagnostics
- simulation: antithetic and Halton variance-reduction checks

Important diagnostics:

| Diagnostic | Result |
|---|---:|
| Mutual information I(V;S) | 2.41 bits |
| Mean 21-bucket posterior std | 46.81 |
| KL at center signals | about 2.27 bits |
| KL at boundary signals | about 3.27 bits |
| PIN proxy, all fills | 59% |
| Kyle lambda proxy | 3.96 |
| Roll spread proxy | 24.08 |

The high-leverage result was not KDE or Avellaneda-Stoikov. Those were too conservative or too tied to historical quirks. The winner was a low-degree polynomial table: smooth enough to avoid raw coordinate overfit, but expressive enough to keep the useful boundary skew. The latest paired run favored Legendre degree-4 over the Chebyshev variants.

## 18. Current Strategy Conclusions

There are now three useful reference points:

| Objective | Best Table |
|---|---|
| maximize historical replay | published coordinate-ascent table |
| maximize robust fresh-round score | Legendre degree-4 table |
| most conservative stress-table | minimax / CVaR variants |

The current recommendation is the Legendre degree-4 table. It beat Legendre degree-6 and the best Chebyshev variant in the latest paired check:

```text
poly_legendre_deg4  objective 21.96 +/- 0.09
poly_legendre_deg6  objective 21.94 +/- 0.08
poly_chebyshev_deg5 objective 21.91 +/- 0.08

legendre_deg4 - legendre_deg6: +0.02 (same 20x100k run)
```

This was measured in a targeted 20 paired-seed check x 100,000 simulated rounds per seed.

## 19. Current Recommended Final Table

| Signal | Bid | Ask |
|---:|---:|---:|
| 0 | 6.19 | 90.72 |
| 50 | 6.19 | 137.96 |
| 100 | 15.42 | 190.95 |
| 150 | 47.16 | 246.64 |
| 200 | 88.02 | 300.09 |
| 250 | 134.06 | 350.74 |
| 300 | 183.49 | 399.69 |
| 350 | 234.99 | 447.99 |
| 400 | 287.47 | 496.44 |
| 450 | 340.11 | 545.61 |
| 500 | 392.36 | 595.79 |
| 550 | 443.89 | 647.05 |
| 600 | 494.65 | 699.18 |
| 650 | 544.83 | 751.75 |
| 700 | 594.88 | 804.06 |
| 750 | 645.50 | 855.18 |
| 800 | 697.63 | 903.89 |
| 850 | 752.29 | 948.55 |
| 900 | 808.81 | 985.36 |
| 950 | 862.16 | 1003.96 |
| 1000 | 908.90 | 1003.96 |

Copy-paste rows:

```text
signal	bid	ask
0	6.19	90.72
50	6.19	137.96
100	15.42	190.95
150	47.16	246.64
200	88.02	300.09
250	134.06	350.74
300	183.49	399.69
350	234.99	447.99
400	287.47	496.44
450	340.11	545.61
500	392.36	595.79
550	443.89	647.05
600	494.65	699.18
650	544.83	751.75
700	594.88	804.06
750	645.50	855.18
800	697.63	903.89
850	752.29	948.55
900	808.81	985.36
950	862.16	1003.96
1000	908.90	1003.96
```

## 20. KKT-Style Gradient Check (Finite Differences)

We added a finite-difference gradient audit on the weighted PnL objective to test local optimality.

For each bid/ask at each signal, we compute:

```text
grad_i ~= (PnL(q_i + eps) - PnL(q_i - eps)) / (2 * eps)
```

We also tag each coordinate as:

- interior
- lower_bound (0)
- upper_bound (1200)
- bid_ask_bound (bid <= ask constraint)

If |grad| remains large on interior points, coordinate ascent likely stopped early.

## 21. Bid-at-Zero Sweep

Because the posterior mean at signal 0 is near 40, we explicitly swept:

```text
bid(0) in {base, 35, 36, 38}
```

This isolates the single highest-leverage boundary adjustment without changing the rest of the table.

## 22. Polynomial Degree Sweep (Legendre 5/6)

We extended the polynomial parameterization to degrees 5 and 6 for Legendre and Chebyshev.

This tests whether added expressiveness captures boundary skew without overfitting.

## 23. Random Opponent Field Median-Rank Test

We now sample 50 random opponent fields (random mid/tilt/spread tables) and compute:

```text
median rank of each candidate table across fields
```

This is a robustness metric that is less sensitive to a single weighted objective.

## 24. Multi-Start Coordinate Ascent with CV Selection

We run 5 random starting tables (random alpha/intercept/spread + jitter),
optimize each with coordinate ascent, then select the best by even/odd CV score.

This is the simplest way to avoid local-optimum traps in the raw 21-point search.

## 25. Validation Commands Used

```text
python historical_simulator.py --self-test
python -m py_compile historical_simulator.py robust_strategy_search.py advanced_math_experiments.py iteration_experiments.py
python advanced_math_experiments.py --search-n 30000 --eval-n 60000 --eval-seeds 8
python advanced_math_experiments.py --run-next-steps --eval-seeds 20 --eval-n 100000
```

## 26. Main Remaining Risk

The biggest remaining uncertainty is participant field composition.

The Legendre degree-4 table is better on the modeled fresh fields and clone scenarios, but the edge over Legendre degree-6 and the best Chebyshev variant is modest. If the real participant field is much more crowded around smooth polynomial strategies than around the older robust table, the older table may still be competitive.

## 27. Boundary Posterior Decoupling + Micro-Undercutting (Attempt)

We decoupled boundary pricing from the central polynomial by resetting the boundary midpoints to the exact truncated-normal posterior mean for signals below 150 and above 850. We tested both boundary definitions:

- strict: s < 150 and s > 850
- inclusive: s <= 150 and s >= 850

After the boundary reset, we applied deterministic micro-undercutting against a consensus opponent surface. The consensus surface was estimated as a weighted median of best quotes by signal bucket (Gaussian bandwidth 60), then we nudged bids up and asks down by 1 tick.

Outcome:

- Both boundary variants collapsed in the robust simulator (objective about -533.00 ± 0.51).
- Win rates exceeded 50%, but edge per fill turned sharply negative, indicating the undercut forced quotes too tight under adverse selection.

This approach is not competitive as implemented; the Legendre degree-4 baseline remains the best candidate.

