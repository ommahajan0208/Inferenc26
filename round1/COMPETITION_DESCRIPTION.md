# Dead Heat: Bid Smart, or Bid Last

## Competition Overview

**Season 0 · Round 01**

**Status**: Open | **Closes in**: 5d 13:33:31

This is a parimutuel horse racing betting competition where participants must strategically allocate a fixed betting pool across eight horses to maximize expected profit over 10,000 simulated races.

---

## The Setup

### Scenario
Eight horses are about to race. Participants have access to:
- Results from 500 historical races involving these exact horses
- A parimutuel betting market with real-time pool fractions
- A starting budget of **£10,000**

### Parimutuel Betting Mechanics

In a parimutuel betting system:

1. **Pool Management**: All money bet on a given horse is pooled together
2. **Track Takeout**: The track takes a 15% commission
3. **Payout Distribution**: Winners receive their proportional share of the remaining 85% pool

#### Payout Formula

If the total pool is $P$, the track takes $0.15P$, leaving $0.85P$.

For a horse $i$ that received fraction $f_i$ of all bets, if you bet $b$ on horse $i$:

$$\text{payout} = b \cdot \frac{0.85}{f_i}$$

$$\text{profit} = \begin{cases} 
b \cdot \frac{0.85}{f_i} - b & \text{if horse } i \text{ wins} \\
-b & \text{if horse } i \text{ loses}
\end{cases}$$

### Betting Constraints

- Total budget: **£10,000**
- May allocate any amount to any combination of horses
- **No arbing across horses** (standard parimutuel rules)
- All stakes must be non-negative
- May stake £0 on any horse

---

## Current Market

The parimutuel pool fractions remain **fixed for scoring purposes**. Your bets are assumed to be negligible and do not move the market.

| Horse | Pool Fraction | Implied Win Prob |
|-------|---------------|------------------|
| Shadowfax | 8.0% | 8.0% |
| Iron Duke | 9.0% | 9.0% |
| Morningstar | 11.0% | 11.0% |
| Red Tide | 13.0% | 13.0% |
| Gallant Fox | 14.0% | 14.0% |
| Blue Streak | 15.0% | 15.0% |
| Copper Prince | 14.0% | 14.0% |
| Last Chance | 16.0% | 16.0% |

---

## Historical Data

### Data Source
- **File**: `race_data.csv`
- **Format**: 500 historical races involving the 8 horses

### Data Columns
- `race_id`: Identifier for each race
- `Shadowfax`, `Iron Duke`, `Morningstar`, `Red Tide`, `Gallant Fox`, `Blue Streak`, `Copper Prince`, `Last Chance`: 
  - Finishing time in seconds for each horse
  - Lower time = better performance
- `winner`: The horse with the lowest finish time

---

## Scoring System

### Evaluation
- Your submission is simulated against **10,000 races** drawn from the true underlying distributions
- Races are generated from the same distributions that produced the historical data
- **Score = Average profit per race (in £)**

### Score Interpretation
- **Higher is better**
- Negative scores are possible (you can lose money on average)
- Maximum possible profit: Limited by your £10,000 stake and market odds
- Minimum possible loss: Full loss of £10,000 (if all bets fail)

---

## Strategic Considerations

### Key Insights

1. **Probability Inference**: The historical data reveals each horse's true win probability. The crowd's pool fractions may or may not accurately reflect these probabilities.

2. **Expected Value Calculation**: 
   For horse $i$ with true win probability $p_i$ and pool fraction $f_i$:
   $$EV = p_i \cdot \frac{0.85}{f_i} - 1$$

3. **Betting Principle**: 
   - **Only bet where your edge is positive** (i.e., where EV > 0)
   - Avoid betting on favorites if the market is overvalued
   - Look for undervalued horses where true probability > implied probability

### Optimal Bet Sizing

Consider the **Kelly Criterion**, which determines optimal bet sizing to maximize long-term wealth growth:

$$f^* = \frac{bp - q}{b}$$

Where:
- $f^*$ = fraction of bankroll to bet
- $b$ = decimal odds ($\frac{0.85}{f_i}$ in this context)
- $p$ = true win probability
- $q$ = 1 - p (loss probability)

**Benefits**: Maximizes expected logarithmic wealth growth while managing risk

### Risk-Adjusted Strategy

1. **Identify positive EV bets**: Calculate true probabilities from historical data
2. **Rank by EV**: Prioritize horses with the highest positive expected values
3. **Size bets**: Use Kelly Criterion or fractional Kelly (e.g., 0.25x Kelly) for conservative betting
4. **Diversify**: Consider spreading stakes across multiple positive EV bets to reduce variance
5. **Leave margin**: Don't necessarily use your entire £10,000 if only marginal opportunities exist

---

## Submission Requirements

### Form Inputs
Specify your stake on each horse (in £):
- Shadowfax: £0
- Iron Duke: £0
- Morningstar: £0
- Red Tide: £0
- Gallant Fox: £0
- Blue Streak: £0
- Copper Prince: £0
- Last Chance: £0

**Total stake must not exceed £10,000**

### Reasoning
Provide a written explanation of your approach and strategy.
- **Minimum**: 50 characters
- **Purpose**: Document your methodology and assumptions

---

## Submission Limits

- **Attempts remaining**: 3 of 3 submissions per round
- Choose your submissions carefully
- Previous submissions do not count against scoring

