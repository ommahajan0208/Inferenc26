# Horse Racing Betting Strategy: Implementation Details

## 1. Objective

The primary goal of this analysis was to develop a systematic, data-driven betting strategy for horse racing. Based on a historical dataset of race outcomes (`race_data.csv`), we aimed to identify undervalued horses and determine the optimal amount to bet on them from a starting portfolio of £10,000, using the Kelly Criterion for risk management.

## 2. What We Thought: The Planned Methodology

We followed a structured, multi-step approach to build the betting model:

1.  **Understand the Data**: The first step was to analyze the contents of `race_data.csv` to understand its structure and the information it contained, specifically focusing on the race winners.

2.  **Compute True Win Probabilities**: By analyzing the frequency of wins for each horse in the historical data, we could establish a "true" or objective probability of winning (`p_i`) for each horse. This is calculated as:
    `True Probability = (Number of Wins for Horse) / (Total Number of Races)`

3.  **Compare to Market Probabilities**: We were given a set of "market" probabilities (`f_i`), which represent the odds offered by bookmakers. The core idea was to find discrepancies between our calculated true probabilities and these market odds.

4.  **Use the Expected Value (EV) Formula**: To quantify the value of a potential bet, we used the Expected Value formula. A positive EV indicates a profitable bet in the long run. The formula, accounting for a 15% bookmaker's commission (vigorish), is:
    `EV = p_i * (0.85 / f_i) - 1`

5.  **Decide How Much to Bet (The Kelly Criterion)**: Simply finding a positive EV isn't enough; we needed to decide how much of our portfolio to risk. For this, we used the Kelly Criterion, a formula designed to maximize long-term growth of capital. The fraction of the portfolio to bet is calculated as:
    `Kelly Fraction = Edge / Odds`
    Where:
    *   `Edge = EV`
    *   `Odds = (0.85 / f_i) - 1`

## 3. What We Did: The Implementation

We implemented the strategy in a Python script (`betting_strategy.py`) using the `pandas` library for data manipulation.

1.  **Data Loading**: The script begins by loading `race_data.csv` into a pandas DataFrame.

2.  **Calculating Probabilities**:
    *   It counts the total number of races and the number of wins for each unique horse in the `winner` column.
    *   It calculates the `true_probability` for each horse based on these counts.
    *   The provided `market_probability` values were stored in a dictionary and mapped to each horse.

3.  **Calculating EV and Identifying Bets**:
    *   The script calculates the `ev_edge` for every horse using the formula described above.
    *   It then filters this list to keep only the horses with an `ev_edge` greater than zero, as these are the only bets worth considering.

4.  **Applying the Kelly Criterion**:
    *   For the identified profitable bets, the script calculates the `kelly_fraction`.
    *   A crucial step was added to normalize the Kelly fractions. If the sum of the fractions for all profitable bets exceeded 1, it meant the strategy was advising to bet more than 100% of the portfolio. The fractions were scaled down proportionally to sum to 1 in such cases.

5.  **Calculating Bet Amounts**:
    *   With the final Kelly fractions, the script calculates the exact monetary amount to bet on each horse from the £10,000 portfolio.

6.  **Reporting**: The script concludes by printing a clean, formatted table summarizing the optimal betting strategy, including the probabilities, EV, Kelly fraction, and final bet amount for each recommended bet.

## 4. What We Found: The Results

The analysis yielded a clear and actionable betting strategy. Three horses were identified as having a positive expected value.

The final output from the script was:

```
Optimal Betting Strategy:
Total Portfolio: £10,000.00
------------------------------------------------------------
            True Win % Market Win %  EV Edge Kelly Fraction Bet Amount
winner
Shadowfax       35.60%       10.00%  202.60%         27.01%  £2,701.33
Morningstar     32.80%       15.00%   85.87%         18.40%  £1,840.00
Iron Duke       14.80%       12.00%    4.83%          0.79%     £79.45
------------------------------------------------------------
Total Bet Amount: £4,620.79
Remaining Cash: £5,379.21
```

**Key Findings**:

*   **Shadowfax** was identified as the most significantly undervalued horse, with a true win probability (35.6%) far exceeding its market probability (10.0%). The model recommends a substantial bet of **£2,701.33**.
*   **Morningstar** was also found to be a strong bet, with a true probability of 32.8% against the market's 15.0%. The recommended bet is **£1,840.00**.
*   **Iron Duke** presented a much smaller, but still positive, edge. The model suggests a small bet of **£79.45**.
*   The total capital allocated to bets is **£4,620.79**, leaving **£5,379.21** of the portfolio in reserve, as dictated by the Kelly Criterion for optimal risk management.
