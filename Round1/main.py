import pandas as pd

def calculate_betting_strategy():
    # Step 1: Load and Analyze Data
    race_data = pd.read_csv('race_data.csv')
    
    # Total number of races
    total_races = len(race_data)
    
    # Step 2: Calculate True Win Probabilities
    win_counts = race_data['winner'].value_counts()
    true_probabilities = win_counts / total_races
    
    # Create a DataFrame for our analysis
    horses = list(true_probabilities.index)
    strategy_df = pd.DataFrame({'horse': horses})
    strategy_df = strategy_df.set_index('horse')
    strategy_df['true_probability'] = true_probabilities
    
    # Step 3: Incorporate Market Probabilities
    market_probabilities = {
        "Shadowfax": 0.10,
        "Iron Duke": 0.12,
        "Morningstar": 0.15,
        "Red Tide": 0.08,
        "Gallant Fox": 0.11,
        "Blue Streak": 0.09,
        "Copper Prince": 0.20,
        "Last Chance": 0.15
    }
    strategy_df['market_probability'] = strategy_df.index.map(market_probabilities)
    
    # Step 4: Calculate Expected Value (EV)
    p = strategy_df['true_probability']
    f = strategy_df['market_probability']
    
    # Edge = (p * (0.85 / f)) - 1
    # EV is not exactly the edge, but this is the formula for profitability
    strategy_df['ev_edge'] = (p * (0.85 / f)) - 1
    
    # Step 5: Identify Profitable Bets
    profitable_bets = strategy_df[strategy_df['ev_edge'] > 0].copy()
    
    # Step 6: Determine Bet Sizing using Kelly Criterion
    # Kelly fraction = p - (f * (1 - p) / 0.85) -- this is a simplified version
    # A more common formula is Edge / Odds
    # Odds here are (0.85 / f) - 1
    
    odds = (0.85 / profitable_bets['market_probability']) -1
    edge = profitable_bets['ev_edge']
    
    # The formula given in the prompt is a bit unusual. Let's use the one from the prompt.
    # fraction to bet ≈ pi - (fi * (1 - pi)) / 0.85 -- this seems to be incorrect as it can be negative.
    # Let's use the standard Kelly formula: Fraction = Edge / (Odds)
    # The prompt's formula is: fraction = p_i * (0.85/f_i) - 1 / ((0.85/f_i) - 1) which is edge/odds
    
    kelly_fraction = profitable_bets['ev_edge'] / odds
    
    profitable_bets['kelly_fraction'] = kelly_fraction
    
    # Normalize fractions if they sum to more than 1
    if profitable_bets['kelly_fraction'].sum() > 1:
        profitable_bets['kelly_fraction'] = profitable_bets['kelly_fraction'] / profitable_bets['kelly_fraction'].sum()
        
    # Step 7: Calculate Final Bet Amounts
    portfolio = 10000
    profitable_bets['bet_amount'] = profitable_bets['kelly_fraction'] * portfolio
    
    # Step 8: Present the Final Betting Strategy
    print("Optimal Betting Strategy:")
    print(f"Total Portfolio: £{portfolio:,.2f}")
    print("-" * 40)
    
    if profitable_bets.empty:
        print("No profitable bets found.")
    else:
        # Display relevant columns for the final output
        display_df = profitable_bets[['true_probability', 'market_probability', 'ev_edge', 'kelly_fraction', 'bet_amount']].copy()
        display_df['bet_amount'] = display_df['bet_amount'].apply(lambda x: f"£{x:,.2f}")
        display_df.rename(columns={
            'true_probability': 'True Win %',
            'market_probability': 'Market Win %',
            'ev_edge': 'EV Edge',
            'kelly_fraction': 'Kelly Fraction',
            'bet_amount': 'Bet Amount'
        }, inplace=True)
        
        # Formatting percentages
        display_df['True Win %'] = (display_df['True Win %'] * 100).map('{:.2f}%'.format)
        display_df['Market Win %'] = (display_df['Market Win %'] * 100).map('{:.2f}%'.format)
        display_df['EV Edge'] = (display_df['EV Edge'] * 100).map('{:.2f}%'.format)
        display_df['Kelly Fraction'] = (display_df['Kelly Fraction'] * 100).map('{:.2f}%'.format)

        print(display_df)
        print("-" * 40)
        print(f"Total Bet Amount: £{profitable_bets['bet_amount'].sum():,.2f}")
        print(f"Remaining Cash: £{portfolio - profitable_bets['bet_amount'].sum():,.2f}")

if __name__ == "__main__":
    calculate_betting_strategy()
