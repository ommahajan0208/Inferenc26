Inference
Rounds
Leaderboard
About
Dashboard
Sign out
← Back
Season 0 · Round 02

The House
The house always wins — unless the counters find you first.

medium
open
Closes in6d 13:27:21
0 submissions so far
The Setup
You run a betting booth in a casino. Each round, a blackjack hand is dealt from a fresh shoe. The hand has a payout value 
V
V that determines how much winners collect. Your job is to take bets on both sides: punters can bet the hand will be worth more than your price (buying from you) or less (selling to you).

You are not the only booth. Five house operators and all other participants are also taking bets on the same hand.

Each round:

A payout value 
V
V is drawn uniformly from 
[
0
,
1000
]
[0,1000].
Each booth operator (you, the house operators, and other participants) gets a peek at the shoe: a private signal 
s
=
V
+
ε
s=V+ε, where 
ε
∼
N
(
0
,
σ
2
)
ε∼N(0,σ 
2
 ) with 
σ
=
50
σ=50 (i.e. normally distributed noise with standard deviation 50 — roughly 95% of signals fall within ±100 of 
V
V).
Each booth posts a bid (the price they'll pay a seller) and an ask (the price they'll charge a buyer) based on their signal.
Punters arrive and take the best available price:
Regular punters have a vague sense of the hand's value. A buyer takes the lowest ask among all booths (if their reservation exceeds it). A seller takes the highest bid (if their reservation is below it). Each punter visits at most one booth.
Card counters know 
V
V exactly. They find the most profitable price available and take it.
After all bets settle, the hand is revealed.
Your PnL each round is the sum across all your fills:

If a punter buys from you at your ask: you earn 
(
ask
−
V
)
(ask−V) per unit.
If a punter sells to you at your bid: you earn 
(
V
−
bid
)
(V−bid) per unit.
The Competition
Five house operators are always running booths. They have different styles: some quote tight, some wide, some skew their prices, some are erratic. Their strategies are fixed and do not change between rounds.

All other participants' booths are also active during final scoring. If another participant quotes tighter than you, the punters go there instead. Your score depends on how you perform against the full field.

Historical Data
auction_history.csv contains 1,000 historical rounds. A reference booth was operating alongside the five house operators. Each row shows:

period — round number
true_value — the realised payout value of the hand
mm_signal — the signal the reference booth received
mm_bid, mm_ask — the reference booth's quotes
mm_num_buys, mm_num_sells — fills the reference booth received
mm_pnl — reference booth's PnL for that round
bot_tight_bid, bot_tight_ask — quotes from the tight operator
bot_wide_bid, bot_wide_ask — quotes from the wide operator
bot_skewed_bid, bot_skewed_ask — quotes from the skewed operator
bot_noisy_bid, bot_noisy_ask — quotes from the noisy operator
bot_fade_bid, bot_fade_ask — quotes from the fade operator
Study how the house operators quote. Work out which ones dominate in which situations. Find a strategy that captures the bets they miss.

Your Submission
Submit a pricing table of 21 entries. For each signal value 
s
∈
{
0
,
50
,
100
,
…
,
1000
}
s∈{0,50,100,…,1000}, enter your bid and ask using the form below.

Constraints:

0
≤
bid
≤
ask
≤
1200
0≤bid≤ask≤1200
Your bid and ask for intermediate signal values are interpolated linearly.
Scoring
All submissions are scored together in a single simulation of 5,000 rounds. Each round, every participant and all five house operators receive independent signals and post quotes. Punters take the best available prices.

Score = average PnL per round.

Higher is better. Negative scores are possible. Your score depends on the strategies submitted by other participants.

Hints
Your signal 
s
s is an unbiased estimate of 
V
V. Centering your prices around 
s
s is a natural starting point.
Adverse selection: card counters always take the best available price. If your ask is the lowest, you get picked off first when the hand turns out to be worth more.
Fill priority: punters go to the booth with the best price. If your spread is wider than every house operator, you rarely get action. If it's tighter, you get more bets but more adverse selection.
The historical data shows all house operator quotes per round. You can reconstruct their strategies and find the gaps.
A spread that beats the house operators but not by too much may perform better than an extremely tight spread that wins every bet (including the bad ones).
Background reading
Market making — how market makers profit from the bid-ask spread, and why they exist.
Bid-ask spread — the gap between buy and sell prices.
Glosten-Milgrom model — the foundational model for how informed traders (card counters) force spreads to widen.
Your answer
For each signal value, set your bid and ask. 0 ≤ bid ≤ ask ≤ 1200.

Signal	Bid	Ask
0	
0
0
50	
0
0
100	
0
0
150	
0
0
200	
0
0
250	
0
0
300	
0
0
350	
0
0
400	
0
0
450	
0
0
500	
0
0
550	
0
0
600	
0
0
650	
0
0
700	
0
0
750	
0
0
800	
0
0
850	
0
0
900	
0
0
950	
0
0
1000	
0
0
Reasoning
Explain your approach. Minimum 50 characters.

0 / 50 minimum

Submit answer
You can update your answer until the round closes.

Built by April.
Support ♥
Privacy
Terms
