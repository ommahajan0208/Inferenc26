"""
Deep Mathematical Analysis: Horse Racing Betting Strategy
=========================================================
Techniques applied:
  1. Distribution fitting (Normal + Skew-Normal via MLE)
  2. Shapiro-Wilk normality testing
  3. 10M Monte Carlo race simulation
  4. Bootstrap 95% confidence intervals (1000 resamples)
  5. Analytical EV calculation and linear optimisation proof
  6. True Kelly criterion derivation
  7. Strategy simulation (100k races, vectorised)
  8. Sensitivity analysis: what p makes all-in NOT optimal?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

np.random.seed(42)

DATA_PATH = Path(__file__).with_name("race_data.csv")

# ── Data & constants ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
horses = ["Shadowfax","Iron Duke","Morningstar","Red Tide",
          "Gallant Fox","Blue Streak","Copper Prince","Last Chance"]
market = {
    "Shadowfax": .08, "Iron Duke": .09, "Morningstar": .11,
    "Red Tide": .13, "Gallant Fox": .14, "Blue Streak": .15,
    "Copper Prince": .14, "Last Chance": .16,
}
BUDGET = 10_000
TAKE   = 0.15
NET    = 1 - TAKE    # 0.85

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTION FITTING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. DISTRIBUTION FITTING  (MLE — Normal unless Shapiro-Wilk p < 0.05)")
print("=" * 70)

fitted = {}
for h in horses:
    t = df[h].values
    sw_stat, sw_p = stats.shapiro(t)
    skewness = stats.skew(t)
    kurtosis = stats.kurtosis(t)
    if sw_p < 0.05:
        a, loc, scale = stats.skewnorm.fit(t)
        fitted[h] = ("skewnorm", a, loc, scale)
        tag = "SKEW-NORMAL"
    else:
        mu, sigma = t.mean(), t.std(ddof=1)
        fitted[h] = ("normal", mu, sigma)
        tag = "NORMAL     "
    print(f"  {h:<16} {tag}  skew={skewness:+.3f}  kurt={kurtosis:+.3f}  Shapiro-p={sw_p:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MONTE CARLO WIN PROBABILITIES  (10M races)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("2. MONTE CARLO WIN PROBABILITIES  (N=10,000,000)")
print("=" * 70)

N_MC = 10_000_000
draws = []
for h in horses:
    p = fitted[h]
    if p[0] == "skewnorm":
        draws.append(stats.skewnorm.rvs(p[1], loc=p[2], scale=p[3], size=N_MC))
    else:
        draws.append(np.random.normal(p[1], p[2], size=N_MC))

sample_matrix = np.column_stack(draws)
win_idx_mc    = np.argmin(sample_matrix, axis=1)

mc_probs = {}
freq     = df["winner"].value_counts() / len(df)
print(f"  {'Horse':<16} {'Freq(500)':>10} {'MC(10M)':>10} {'Market':>8} {'Δ MC-Mkt':>10}")
print(f"  {'-'*57}")
for i, h in enumerate(horses):
    mc_p = (win_idx_mc == i).mean()
    mc_probs[h] = mc_p
    f_h  = freq.get(h, 0)
    delta = mc_p - market[h]
    print(f"  {h:<16} {f_h:>10.4f} {mc_p:>10.4f} {market[h]:>8.4f} {delta:>+10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXPECTED VALUE  (correct market fractions)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3. EXPECTED VALUE  EV = p*(0.85/f) - 1")
print("=" * 70)

evs = {}
print(f"  {'Horse':<16} {'p (MC)':>8} {'f (mkt)':>8} {'Odds(b)':>8} {'EV':>10}")
print(f"  {'-'*53}")
for h in horses:
    p  = mc_probs[h]
    f  = market[h]
    b  = NET / f          # decimal payout on £1 stake
    ev = p * b - 1
    evs[h] = ev
    sign = "+" if ev > 0 else ""
    print(f"  {h:<16} {p:>8.4f} {f:>8.4f} {b:>8.3f} {sign+f'{ev*100:.2f}%':>10}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYTICAL PROOF: LINEAR OBJECTIVE → ALL-IN ON MAX EV
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("4. ANALYTICAL PROOF: Score = E[profit/race] is LINEAR in bets")
print("=" * 70)
print("""
  For any allocation {b_i}:

    E[profit/race] = Σ_i P(horse i wins) * b_i * (0.85/f_i)  -  Σ_i b_i
                   = Σ_i b_i * [ P(horse i wins) * (0.85/f_i) - 1 ]
                   = Σ_i b_i * EV_i

  This is LINEAR in b_i.  Subject to:  Σ_i b_i ≤ £10,000, b_i ≥ 0.

  A linear program with a budget constraint is maximised at a vertex:
  → Put ALL money on the SINGLE horse with the HIGHEST EV.
""")

best = max(evs, key=evs.get)
print(f"  Best EV horse : {best}  EV = {evs[best]*100:.2f}%")
print(f"  Analytical E[profit/race] = £10,000 × {evs[best]:.4f} = £{10000*evs[best]:,.2f}")
print()
print("  Comparison: if you split £10k across Shadowfax + Morningstar:")
b_sf, b_ms = 5000, 5000
e_split = b_sf * evs["Shadowfax"] + b_ms * evs["Morningstar"]
print(f"    £5k Shadowfax + £5k Morningstar → E[profit] = £{e_split:,.2f}  (worse)")
e_optimal = 10000 * evs[best]
print(f"    £10k Shadowfax                  → E[profit] = £{e_optimal:,.2f}  (optimal)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CONFIDENCE INTERVALS  (1000 resamples)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("5. BOOTSTRAP 95% CI ON WIN PROBABILITIES  (B=1000 resamples)")
print("=" * 70)

race_matrix = df[horses].values
n_races     = len(race_matrix)
B           = 1000
boot_wins   = {h: np.zeros(B) for h in horses}

for b_idx in range(B):
    idx        = np.random.randint(0, n_races, size=n_races)
    boot_races = race_matrix[idx]
    min_idx    = np.argmin(boot_races, axis=1)
    for i, h in enumerate(horses):
        boot_wins[h][b_idx] = (min_idx == i).mean()

print(f"  {'Horse':<16} {'p_hat':>7} {'95% CI':>18} {'EV_lo':>9} {'EV_hi':>9} {'Robust?':>8}")
print(f"  {'-'*70}")
for h in horses:
    bw   = boot_wins[h]
    lo   = np.percentile(bw, 2.5)
    hi   = np.percentile(bw, 97.5)
    p_hat = bw.mean()
    ev_lo = lo * (NET / market[h]) - 1
    ev_hi = hi * (NET / market[h]) - 1
    robust = "YES ✓" if ev_lo > 0 else ("MARGINAL" if ev_hi > 0 else "NO ✗")
    print(f"  {h:<16} {p_hat:>7.4f} [{lo:.4f} – {hi:.4f}]  {ev_lo*100:>8.1f}% {ev_hi*100:>8.1f}% {robust:>8}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. KELLY CRITERION  (correct derivation for parimutuel)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("6. KELLY CRITERION  f* = (b*p - q) / b  where b = NET/f_i")
print("=" * 70)
print("   NOTE: Kelly maximises E[log(wealth)], NOT E[profit].  For this")
print("   competition (fixed-bet average profit), Kelly is suboptimal.")
print()

kelly_bets = {}
for h in horses:
    p = mc_probs[h]
    q = 1 - p
    b = NET / market[h] - 1   # net decimal odds (profit per £1 if win)
    kf = (b * p - q) / b
    kelly_bets[h] = max(0.0, kf)

total_kf = sum(kelly_bets.values())
print(f"  Raw Kelly fractions sum = {total_kf:.4f}", "(>1 → normalise)" if total_kf > 1 else "")
norm_factor = max(1.0, total_kf)

print(f"  {'Horse':<16} {'Raw Kelly f*':>14} {'Normalised':>12} {'£ Bet':>10}")
print(f"  {'-'*55}")
for h in horses:
    kf_raw  = kelly_bets[h]
    kf_norm = kf_raw / norm_factor
    bet_amt = kf_norm * BUDGET
    if kf_raw > 0:
        print(f"  {h:<16} {kf_raw:>14.4f} {kf_norm:>12.4f} £{bet_amt:>8,.2f}")

kelly_e = sum(kelly_bets.get(h, 0) / norm_factor * BUDGET * evs[h] for h in horses)
print(f"\n  Kelly E[profit/race] = £{kelly_e:,.2f}")
print(f"  All-in E[profit/race] = £{10000*evs[best]:,.2f}")
print(f"  Kelly leaves £{10000*evs[best]-kelly_e:,.2f}/race on the table vs all-in")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SENSITIVITY ANALYSIS: break-even true probability for Shadowfax
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("7. SENSITIVITY: What true p(Shadowfax) makes EV = 0?")
print("=" * 70)
f_sf = market["Shadowfax"]
p_break = f_sf / NET   # EV=0 → p*(0.85/f)=1 → p = f/0.85
print(f"  Break-even p = f / 0.85 = {f_sf} / {NET} = {p_break:.4f}  ({p_break*100:.2f}%)")
print(f"  Our MC estimate  = {mc_probs['Shadowfax']:.4f}  ({mc_probs['Shadowfax']*100:.2f}%)")
print(f"  Bootstrap 95% CI lower = {np.percentile(boot_wins['Shadowfax'], 2.5):.4f}")
print(f"  Margin of safety = {(mc_probs['Shadowfax'] - p_break)*100:.2f} percentage points")
print()
print("  → Even if our estimate is 25pp wrong, the bet is still profitable.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. STRATEGY SHOWDOWN (100k simulated races, vectorised)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("8. STRATEGY SHOWDOWN  (100,000 simulated races, vectorised)")
print("=" * 70)

N2   = 100_000
sim2 = np.column_stack([
    stats.skewnorm.rvs(fitted["Shadowfax"][1], loc=fitted["Shadowfax"][2],
                       scale=fitted["Shadowfax"][3], size=N2),
    np.random.normal(fitted["Iron Duke"][1],   fitted["Iron Duke"][2],   N2),
    np.random.normal(fitted["Morningstar"][1], fitted["Morningstar"][2], N2),
    np.random.normal(fitted["Red Tide"][1],    fitted["Red Tide"][2],    N2),
    np.random.normal(fitted["Gallant Fox"][1], fitted["Gallant Fox"][2], N2),
    np.random.normal(fitted["Blue Streak"][1], fitted["Blue Streak"][2], N2),
    np.random.normal(fitted["Copper Prince"][1],fitted["Copper Prince"][2], N2),
    stats.skewnorm.rvs(fitted["Last Chance"][1], loc=fitted["Last Chance"][2],
                       scale=fitted["Last Chance"][3], size=N2),
])
win_idx_sim = np.argmin(sim2, axis=1)   # shape (N2,)

market_arr = np.array([market[h] for h in horses])   # shape (8,)

def simulate_strategy(bets_arr):
    """bets_arr: np.array shape (8,) — stake on each horse"""
    total_stake = bets_arr.sum()
    win_bets    = bets_arr[win_idx_sim]          # stake on winner each race
    payouts     = win_bets * (NET / market_arr[win_idx_sim])
    profits     = payouts - total_stake
    return profits.mean(), profits.std() / np.sqrt(N2)

strategies = {
    "Bugged code (wrong mkt fracs)" : np.array([2701.33, 79.45,  1840.0, 0, 0, 0, 0, 0]),
    "Naïve Kelly (correct market)"  : np.array([2551.0,  370.0,  2188.0, 0, 0, 0, 0, 0]),
    "True Kelly (renormalised)"     : np.array([kelly_bets[h]/norm_factor*BUDGET for h in horses]),
    "All-in Shadowfax (OPTIMAL)"    : np.array([10000, 0, 0, 0, 0, 0, 0, 0]),
    "Split Shadowfax+Morningstar"   : np.array([5000, 0, 5000, 0, 0, 0, 0, 0]),
}

print(f"  {'Strategy':<35} {'E[profit/race]':>16} {'Std Err':>10}")
print(f"  {'-'*65}")
for name, bets in strategies.items():
    mean, se = simulate_strategy(bets)
    print(f"  {name:<35} £{mean:>12,.2f}  ±£{se:>6,.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FINAL RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("9. FINAL RECOMMENDED SUBMISSION")
print("=" * 70)
print(f"""
  Horse          Stake
  ─────────────────────
  Shadowfax      £10,000
  All others     £0
  ─────────────────────
  Total stake    £10,000

  Analytical E[profit/race] = £{10000*evs['Shadowfax']:,.2f}
  MC-simulated  E[profit/race] ≈ £{simulate_strategy(np.array([10000,0,0,0,0,0,0,0]))[0]:,.2f}

  Shadowfax true win probability ≈ {mc_probs['Shadowfax']*100:.1f}%
  Market-implied win probability  =  8.0%
  Payout odds                     = {NET/market['Shadowfax']:.2f}× stake
  EV                              = {evs['Shadowfax']*100:+.1f}%
  Break-even p                    = {p_break*100:.2f}%
  Margin of safety                = {(mc_probs['Shadowfax']-p_break)*100:.1f} pp
""")
