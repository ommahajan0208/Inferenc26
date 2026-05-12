"""Math-heavy strategy experiments for Round 2.

This builds on robust_strategy_search.py and tries the deeper ideas:
- posterior boundary effects,
- Glosten-Milgrom-style adverse-selection spread estimates,
- direct competitive quote regressions,
- fill-rate x edge decomposition,
- robust objective variants: weighted-risk, minimax, and CVaR.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

import historical_simulator as sim
import robust_strategy_search as robust


ObjectiveFn = Callable[[pd.DataFrame], float]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    description: str
    objective: ObjectiveFn


@dataclass(frozen=True)
class PolyParams:
    basis: str
    degree: int
    mid_coef: np.ndarray
    h_bid_coef: np.ndarray
    h_ask_coef: np.ndarray

    def copy(self) -> "PolyParams":
        return PolyParams(
            basis=self.basis,
            degree=self.degree,
            mid_coef=self.mid_coef.copy(),
            h_bid_coef=self.h_bid_coef.copy(),
            h_ask_coef=self.h_ask_coef.copy(),
        )


@dataclass(frozen=True)
class AsymPolyParams:
    basis: str
    degree_mid: int
    degree_bid: int
    degree_ask: int
    mid_coef: np.ndarray
    h_bid_coef: np.ndarray
    h_ask_coef: np.ndarray

    def copy(self) -> "AsymPolyParams":
        return AsymPolyParams(
            basis=self.basis,
            degree_mid=self.degree_mid,
            degree_bid=self.degree_bid,
            degree_ask=self.degree_ask,
            mid_coef=self.mid_coef.copy(),
            h_bid_coef=self.h_bid_coef.copy(),
            h_ask_coef=self.h_ask_coef.copy(),
        )


def weighted_risk_objective(metrics: pd.DataFrame) -> float:
    pnl = metrics["pnl"].to_numpy(dtype=float)
    weights = metrics["weight"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * pnl))
    scenario_std = float(np.sqrt(np.sum(weights * (pnl - weighted) ** 2)))
    return weighted - 0.5 * scenario_std


def minimax_objective(metrics: pd.DataFrame) -> float:
    return float(metrics["pnl"].min())


def cvar_bottom3_objective(metrics: pd.DataFrame) -> float:
    return float(metrics.sort_values("pnl").head(3)["pnl"].mean())


def dro_moment_objective(metrics: pd.DataFrame) -> float:
    """Moment-ambiguity DRO proxy using scenario dispersion as ambiguity size."""
    pnl = metrics["pnl"].to_numpy(dtype=float)
    weights = metrics["weight"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * pnl))
    scenario_std = float(np.sqrt(np.sum(weights * (pnl - weighted) ** 2)))
    worst_gap = max(0.0, weighted - float(pnl.min()))
    return weighted - 0.75 * scenario_std - 0.05 * worst_gap


def chance_floor_objective(metrics: pd.DataFrame, floor: float = 16.0, alpha: float = 0.25) -> float:
    """Chance-constrained proxy: penalize too much weight below a scenario PnL floor."""
    pnl = metrics["pnl"].to_numpy(dtype=float)
    weights = metrics["weight"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * pnl))
    below_weight = float(np.sum(weights[pnl < floor]))
    shortfall = float(np.sum(weights * np.maximum(floor - pnl, 0.0)))
    violation = max(0.0, below_weight - alpha)
    return weighted - 0.35 * float(pnl.std(ddof=0)) - 10.0 * violation - shortfall


def spectral_risk_objective(metrics: pd.DataFrame) -> float:
    """Coherent spectral-risk proxy that puts extra weight on bad scenarios."""
    ordered = metrics.sort_values("pnl")
    pnl = ordered["pnl"].to_numpy(dtype=float)
    weights = ordered["weight"].to_numpy(dtype=float)
    risk_aversion = np.linspace(2.0, 0.5, len(pnl))
    spectral_weights = weights * risk_aversion
    spectral_weights = spectral_weights / spectral_weights.sum()
    return float(np.sum(spectral_weights * pnl))


OBJECTIVES = [
    ObjectiveSpec(
        name="weighted_risk",
        description="weighted mean PnL - 0.5 * scenario std",
        objective=weighted_risk_objective,
    ),
    ObjectiveSpec(
        name="minimax",
        description="maximize the worst scenario PnL",
        objective=minimax_objective,
    ),
    ObjectiveSpec(
        name="cvar_bottom3",
        description="maximize average of the three worst scenarios",
        objective=cvar_bottom3_objective,
    ),
    ObjectiveSpec(
        name="dro_moment",
        description="distributionally robust moment-ambiguity proxy",
        objective=dro_moment_objective,
    ),
    ObjectiveSpec(
        name="chance_floor",
        description="weighted PnL with a chance constraint on weak scenarios",
        objective=chance_floor_objective,
    ),
    ObjectiveSpec(
        name="spectral_risk",
        description="coherent spectral risk measure over scenario PnLs",
        objective=spectral_risk_objective,
    ),
]


def posterior_boundary_report() -> pd.DataFrame:
    signals = np.array([0, 25, 50, 75, 100, 150, 850, 900, 925, 950, 975, 1000], dtype=float)
    posterior = robust.posterior_mean_uniform_normal(signals)
    return pd.DataFrame(
        {
            "signal": signals.astype(int),
            "posterior_mean": posterior,
            "posterior_minus_signal": posterior - signals,
        }
    )


def signal_density(signals: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    signals = np.asarray(signals, dtype=float)
    upper = signals / sigma
    lower = (signals - 1000.0) / sigma
    return (robust.norm_cdf(upper) - robust.norm_cdf(lower)) / 1000.0


def posterior_grid(signal: float, grid: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    likelihood = robust.norm_pdf((signal - grid) / sigma) / sigma
    norm = np.trapezoid(likelihood, grid)
    if norm <= 0.0:
        raise ValueError("Posterior grid has zero mass")
    return likelihood / norm


def information_theory_report(sigma: float = 50.0) -> pd.DataFrame:
    s_grid = np.linspace(-350.0, 1350.0, 12_001)
    ps = signal_density(s_grid, sigma=sigma)
    entropy_signal = -float(np.trapezoid(np.where(ps > 0.0, ps * np.log(ps), 0.0), s_grid))
    entropy_noise = 0.5 * np.log(2.0 * np.pi * np.e * sigma * sigma)
    mutual_info_nats = entropy_signal - entropy_noise
    mutual_info_bits = mutual_info_nats / np.log(2.0)

    v_grid = np.linspace(0.0, 1000.0, 4001)
    bucket_signals = sim.OFFICIAL_SIGNALS
    posterior_vars = []
    posterior_stds = []
    for signal in bucket_signals:
        density = posterior_grid(float(signal), v_grid, sigma=sigma)
        mean = float(np.trapezoid(v_grid * density, v_grid))
        var = float(np.trapezoid((v_grid - mean) ** 2 * density, v_grid))
        posterior_vars.append(var)
        posterior_stds.append(np.sqrt(max(var, 0.0)))

    prior_var = 1000.0**2 / 12.0
    mean_posterior_var = float(np.mean(posterior_vars))
    return pd.DataFrame(
        [
            {
                "metric": "I(V;S)",
                "value": mutual_info_bits,
                "unit": "bits",
                "interpretation": "theoretical signal content per private peek",
            },
            {
                "metric": "h(S)-h(noise)",
                "value": mutual_info_nats,
                "unit": "nats",
                "interpretation": "same mutual information in natural units",
            },
            {
                "metric": "prior_std",
                "value": np.sqrt(prior_var),
                "unit": "value",
                "interpretation": "uncertainty before seeing signal",
            },
            {
                "metric": "mean_bucket_posterior_std",
                "value": float(np.mean(posterior_stds)),
                "unit": "value",
                "interpretation": "average uncertainty at the 21 submitted signals",
            },
            {
                "metric": "variance_reduction",
                "value": 1.0 - mean_posterior_var / prior_var,
                "unit": "fraction",
                "interpretation": "how much signal reduces V variance",
            },
        ]
    )


def posterior_kl_report(sigma: float = 50.0) -> pd.DataFrame:
    grid = np.linspace(0.0, 1000.0, 4001)
    prior_density = 1.0 / 1000.0
    rows = []
    for signal in sim.OFFICIAL_SIGNALS:
        density = posterior_grid(float(signal), grid, sigma=sigma)
        kl_nats = float(
            np.trapezoid(
                np.where(density > 0.0, density * np.log(density / prior_density), 0.0),
                grid,
            )
        )
        mean = float(np.trapezoid(grid * density, grid))
        std = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))
        rows.append(
            {
                "signal": int(signal),
                "posterior_mean": mean,
                "posterior_std": std,
                "kl_bits": kl_nats / np.log(2.0),
            }
        )
    return pd.DataFrame(rows)


def lloyd_max_knots(
    n_knots: int,
    grid: np.ndarray,
    weights: np.ndarray,
    iterations: int = 80,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    knots = np.interp((np.arange(n_knots) + 0.5) / n_knots, cdf, grid)
    for _ in range(iterations):
        boundaries = np.r_[
            grid[0],
            (knots[:-1] + knots[1:]) / 2.0,
            grid[-1],
        ]
        next_knots = knots.copy()
        for idx in range(n_knots):
            mask = (grid >= boundaries[idx]) & (grid <= boundaries[idx + 1])
            if np.any(mask):
                mass = float(weights[mask].sum())
                if mass > 0.0:
                    next_knots[idx] = float(np.sum(grid[mask] * weights[mask]) / mass)
        if np.max(np.abs(next_knots - knots)) < 1e-6:
            break
        knots = next_knots
    return knots


def quantization_distortion(grid: np.ndarray, weights: np.ndarray, knots: np.ndarray) -> float:
    nearest = np.min((grid[:, None] - knots[None, :]) ** 2, axis=1)
    return float(np.sum(weights * nearest) / np.sum(weights))


def rate_distortion_report() -> pd.DataFrame:
    grid = np.linspace(0.0, 1000.0, 5001)
    weights = signal_density(grid)
    weights = weights / weights.sum()
    rows = []
    for n_knots in (8, 21):
        uniform = np.linspace(0.0, 1000.0, n_knots)
        optimal = lloyd_max_knots(n_knots, grid, weights)
        uniform_mse = quantization_distortion(grid, weights, uniform)
        optimal_mse = quantization_distortion(grid, weights, optimal)
        rows.append(
            {
                "knots": n_knots,
                "uniform_mse": uniform_mse,
                "lloyd_mse": optimal_mse,
                "mse_reduction": 1.0 - optimal_mse / uniform_mse,
                "suggested_knots": ", ".join(f"{value:.0f}" for value in optimal),
            }
        )
    return pd.DataFrame(rows)


def competitive_regression_report(df: pd.DataFrame) -> pd.DataFrame:
    true_value = df["true_value"].to_numpy(dtype=float)
    design = np.column_stack([true_value, np.ones(len(df))])
    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    best_bid = bot_bids.max(axis=1)
    best_ask = bot_asks.min(axis=1)
    rows = []
    for name in sim.BOT_NAMES:
        bid = df[f"bot_{name}_bid"].to_numpy(dtype=float)
        ask = df[f"bot_{name}_ask"].to_numpy(dtype=float)
        bid_coef = np.linalg.lstsq(design, bid, rcond=None)[0]
        ask_coef = np.linalg.lstsq(design, ask, rcond=None)[0]
        bid_resid = bid - design @ bid_coef
        ask_resid = ask - design @ ask_coef
        rows.append(
            {
                "bot": name,
                "bid_alpha": bid_coef[0],
                "bid_beta": bid_coef[1],
                "bid_resid_std": bid_resid.std(ddof=1),
                "ask_alpha": ask_coef[0],
                "ask_gamma": ask_coef[1],
                "ask_resid_std": ask_resid.std(ddof=1),
                "ask_best": float(np.mean(np.abs(ask - best_ask) <= sim.TIE_TOL)),
                "bid_best": float(np.mean(np.abs(bid - best_bid) <= sim.TIE_TOL)),
            }
        )
    return pd.DataFrame(rows)


def huber_regression(x: np.ndarray, y: np.ndarray, c: float = 1.345, iterations: int = 40) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x))])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iterations):
        resid = y - design @ coef
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 1e-9:
            break
        u = resid / scale
        weights = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_y = y * np.sqrt(weights)
        next_coef = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(next_coef - coef)) < 1e-8:
            coef = next_coef
            break
        coef = next_coef
    return coef


def robust_regression_report(df: pd.DataFrame) -> pd.DataFrame:
    true_value = df["true_value"].to_numpy(dtype=float)
    rows = []
    for name in sim.BOT_NAMES:
        for side in ("bid", "ask"):
            y = df[f"bot_{name}_{side}"].to_numpy(dtype=float)
            design = np.column_stack([true_value, np.ones(len(df))])
            ols = np.linalg.lstsq(design, y, rcond=None)[0]
            huber = huber_regression(true_value, y)
            ols_resid = y - design @ ols
            huber_resid = y - design @ huber
            rows.append(
                {
                    "bot": name,
                    "side": side,
                    "ols_alpha": ols[0],
                    "huber_alpha": huber[0],
                    "ols_intercept": ols[1],
                    "huber_intercept": huber[1],
                    "ols_rmse": float(np.sqrt(np.mean(ols_resid**2))),
                    "huber_rmse": float(np.sqrt(np.mean(huber_resid**2))),
                }
            )
    return pd.DataFrame(rows)


def conformal_competitor_report(df: pd.DataFrame) -> pd.DataFrame:
    signal = df["mm_signal"].to_numpy(dtype=float)
    center = robust.posterior_mean_uniform_normal(signal)
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    residuals = {
        "best_ask_minus_center": bot_asks.min(axis=1) - center,
        "center_minus_best_bid": center - bot_bids.max(axis=1),
    }
    rows = []
    for name, resid in residuals.items():
        for coverage in (0.80, 0.90, 0.95):
            lo = float(np.quantile(resid, (1.0 - coverage) / 2.0))
            hi = float(np.quantile(resid, 1.0 - (1.0 - coverage) / 2.0))
            rows.append(
                {
                    "quantity": name,
                    "coverage": coverage,
                    "lower": lo,
                    "upper": hi,
                    "width": hi - lo,
                }
            )
    return pd.DataFrame(rows)


def gm_adverse_selection_report(df: pd.DataFrame) -> pd.DataFrame:
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    true_value = df["true_value"].to_numpy(dtype=float)
    mm_bid = df["mm_bid"].to_numpy(dtype=float)
    mm_ask = df["mm_ask"].to_numpy(dtype=float)
    mid = (mm_bid + mm_ask) / 2.0
    rows = []
    side_specs = [
        (
            "ask",
            bot_asks.min(axis=1) - mm_ask,
            df["mm_num_buys"].to_numpy(dtype=float),
            true_value > mm_ask,
            mm_ask - true_value,
        ),
        (
            "bid",
            mm_bid - bot_bids.max(axis=1),
            df["mm_num_sells"].to_numpy(dtype=float),
            true_value < mm_bid,
            true_value - mm_bid,
        ),
    ]
    for side, edge, fills, adverse, per_fill_pnl in side_specs:
        won = edge > sim.TIE_TOL
        weight = fills * won
        total_fills = float(weight.sum())
        adverse_fills = float((weight * adverse).sum())
        lambda_hat = sim.safe_fraction(adverse_fills, total_fills)
        abs_error = np.abs(true_value - mid)
        mean_abs_trade_error = sim.safe_fraction(float((weight * abs_error).sum()), total_fills)
        gm_half_spread = lambda_hat * mean_abs_trade_error
        pnl_per_fill = sim.safe_fraction(float((weight * per_fill_pnl).sum()), total_fills)
        rows.append(
            {
                "side": side,
                "win_rounds": int(won.sum()),
                "fills": total_fills,
                "lambda_adverse": lambda_hat,
                "mean_abs_v_minus_mid": mean_abs_trade_error,
                "gm_half_spread_proxy": gm_half_spread,
                "realized_pnl_per_fill": pnl_per_fill,
            }
        )
    return pd.DataFrame(rows)


def microstructure_risk_report(df: pd.DataFrame) -> pd.DataFrame:
    true_value = df["true_value"].to_numpy(dtype=float)
    mm_bid = df["mm_bid"].to_numpy(dtype=float)
    mm_ask = df["mm_ask"].to_numpy(dtype=float)
    mid = (mm_bid + mm_ask) / 2.0
    spread = mm_ask - mm_bid
    buys = df["mm_num_buys"].to_numpy(dtype=float)
    sells = df["mm_num_sells"].to_numpy(dtype=float)
    fills = buys + sells
    signed_flow = buys - sells
    price_move = true_value - mid
    flow_var = float(np.var(signed_flow))
    kyle_lambda = sim.safe_fraction(float(np.cov(signed_flow, price_move, ddof=1)[0, 1]), flow_var)
    adverse_fills = buys * (true_value > mm_ask) + sells * (true_value < mm_bid)
    pin_proxy = sim.safe_fraction(float(adverse_fills.sum()), float(fills.sum()))
    pnl_per_fill = np.divide(
        np.abs(df["mm_pnl"].to_numpy(dtype=float)),
        np.maximum(fills, 1.0),
    )
    amihud_proxy = float(np.mean(pnl_per_fill / np.maximum(np.abs(mid), 1.0)))
    spread_diff = np.diff(spread)
    roll_cov = float(np.cov(spread_diff[1:], spread_diff[:-1], ddof=1)[0, 1])
    roll_spread_proxy = 2.0 * np.sqrt(max(-roll_cov, 0.0))
    ask_pin = sim.safe_fraction(float((buys * (true_value > mm_ask)).sum()), float(buys.sum()))
    bid_pin = sim.safe_fraction(float((sells * (true_value < mm_bid)).sum()), float(sells.sum()))
    return pd.DataFrame(
        [
            {
                "metric": "PIN_proxy_all_fills",
                "value": pin_proxy,
                "interpretation": "fraction of reference fills that were directionally adverse",
            },
            {
                "metric": "PIN_proxy_ask",
                "value": ask_pin,
                "interpretation": "adverse fraction among buyer fills",
            },
            {
                "metric": "PIN_proxy_bid",
                "value": bid_pin,
                "interpretation": "adverse fraction among seller fills",
            },
            {
                "metric": "Kyle_lambda_proxy",
                "value": kyle_lambda,
                "interpretation": "value move per unit signed flow",
            },
            {
                "metric": "Roll_spread_proxy",
                "value": roll_spread_proxy,
                "interpretation": "serial-covariance implied spread proxy",
            },
            {
                "metric": "Amihud_proxy",
                "value": amihud_proxy,
                "interpretation": "absolute PnL impact per fill per unit mid",
            },
        ]
    )


def fill_decomposition(
    table: pd.DataFrame,
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    scenario: str = "no_clones",
) -> pd.DataFrame:
    field = next(field for field in fields if field.name == scenario)
    bid, ask = sim.table_quotes(field.our_signal, table)
    ask_share, ask_won = robust.tie_aware_share(
        ask,
        field.best_ask,
        field.best_ask_count,
        side="ask",
    )
    bid_share, bid_won = robust.tie_aware_share(
        bid,
        field.best_bid,
        field.best_bid_count,
        side="bid",
    )
    side_data = [
        (
            "ask",
            ask_won,
            ask_share,
            np.maximum(field.best_ask - ask, 0.0),
            field.true_value > ask,
            ask - field.true_value,
            fill_models.ask,
        ),
        (
            "bid",
            bid_won,
            bid_share,
            np.maximum(bid - field.best_bid, 0.0),
            field.true_value < bid,
            field.true_value - bid,
            fill_models.bid,
        ),
    ]
    rows = []
    n = len(field.true_value)
    for side, won, share, edge, adverse, per_unit_pnl, model in side_data:
        expected_fills = share * model.predict(edge, adverse)
        pnl = expected_fills * per_unit_pnl
        total_wins = int(won.sum())
        total_fills = float(expected_fills.sum())
        rows.append(
            {
                "side": side,
                "win_prob": float(won.mean()),
                "fills_given_win": sim.safe_fraction(total_fills, total_wins),
                "edge_per_fill": sim.safe_fraction(float(pnl.sum()), total_fills),
                "pnl_per_round": float(pnl.sum() / n),
                "adverse_given_win": sim.safe_fraction((won & adverse).sum(), total_wins),
                "mean_quote_edge": sim.safe_fraction(float((edge * won).sum()), total_wins),
            }
        )
    return pd.DataFrame(rows)


def previous_best_table() -> pd.DataFrame:
    """The refined robust table from the previous best reported result."""
    return pd.DataFrame(
        {
            "signal": sim.OFFICIAL_SIGNALS.astype(int),
            "bid": [
                5.81,
                5.81,
                5.81,
                32.55,
                72.99,
                120.91,
                173.53,
                228.69,
                284.65,
                340.08,
                394.10,
                446.23,
                496.42,
                545.03,
                592.86,
                641.10,
                691.39,
                745.56,
                803.99,
                862.80,
                920.50,
            ],
            "ask": [
                94.42,
                140.58,
                192.62,
                247.73,
                300.89,
                351.49,
                400.57,
                449.12,
                497.88,
                547.36,
                597.79,
                649.17,
                701.26,
                753.54,
                805.26,
                855.42,
                902.76,
                945.58,
                979.99,
                997.26,
                1002.90,
            ],
        }
    )


def robust_weighted_prior_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": sim.OFFICIAL_SIGNALS.astype(int),
            "bid": [
                0.00,
                0.00,
                18.00,
                45.84,
                76.00,
                131.00,
                186.00,
                241.00,
                296.00,
                346.50,
                397.00,
                447.50,
                498.00,
                550.00,
                602.00,
                654.00,
                706.00,
                753.16,
                798.00,
                859.95,
                908.00,
            ],
            "ask": [
                81.00,
                130.55,
                194.00,
                238.84,
                286.00,
                338.50,
                391.00,
                443.50,
                496.00,
                550.00,
                604.00,
                658.00,
                712.00,
                759.50,
                807.00,
                854.50,
                902.00,
                943.66,
                983.00,
                998.45,
                1000.00,
            ],
        }
    )


def pava_non_decreasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(weights, dtype=float)

    block_values: list[float] = []
    block_weights: list[float] = []
    block_counts: list[int] = []
    for value, weight in zip(y, weights):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_counts.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / total_weight
            total_count = block_counts[-2] + block_counts[-1]
            block_values[-2:] = [pooled]
            block_weights[-2:] = [total_weight]
            block_counts[-2:] = [total_count]

    out = np.empty_like(y, dtype=float)
    cursor = 0
    for value, count in zip(block_values, block_counts):
        out[cursor : cursor + count] = value
        cursor += count
    return out


def enforce_table_shape(table: pd.DataFrame, min_spread: float = 1.0) -> pd.DataFrame:
    shaped = table.copy(deep=True)
    bid = pava_non_decreasing(shaped["bid"].to_numpy(dtype=float))
    ask = pava_non_decreasing(shaped["ask"].to_numpy(dtype=float))
    crossed = bid > ask - min_spread
    if np.any(crossed):
        mid = (bid[crossed] + ask[crossed]) / 2.0
        bid[crossed] = mid - min_spread / 2.0
        ask[crossed] = mid + min_spread / 2.0
        bid = pava_non_decreasing(bid)
        ask = pava_non_decreasing(np.maximum(ask, bid + min_spread))
    shaped["bid"] = np.round(np.clip(bid, 0.0, 1200.0), 2)
    shaped["ask"] = np.round(np.clip(ask, 0.0, 1200.0), 2)
    robust.validate_table(shaped, require_monotone=True)
    return shaped


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0.0
    if not np.any(positive):
        return float(np.median(values))
    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    return float(values[min(idx, len(values) - 1)])


def consensus_table_from_fields(
    fields: list[robust.ScenarioField],
    bandwidth: float = 60.0,
) -> pd.DataFrame:
    signals = sim.OFFICIAL_SIGNALS
    consensus_bid = []
    consensus_ask = []
    for signal in signals:
        bid_chunks = []
        ask_chunks = []
        weight_chunks = []
        for field in fields:
            weights = np.exp(-0.5 * ((field.our_signal - signal) / bandwidth) ** 2)
            weights *= field.weight
            bid_chunks.append(field.best_bid)
            ask_chunks.append(field.best_ask)
            weight_chunks.append(weights)
        bids = np.concatenate(bid_chunks)
        asks = np.concatenate(ask_chunks)
        weights = np.concatenate(weight_chunks)
        consensus_bid.append(weighted_median(bids, weights))
        consensus_ask.append(weighted_median(asks, weights))
    return pd.DataFrame(
        {
            "signal": signals.astype(int),
            "bid": np.round(np.asarray(consensus_bid, dtype=float), 2),
            "ask": np.round(np.asarray(consensus_ask, dtype=float), 2),
        }
    )


def boundary_override_table(
    base_table: pd.DataFrame,
    include_edges: bool,
    low_cutoff: float = 150.0,
    high_cutoff: float = 850.0,
    mid_shift_low: float = 0.0,
    mid_shift_high: float = 0.0,
) -> pd.DataFrame:
    signal = base_table["signal"].to_numpy(dtype=float)
    bid = base_table["bid"].to_numpy(dtype=float)
    ask = base_table["ask"].to_numpy(dtype=float)
    mid = (bid + ask) / 2.0
    h_bid = mid - bid
    h_ask = ask - mid
    base_mid = robust.posterior_mean_uniform_normal(signal)
    if include_edges:
        low_mask = signal <= low_cutoff
        high_mask = signal >= high_cutoff
    else:
        low_mask = signal < low_cutoff
        high_mask = signal > high_cutoff
    mid = mid.copy()
    mid[low_mask] = base_mid[low_mask] + mid_shift_low
    mid[high_mask] = base_mid[high_mask] + mid_shift_high
    bid = mid - h_bid
    ask = mid + h_ask
    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(np.clip(bid, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(ask, 0.0, 1200.0), 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def apply_micro_undercut(
    table: pd.DataFrame,
    consensus_table: pd.DataFrame,
    eps_bid: float = 1.0,
    eps_ask: float = 1.0,
) -> pd.DataFrame:
    adjusted = table.copy(deep=True)
    consensus_bid = consensus_table["bid"].to_numpy(dtype=float)
    consensus_ask = consensus_table["ask"].to_numpy(dtype=float)
    bids = np.maximum(adjusted["bid"].to_numpy(dtype=float), consensus_bid + eps_bid)
    asks = np.minimum(adjusted["ask"].to_numpy(dtype=float), consensus_ask - eps_ask)
    adjusted["bid"] = np.round(bids, 2)
    adjusted["ask"] = np.round(asks, 2)
    return enforce_table_shape(adjusted, min_spread=5.0)


def boundary_micro_undercut_table(
    base_table: pd.DataFrame,
    consensus_table: pd.DataFrame,
    include_edges: bool,
    low_cutoff: float = 150.0,
    high_cutoff: float = 850.0,
    mid_shift_low: float = 0.0,
    mid_shift_high: float = 0.0,
    eps_bid: float = 1.0,
    eps_ask: float = 1.0,
) -> pd.DataFrame:
    boundary = boundary_override_table(
        base_table,
        include_edges=include_edges,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        mid_shift_low=mid_shift_low,
        mid_shift_high=mid_shift_high,
    )
    return apply_micro_undercut(boundary, consensus_table, eps_bid=eps_bid, eps_ask=eps_ask)


def kde_local_best_response_table(
    df: pd.DataFrame,
    fill_models: sim.FillModels,
    bandwidth: float = 85.0,
    grid_step: float = 2.0,
) -> pd.DataFrame:
    signal = df["mm_signal"].to_numpy(dtype=float)
    true_value = df["true_value"].to_numpy(dtype=float)
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    best_ask = bot_asks.min(axis=1)
    best_bid = bot_bids.max(axis=1)
    quote_grid = np.arange(0.0, 1200.0 + grid_step * 0.5, grid_step)
    rows = []
    for official_signal in sim.OFFICIAL_SIGNALS:
        weights = np.exp(-0.5 * ((signal - official_signal) / bandwidth) ** 2)
        weights = weights / np.maximum(weights.sum(), 1e-12)
        ask_scores = []
        bid_scores = []
        for quote in quote_grid:
            ask_won = quote <= best_ask + sim.TIE_TOL
            ask_edge = np.maximum(best_ask - quote, 0.0)
            ask_adverse = true_value > quote
            ask_fills = ask_won * fill_models.ask.predict(ask_edge, ask_adverse)
            ask_scores.append(float(np.sum(weights * ask_fills * (quote - true_value))))

            bid_won = quote >= best_bid - sim.TIE_TOL
            bid_edge = np.maximum(quote - best_bid, 0.0)
            bid_adverse = true_value < quote
            bid_fills = bid_won * fill_models.bid.predict(bid_edge, bid_adverse)
            bid_scores.append(float(np.sum(weights * bid_fills * (true_value - quote))))

        rows.append(
            {
                "signal": int(official_signal),
                "bid": float(quote_grid[int(np.argmax(bid_scores))]),
                "ask": float(quote_grid[int(np.argmax(ask_scores))]),
            }
        )
    return enforce_table_shape(pd.DataFrame(rows), min_spread=5.0)


def estimate_fill_decay_k(fill_model: sim.SideFillModel) -> float:
    finite_edges = np.array([2.5, 7.5, 15.0, 30.0, 60.0], dtype=float)
    mean_fills = fill_model.estimates[: len(finite_edges), :].mean(axis=1)
    positive = mean_fills > 1e-9
    if positive.sum() < 2:
        return 0.03
    slope = np.polyfit(finite_edges[positive], np.log(mean_fills[positive]), 1)[0]
    return float(np.clip(abs(slope), 0.005, 0.08))


def posterior_std_by_signal(signals: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    grid = np.linspace(0.0, 1000.0, 2001)
    stds = []
    for signal in np.asarray(signals, dtype=float):
        density = posterior_grid(float(signal), grid, sigma=sigma)
        mean = float(np.trapezoid(grid * density, grid))
        stds.append(float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid))))
    return np.asarray(stds, dtype=float)


def avellaneda_stoikov_table(fill_models: sim.FillModels, gamma: float = 0.012) -> pd.DataFrame:
    signal = sim.OFFICIAL_SIGNALS
    center = robust.posterior_mean_uniform_normal(signal) - 4.0
    posterior_var = posterior_std_by_signal(signal) ** 2
    k_ask = estimate_fill_decay_k(fill_models.ask)
    k_bid = estimate_fill_decay_k(fill_models.bid)
    ask_spread = gamma * posterior_var / 2.0 + (1.0 / gamma) * np.log1p(gamma / k_ask) + 18.0
    bid_spread = gamma * posterior_var / 2.0 + (1.0 / gamma) * np.log1p(gamma / k_bid) + 26.0
    ask_spread = np.clip(ask_spread, 65.0, 170.0)
    bid_spread = np.clip(bid_spread, 75.0, 190.0)
    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(np.clip(center - bid_spread, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(center + ask_spread, 0.0, 1200.0), 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def poly_basis(signals: np.ndarray, degree: int | str, basis: str | int) -> np.ndarray:
    # Accept swapped (basis, degree) order to avoid accidental misuse.
    if isinstance(degree, str) and isinstance(basis, (int, np.integer)):
        degree, basis = int(basis), degree
    if not isinstance(degree, (int, np.integer)):
        raise TypeError("poly_basis degree must be an int")
    if not isinstance(basis, str):
        raise TypeError("poly_basis basis must be a str")
    x = 2.0 * np.asarray(signals, dtype=float) / 1000.0 - 1.0
    columns = []
    if basis == "chebyshev":
        columns.append(np.ones_like(x))
        if degree >= 1:
            columns.append(x)
        for n in range(2, degree + 1):
            columns.append(2.0 * x * columns[-1] - columns[-2])
    elif basis == "legendre":
        columns.append(np.ones_like(x))
        if degree >= 1:
            columns.append(x)
        for n in range(2, degree + 1):
            columns.append(((2 * n - 1) * x * columns[-1] - (n - 1) * columns[-2]) / n)
    else:
        raise ValueError(f"Unknown polynomial basis: {basis}")
    return np.column_stack(columns[: degree + 1])


def fit_poly_params_from_table(table: pd.DataFrame, basis: str, degree: int) -> PolyParams:
    signal = table["signal"].to_numpy(dtype=float)
    base_mid = robust.posterior_mean_uniform_normal(signal)
    bid = table["bid"].to_numpy(dtype=float)
    ask = table["ask"].to_numpy(dtype=float)
    mid = (bid + ask) / 2.0
    design = poly_basis(signal, degree, basis)
    mid_coef = np.linalg.lstsq(design, mid - base_mid, rcond=None)[0]
    h_bid_coef = np.linalg.lstsq(design, mid - bid, rcond=None)[0]
    h_ask_coef = np.linalg.lstsq(design, ask - mid, rcond=None)[0]
    return PolyParams(
        basis=basis,
        degree=degree,
        mid_coef=mid_coef,
        h_bid_coef=h_bid_coef,
        h_ask_coef=h_ask_coef,
    )


def make_poly_table(params: PolyParams) -> pd.DataFrame:
    signal = sim.OFFICIAL_SIGNALS
    design = poly_basis(signal, params.degree, params.basis)
    base_mid = robust.posterior_mean_uniform_normal(signal)
    mid = base_mid + design @ params.mid_coef
    h_bid = np.clip(design @ params.h_bid_coef, 40.0, 240.0)
    h_ask = np.clip(design @ params.h_ask_coef, 40.0, 240.0)
    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(np.clip(mid - h_bid, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(mid + h_ask, 0.0, 1200.0), 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def bounded_poly_update(params: PolyParams, array_name: str, index: int, delta: float) -> PolyParams:
    candidate = params.copy()
    target = getattr(candidate, array_name)
    if index == 0:
        bound = 240.0
    else:
        bound = 120.0
    target[index] = float(np.clip(target[index] + delta, -bound, bound))
    return candidate


def optimize_poly_params(
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
    start_table: pd.DataFrame,
    basis: str,
    degree: int = 4,
    steps: list[float] | None = None,
    max_passes_per_step: int = 2,
) -> tuple[PolyParams, pd.DataFrame, float, pd.DataFrame]:
    if steps is None:
        steps = [8.0, 4.0, 2.0, 1.0]
    params = fit_poly_params_from_table(start_table, basis=basis, degree=degree)
    table = make_poly_table(params)
    best_value, best_metrics = evaluate_table_for_objective(table, fields, fill_models, objective)
    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for array_name in ("mid_coef", "h_bid_coef", "h_ask_coef"):
                for index in range(degree + 1):
                    for direction in (1.0, -1.0):
                        candidate_params = bounded_poly_update(params, array_name, index, direction * step)
                        try:
                            candidate_table = make_poly_table(candidate_params)
                        except ValueError:
                            continue
                        candidate_value, candidate_metrics = evaluate_table_for_objective(
                            candidate_table,
                            fields,
                            fill_models,
                            objective,
                        )
                        if candidate_value > best_value + 1e-9:
                            params = candidate_params
                            table = candidate_table
                            best_value = candidate_value
                            best_metrics = candidate_metrics
                            improved = True
            if not improved:
                break
    return params, table, best_value, best_metrics


def build_fields_for_seeds(
    bot_models: list[robust.BotQuoteModel],
    n_rounds: int,
    seeds: int,
    first_seed: int,
    scenario_tables: dict[str, list[pd.DataFrame]],
    scenario_weights: dict[str, float] | None = None,
) -> list[list[robust.ScenarioField]]:
    fields_by_seed = []
    for seed_offset in range(seeds):
        kwargs = {"scenario_tables": scenario_tables}
        if scenario_weights is not None:
            kwargs["scenario_weights"] = scenario_weights
        fields_by_seed.append(
            robust.build_scenario_fields(
                bot_models,
                n_rounds=n_rounds,
                seed=first_seed + seed_offset,
                **kwargs,
            )
        )
    return fields_by_seed


def evaluate_objective_across_fields(
    table: pd.DataFrame,
    fields_by_seed: list[list[robust.ScenarioField]],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
) -> tuple[np.ndarray, dict[str, float]]:
    values = []
    for fields in fields_by_seed:
        value, _ = evaluate_table_for_objective(table, fields, fill_models, objective)
        values.append(value)
    arr = np.asarray(values, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return arr, {"objective_mean": float(arr.mean()), "objective_se": se}


def sensitivity_audit_table(
    table: pd.DataFrame,
    fields_by_seed: list[list[robust.ScenarioField]],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
    deltas: list[float] | None = None,
) -> pd.DataFrame:
    if deltas is None:
        deltas = [1.0, 3.0, 5.0]

    baseline_values, baseline_summary = evaluate_objective_across_fields(
        table,
        fields_by_seed,
        fill_models,
        objective,
    )

    rows = []
    for row_idx, signal in enumerate(table["signal"].to_numpy(dtype=int)):
        bid = float(table.loc[row_idx, "bid"])
        ask = float(table.loc[row_idx, "ask"])
        for side in ("bid", "ask"):
            base_value = bid if side == "bid" else ask
            for delta in deltas:
                for direction in (1.0, -1.0):
                    target_delta = direction * delta
                    new_value = base_value + target_delta
                    clipped = False
                    if new_value < 0.0:
                        new_value = 0.0
                        clipped = True
                    elif new_value > 1200.0:
                        new_value = 1200.0
                        clipped = True

                    valid = True
                    if side == "bid" and new_value > ask - sim.TIE_TOL:
                        valid = False
                    if side == "ask" and new_value < bid + sim.TIE_TOL:
                        valid = False

                    diff_mean = np.nan
                    diff_se = np.nan
                    objective_mean = np.nan
                    objective_se = np.nan
                    if valid:
                        candidate = table.copy(deep=True)
                        candidate.loc[row_idx, side] = new_value
                        try:
                            robust.validate_table(candidate)
                        except ValueError:
                            valid = False
                        if valid:
                            candidate_values, candidate_summary = evaluate_objective_across_fields(
                                candidate,
                                fields_by_seed,
                                fill_models,
                                objective,
                            )
                            diff = candidate_values - baseline_values
                            diff_mean = float(diff.mean())
                            diff_se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
                            objective_mean = candidate_summary["objective_mean"]
                            objective_se = candidate_summary["objective_se"]

                    rows.append(
                        {
                            "signal": signal,
                            "side": side,
                            "delta": target_delta,
                            "actual_delta": float(new_value - base_value),
                            "clipped": clipped,
                            "valid": valid,
                            "objective_mean": objective_mean,
                            "objective_se": objective_se,
                            "diff_mean": diff_mean,
                            "diff_se": diff_se,
                            "baseline_mean": baseline_summary["objective_mean"],
                        }
                    )
    result = pd.DataFrame(rows)
    result["abs_diff"] = result["diff_mean"].abs()
    return result


def objective_value(
    table: pd.DataFrame,
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    kind: str = "weighted_pnl",
) -> float:
    score = robust.evaluate_table_on_fields(table, fields, fill_models)
    if kind == "weighted_pnl":
        return score.weighted_pnl
    if kind == "weighted_risk":
        return score.objective
    raise ValueError(f"Unknown objective kind: {kind}")


def finite_difference_gradient_report(
    table: pd.DataFrame,
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    epsilon: float = 1.0,
    objective_kind: str = "weighted_pnl",
) -> pd.DataFrame:
    base_value = objective_value(table, fields, fill_models, kind=objective_kind)
    rows = []

    def status_for(side: str, bid: float, ask: float) -> str:
        if side == "bid":
            if bid <= 0.0 + sim.TIE_TOL:
                return "lower_bound"
            if bid >= ask - sim.TIE_TOL:
                return "bid_ask_bound"
            if bid >= 1200.0 - sim.TIE_TOL:
                return "upper_bound"
            return "interior"
        if ask <= 0.0 + sim.TIE_TOL:
            return "lower_bound"
        if ask >= 1200.0 - sim.TIE_TOL:
            return "upper_bound"
        if ask <= bid + sim.TIE_TOL:
            return "bid_ask_bound"
        return "interior"

    for row_idx, signal in enumerate(table["signal"].to_numpy(dtype=int)):
        bid = float(table.loc[row_idx, "bid"])
        ask = float(table.loc[row_idx, "ask"])
        for side in ("bid", "ask"):
            base_quote = bid if side == "bid" else ask
            status = status_for(side, bid, ask)
            directional = {}
            for label, delta in (("plus", epsilon), ("minus", -epsilon)):
                candidate = table.copy(deep=True)
                target = float(np.clip(base_quote + delta, 0.0, 1200.0))
                candidate.loc[row_idx, side] = target
                cand_bid = float(candidate.loc[row_idx, "bid"])
                cand_ask = float(candidate.loc[row_idx, "ask"])
                valid = True
                if cand_bid > cand_ask + sim.TIE_TOL:
                    valid = False
                if valid:
                    try:
                        robust.validate_table(candidate)
                    except ValueError:
                        valid = False

                actual_delta = target - base_quote
                if valid and abs(actual_delta) > 1e-12:
                    value = objective_value(candidate, fields, fill_models, kind=objective_kind)
                    directional[label] = {
                        "valid": True,
                        "value": value,
                        "delta": actual_delta,
                        "grad": (value - base_value) / actual_delta,
                        "improvement": value - base_value,
                    }
                else:
                    directional[label] = {
                        "valid": False,
                        "value": np.nan,
                        "delta": actual_delta,
                        "grad": np.nan,
                        "improvement": np.nan,
                    }

            grad = np.nan
            if directional["plus"]["valid"] and directional["minus"]["valid"]:
                denom = directional["plus"]["delta"] - directional["minus"]["delta"]
                if abs(denom) > 1e-12:
                    grad = (directional["plus"]["value"] - directional["minus"]["value"]) / denom
            elif directional["plus"]["valid"]:
                grad = directional["plus"]["grad"]
            elif directional["minus"]["valid"]:
                grad = directional["minus"]["grad"]

            best_direction = ""
            best_improvement = 0.0
            for label in ("plus", "minus"):
                if directional[label]["valid"] and directional[label]["improvement"] > best_improvement:
                    best_improvement = directional[label]["improvement"]
                    best_direction = label

            rows.append(
                {
                    "signal": signal,
                    "side": side,
                    "quote": base_quote,
                    "status": status,
                    "grad": grad,
                    "abs_grad": abs(grad) if np.isfinite(grad) else np.nan,
                    "best_direction": best_direction,
                    "best_improvement": best_improvement,
                    "valid_plus": directional["plus"]["valid"],
                    "valid_minus": directional["minus"]["valid"],
                }
            )

    return pd.DataFrame(rows)


def adjust_bid_at_signal(table: pd.DataFrame, signal: int, new_bid: float) -> pd.DataFrame:
    adjusted = table.copy(deep=True)
    signal = int(signal)
    matches = np.where(adjusted["signal"].to_numpy(dtype=int) == signal)[0]
    if len(matches) == 0:
        raise ValueError(f"Signal {signal} not found in table")
    row_idx = int(matches[0])
    adjusted.loc[row_idx, "bid"] = float(np.clip(new_bid, 0.0, 1200.0))
    robust.validate_table(adjusted)
    return adjusted


def shift_table(
    table: pd.DataFrame,
    bid_shift: float = 0.0,
    ask_shift: float = 0.0,
    min_spread: float = 0.0,
) -> pd.DataFrame:
    shifted = table.copy(deep=True)
    bids = np.clip(shifted["bid"].to_numpy(dtype=float) + bid_shift, 0.0, 1200.0)
    asks = np.clip(shifted["ask"].to_numpy(dtype=float) + ask_shift, 0.0, 1200.0)
    crossed = bids > asks - min_spread
    if np.any(crossed):
        mid = (bids[crossed] + asks[crossed]) / 2.0
        bids[crossed] = mid - min_spread / 2.0
        asks[crossed] = mid + min_spread / 2.0
    shifted["bid"] = np.round(bids, 2)
    shifted["ask"] = np.round(asks, 2)
    robust.validate_table(shifted)
    return shifted


def summarize_single_scenario_metrics(
    table: pd.DataFrame,
    fields_by_seed: list[list[robust.ScenarioField]],
    fill_models: sim.FillModels,
) -> dict[str, float]:
    metrics_by_seed = {
        "pnl": [],
        "fills": [],
        "ask_win": [],
        "bid_win": [],
        "adv_ask_win": [],
        "adv_bid_win": [],
    }
    for fields in fields_by_seed:
        result = robust.evaluate_table_on_fields(table, fields, fill_models)
        row = result.metrics.iloc[0]
        metrics_by_seed["pnl"].append(float(row["pnl"]))
        metrics_by_seed["fills"].append(float(row["fills"]))
        metrics_by_seed["ask_win"].append(float(row["ask_win"]))
        metrics_by_seed["bid_win"].append(float(row["bid_win"]))
        metrics_by_seed["adv_ask_win"].append(float(row["adv_ask_win"]))
        metrics_by_seed["adv_bid_win"].append(float(row["adv_bid_win"]))

    summary = {}
    for key, values in metrics_by_seed.items():
        arr = np.asarray(values, dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        summary[key] = float(arr.mean())
        summary[f"{key}_se"] = se
    return summary


def crowding_sweep_report(
    table: pd.DataFrame,
    bot_models: list[robust.BotQuoteModel],
    fill_models: sim.FillModels,
    n_rounds: int,
    seeds: int,
    first_seed: int,
    clone_table: pd.DataFrame,
    max_clones: int = 10,
) -> pd.DataFrame:
    rows = []
    for clones in range(max_clones + 1):
        scenario_tables = {"clones": [clone_table for _ in range(clones)]}
        scenario_weights = {"clones": 1.0}
        fields_by_seed = build_fields_for_seeds(
            bot_models,
            n_rounds=n_rounds,
            seeds=seeds,
            first_seed=first_seed,
            scenario_tables=scenario_tables,
            scenario_weights=scenario_weights,
        )
        summary = summarize_single_scenario_metrics(table, fields_by_seed, fill_models)
        rows.append({"clones": clones, **summary})
    return pd.DataFrame(rows)


def random_param_table(
    rng: np.random.Generator,
    mid_offset_range: tuple[float, float] = (-40.0, 40.0),
    mid_tilt_range: tuple[float, float] = (-20.0, 20.0),
    h_bid_range: tuple[float, float] = (70.0, 150.0),
    h_ask_range: tuple[float, float] = (70.0, 150.0),
    h_tilt_range: tuple[float, float] = (-20.0, 20.0),
) -> pd.DataFrame:
    signal = sim.OFFICIAL_SIGNALS
    base_mid = robust.posterior_mean_uniform_normal(signal)
    axis = (signal - 500.0) / 500.0
    mid_offset = float(rng.uniform(*mid_offset_range))
    mid_tilt = float(rng.uniform(*mid_tilt_range))
    mid = base_mid + mid_offset + mid_tilt * axis

    h_bid_base = float(rng.uniform(*h_bid_range))
    h_ask_base = float(rng.uniform(*h_ask_range))
    h_bid_tilt = float(rng.uniform(*h_tilt_range))
    h_ask_tilt = float(rng.uniform(*h_tilt_range))
    h_bid = np.clip(h_bid_base + h_bid_tilt * axis, 40.0, 240.0)
    h_ask = np.clip(h_ask_base + h_ask_tilt * axis, 40.0, 240.0)

    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(np.clip(mid - h_bid, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(mid + h_ask, 0.0, 1200.0), 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def median_rank_random_fields(
    candidates: dict[str, pd.DataFrame],
    bot_models: list[robust.BotQuoteModel],
    fill_models: sim.FillModels,
    n_rounds: int,
    n_fields: int = 50,
    opponents_per_field: int = 6,
    seed: int = 20260609,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ranks_by_name = {name: [] for name in candidates}
    pnl_by_name = {name: [] for name in candidates}

    for field_id in range(n_fields):
        opponent_tables = [random_param_table(rng) for _ in range(opponents_per_field)]
        fields = robust.build_scenario_fields(
            bot_models,
            n_rounds=n_rounds,
            seed=seed + 1000 + field_id,
            scenario_tables={"random": opponent_tables},
            scenario_weights={"random": 1.0},
        )
        pnls = {}
        for name, table in candidates.items():
            result = robust.evaluate_table_on_fields(table, fields, fill_models)
            pnl = float(result.metrics.loc[result.metrics["scenario"] == "random", "pnl"].iloc[0])
            pnls[name] = pnl
            pnl_by_name[name].append(pnl)

        ranks = pd.Series(pnls).rank(ascending=False, method="average")
        for name, rank in ranks.items():
            ranks_by_name[name].append(float(rank))

    rows = []
    for name in candidates:
        ranks = np.asarray(ranks_by_name[name], dtype=float)
        pnls = np.asarray(pnl_by_name[name], dtype=float)
        rank_se = float(ranks.std(ddof=1) / np.sqrt(len(ranks))) if len(ranks) > 1 else 0.0
        rows.append(
            {
                "strategy": name,
                "median_rank": float(np.median(ranks)),
                "rank_mean": float(ranks.mean()),
                "rank_se": rank_se,
                "median_pnl": float(np.median(pnls)),
                "pnl_mean": float(pnls.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["median_rank", "pnl_mean"], ascending=[True, False])


def average_tables(tables: list[pd.DataFrame], min_spread: float = 5.0) -> pd.DataFrame:
    if not tables:
        raise ValueError("Need at least one table to average")
    signals = tables[0]["signal"].to_numpy(dtype=int)
    bids = np.mean([table["bid"].to_numpy(dtype=float) for table in tables], axis=0)
    asks = np.mean([table["ask"].to_numpy(dtype=float) for table in tables], axis=0)
    averaged = pd.DataFrame({"signal": signals, "bid": bids, "ask": asks})
    return enforce_table_shape(averaged, min_spread=min_spread)


def bot_dominance_report(df: pd.DataFrame) -> pd.DataFrame:
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    best_ask = bot_asks.min(axis=1)
    best_bid = bot_bids.max(axis=1)
    rows = []
    for idx, name in enumerate(sim.BOT_NAMES):
        rows.append(
            {
                "bot": name,
                "best_ask": float(np.mean(np.abs(bot_asks[:, idx] - best_ask) <= sim.TIE_TOL)),
                "best_bid": float(np.mean(np.abs(bot_bids[:, idx] - best_bid) <= sim.TIE_TOL)),
            }
        )
    return pd.DataFrame(rows)


def bot_dominance_by_block_report(df: pd.DataFrame, blocks: int = 5) -> pd.DataFrame:
    df_sorted = df.sort_values("period").reset_index(drop=True)
    rows = []
    for block_id, block in enumerate(np.array_split(df_sorted, blocks)):
        bot_asks = block.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
        bot_bids = block.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
        best_ask = bot_asks.min(axis=1)
        best_bid = bot_bids.max(axis=1)
        for idx, name in enumerate(sim.BOT_NAMES):
            rows.append(
                {
                    "block": block_id + 1,
                    "period_start": int(block["period"].iloc[0]),
                    "period_end": int(block["period"].iloc[-1]),
                    "bot": name,
                    "best_ask": float(np.mean(np.abs(bot_asks[:, idx] - best_ask) <= sim.TIE_TOL)),
                    "best_bid": float(np.mean(np.abs(bot_bids[:, idx] - best_bid) <= sim.TIE_TOL)),
                }
            )
    return pd.DataFrame(rows)


def nonlinear_fit_report(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    true_value = df["true_value"].to_numpy(dtype=float)
    rows = []
    for name in sim.BOT_NAMES:
        for side in ("bid", "ask"):
            y = df[f"bot_{name}_{side}"].to_numpy(dtype=float)
            coef_linear = np.polyfit(true_value, y, 1)
            pred_linear = np.polyval(coef_linear, true_value)
            rmse_linear = float(np.sqrt(np.mean((y - pred_linear) ** 2)))
            coef_poly = np.polyfit(true_value, y, degree)
            pred_poly = np.polyval(coef_poly, true_value)
            rmse_poly = float(np.sqrt(np.mean((y - pred_poly) ** 2)))
            rmse_gain = rmse_linear - rmse_poly
            gain_pct = rmse_gain / rmse_linear if rmse_linear > 0 else 0.0
            rows.append(
                {
                    "bot": name,
                    "side": side,
                    "rmse_linear": rmse_linear,
                    "rmse_poly": rmse_poly,
                    "rmse_gain": rmse_gain,
                    "rmse_gain_pct": gain_pct,
                    "poly_top_coef": float(coef_poly[0]),
                }
            )
    return pd.DataFrame(rows)


def bot_regime_block_report(df: pd.DataFrame, blocks: int = 5) -> pd.DataFrame:
    df_sorted = df.sort_values("period").reset_index(drop=True)
    rows = []
    for block_id, block in enumerate(np.array_split(df_sorted, blocks)):
        true_value = block["true_value"].to_numpy(dtype=float)
        design = np.column_stack([true_value, np.ones(len(block))])
        for name in sim.BOT_NAMES:
            bid = block[f"bot_{name}_bid"].to_numpy(dtype=float)
            ask = block[f"bot_{name}_ask"].to_numpy(dtype=float)
            mid = (bid + ask) / 2.0
            spread = ask - bid
            coef = np.linalg.lstsq(design, mid, rcond=None)[0]
            rows.append(
                {
                    "block": block_id + 1,
                    "period_start": int(block["period"].iloc[0]),
                    "period_end": int(block["period"].iloc[-1]),
                    "bot": name,
                    "mid_alpha": float(coef[0]),
                    "mid_intercept": float(coef[1]),
                    "spread_mean": float(spread.mean()),
                    "spread_std": float(spread.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def bot_regime_shift_summary(df: pd.DataFrame, blocks: int = 5) -> pd.DataFrame:
    block_report = bot_regime_block_report(df, blocks)
    rows = []
    for name, group in block_report.groupby("bot"):
        rows.append(
            {
                "bot": name,
                "mid_alpha_min": float(group["mid_alpha"].min()),
                "mid_alpha_max": float(group["mid_alpha"].max()),
                "mid_alpha_range": float(group["mid_alpha"].max() - group["mid_alpha"].min()),
                "mid_intercept_range": float(group["mid_intercept"].max() - group["mid_intercept"].min()),
                "spread_mean_range": float(group["spread_mean"].max() - group["spread_mean"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("mid_alpha_range", ascending=False)


def fit_asym_poly_params_from_table(
    table: pd.DataFrame,
    basis: str,
    degree_mid: int,
    degree_bid: int,
    degree_ask: int,
) -> AsymPolyParams:
    signal = table["signal"].to_numpy(dtype=float)
    base_mid = robust.posterior_mean_uniform_normal(signal)
    bid = table["bid"].to_numpy(dtype=float)
    ask = table["ask"].to_numpy(dtype=float)
    mid = (bid + ask) / 2.0
    design_mid = poly_basis(signal, degree_mid, basis)
    design_bid = poly_basis(signal, degree_bid, basis)
    design_ask = poly_basis(signal, degree_ask, basis)
    mid_coef = np.linalg.lstsq(design_mid, mid - base_mid, rcond=None)[0]
    h_bid_coef = np.linalg.lstsq(design_bid, mid - bid, rcond=None)[0]
    h_ask_coef = np.linalg.lstsq(design_ask, ask - mid, rcond=None)[0]
    return AsymPolyParams(
        basis=basis,
        degree_mid=degree_mid,
        degree_bid=degree_bid,
        degree_ask=degree_ask,
        mid_coef=mid_coef,
        h_bid_coef=h_bid_coef,
        h_ask_coef=h_ask_coef,
    )


def make_asym_poly_table(params: AsymPolyParams) -> pd.DataFrame:
    signal = sim.OFFICIAL_SIGNALS
    design_mid = poly_basis(signal, params.degree_mid, params.basis)
    design_bid = poly_basis(signal, params.degree_bid, params.basis)
    design_ask = poly_basis(signal, params.degree_ask, params.basis)
    base_mid = robust.posterior_mean_uniform_normal(signal)
    mid = base_mid + design_mid @ params.mid_coef
    h_bid = np.clip(design_bid @ params.h_bid_coef, 40.0, 240.0)
    h_ask = np.clip(design_ask @ params.h_ask_coef, 40.0, 240.0)
    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(np.clip(mid - h_bid, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(mid + h_ask, 0.0, 1200.0), 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def bounded_asym_poly_update(params: AsymPolyParams, array_name: str, index: int, delta: float) -> AsymPolyParams:
    candidate = params.copy()
    target = getattr(candidate, array_name)
    bound = 240.0 if index == 0 else 120.0
    target[index] = float(np.clip(target[index] + delta, -bound, bound))
    return candidate


def optimize_asym_poly_params(
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
    start_table: pd.DataFrame,
    basis: str,
    degree_mid: int,
    degree_bid: int,
    degree_ask: int,
    steps: list[float] | None = None,
    max_passes_per_step: int = 2,
) -> tuple[AsymPolyParams, pd.DataFrame, float, pd.DataFrame]:
    if steps is None:
        steps = [8.0, 4.0, 2.0, 1.0]
    params = fit_asym_poly_params_from_table(
        start_table,
        basis=basis,
        degree_mid=degree_mid,
        degree_bid=degree_bid,
        degree_ask=degree_ask,
    )
    table = make_asym_poly_table(params)
    best_value, best_metrics = evaluate_table_for_objective(table, fields, fill_models, objective)
    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for array_name, degree in (
                ("mid_coef", degree_mid),
                ("h_bid_coef", degree_bid),
                ("h_ask_coef", degree_ask),
            ):
                for index in range(degree + 1):
                    for direction in (1.0, -1.0):
                        candidate_params = bounded_asym_poly_update(
                            params,
                            array_name,
                            index,
                            direction * step,
                        )
                        try:
                            candidate_table = make_asym_poly_table(candidate_params)
                        except ValueError:
                            continue
                        candidate_value, candidate_metrics = evaluate_table_for_objective(
                            candidate_table,
                            fields,
                            fill_models,
                            objective,
                        )
                        if candidate_value > best_value + 1e-9:
                            params = candidate_params
                            table = candidate_table
                            best_value = candidate_value
                            best_metrics = candidate_metrics
                            improved = True
            if not improved:
                break
    return params, table, best_value, best_metrics


def gp_smooth_table(
    source: pd.DataFrame,
    length_scale: float = 80.0,
    noise_std: float = 4.0,
) -> pd.DataFrame:
    signal = source["signal"].to_numpy(dtype=float)
    distance2 = (signal[:, None] - signal[None, :]) ** 2
    kernel = np.exp(-0.5 * distance2 / (length_scale * length_scale))
    regularized = kernel + (noise_std * noise_std / 1000.0) * np.eye(len(signal))
    weights_bid = np.linalg.solve(regularized, source["bid"].to_numpy(dtype=float))
    weights_ask = np.linalg.solve(regularized, source["ask"].to_numpy(dtype=float))
    pred_kernel = kernel
    table = pd.DataFrame(
        {
            "signal": signal.astype(int),
            "bid": np.round(pred_kernel @ weights_bid, 2),
            "ask": np.round(pred_kernel @ weights_ask, 2),
        }
    )
    return enforce_table_shape(table, min_spread=5.0)


def evaluate_table_for_objective(
    table: pd.DataFrame,
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
) -> tuple[float, pd.DataFrame]:
    score = robust.evaluate_table_on_fields(table, fields, fill_models)
    return objective(score.metrics), score.metrics


def optimize_for_objective(
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    objective: ObjectiveFn,
    start: robust.SmoothParams,
    steps: list[float],
    max_passes_per_step: int,
) -> tuple[robust.SmoothParams, pd.DataFrame, float, pd.DataFrame]:
    params = start
    table = robust.make_smooth_table(params)
    best_value, best_metrics = evaluate_table_for_objective(table, fields, fill_models, objective)

    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for array_name in ("mid_shift", "h_bid", "h_ask"):
                for index in range(len(robust.KNOT_SIGNALS)):
                    for direction in (1.0, -1.0):
                        candidate_params = robust.bounded_update(params, array_name, index, direction * step)
                        try:
                            candidate_table = robust.make_smooth_table(candidate_params)
                        except ValueError:
                            continue
                        candidate_value, candidate_metrics = evaluate_table_for_objective(
                            candidate_table,
                            fields,
                            fill_models,
                            objective,
                        )
                        if candidate_value > best_value + 1e-9:
                            params = candidate_params
                            table = candidate_table
                            best_value = candidate_value
                            best_metrics = candidate_metrics
                            improved = True
            if not improved:
                break

    return params, table, best_value, best_metrics


def summarize_objective_candidates(
    tables: dict[str, pd.DataFrame],
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, table in tables.items():
        result = robust.evaluate_table_on_fields(table, fields, fill_models)
        metrics = result.metrics
        rows.append(
            {
                "strategy": name,
                "weighted_risk": weighted_risk_objective(metrics),
                "minimax": minimax_objective(metrics),
                "cvar_bottom3": cvar_bottom3_objective(metrics),
                "weighted_pnl": result.weighted_pnl,
                "hist_pnl": robust.historical_replay_pnl(table, df, fill_models),
                "no_clone": float(metrics.loc[metrics["scenario"] == "no_clones", "pnl"].iloc[0]),
                "coord_x3": float(metrics.loc[metrics["scenario"] == "coord_x3", "pnl"].iloc[0]),
                "coord_x5": float(metrics.loc[metrics["scenario"] == "coord_x5", "pnl"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_risk", ascending=False)


def game_theory_proxy_report(
    tables: dict[str, pd.DataFrame],
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
) -> pd.DataFrame:
    """Nash/Stackelberg/competition proxies using the modeled clone scenarios."""
    rows = []
    for name, table in tables.items():
        metrics = robust.evaluate_table_on_fields(table, fields, fill_models).metrics
        by_name = {row["scenario"]: row for row in metrics.to_dict("records")}
        no_clone = float(by_name["no_clones"]["pnl"])
        coord_x1 = float(by_name["coord_x1"]["pnl"])
        coord_x3 = float(by_name["coord_x3"]["pnl"])
        coord_x5 = float(by_name["coord_x5"]["pnl"])
        fresh_x3 = float(by_name["fresh_x3"]["pnl"])
        mixed = float(by_name["mixed_coord2_fresh2"]["pnl"])
        rows.append(
            {
                "strategy": name,
                "stackelberg_no_clone": no_clone,
                "coord_x1": coord_x1,
                "nash_floor_proxy": min(coord_x3, coord_x5, fresh_x3, mixed),
                "crowding_loss_x5": no_clone - coord_x5,
                "differentiated_field": mixed,
            }
        )
    return pd.DataFrame(rows).sort_values("nash_floor_proxy", ascending=False)


def round_pnl_on_field(
    table: pd.DataFrame,
    field: robust.ScenarioField,
    fill_models: sim.FillModels,
) -> np.ndarray:
    bid, ask = sim.table_quotes(field.our_signal, table)
    ask_share, _ = robust.tie_aware_share(ask, field.best_ask, field.best_ask_count, side="ask")
    bid_share, _ = robust.tie_aware_share(bid, field.best_bid, field.best_bid_count, side="bid")
    ask_edges = np.maximum(field.best_ask - ask, 0.0)
    bid_edges = np.maximum(bid - field.best_bid, 0.0)
    adverse_ask = field.true_value > ask
    adverse_bid = field.true_value < bid
    buys = ask_share * fill_models.ask.predict(ask_edges, adverse_ask)
    sells = bid_share * fill_models.bid.predict(bid_edges, adverse_bid)
    return buys * (ask - field.true_value) + sells * (field.true_value - bid)


def risk_theory_report(
    tables: dict[str, pd.DataFrame],
    fields: list[robust.ScenarioField],
    fill_models: sim.FillModels,
    max_tables: int = 12,
) -> pd.DataFrame:
    rows = []
    for idx, (name, table) in enumerate(tables.items()):
        if idx >= max_tables:
            break
        scenario_means = []
        weighted_losses = []
        for field in fields:
            pnl = round_pnl_on_field(table, field, fill_models)
            scenario_means.append(float(pnl.mean()))
            sample = -pnl
            weighted_losses.append(sample)
        combined_loss = np.concatenate(weighted_losses)
        combined_pnl = -combined_loss
        mean = float(combined_pnl.mean())
        std = float(combined_pnl.std(ddof=1))
        tail_cut = float(np.quantile(combined_loss, 0.95))
        tail_excess = combined_loss[combined_loss > tail_cut] - tail_cut
        if len(tail_excess) >= 25:
            shape, _, scale = stats.genpareto.fit(tail_excess, floc=0.0)
            evt_99_loss = tail_cut + float(stats.genpareto.ppf(0.80, shape, loc=0.0, scale=scale))
        else:
            shape = np.nan
            evt_99_loss = float(np.quantile(combined_loss, 0.99))
        cvar_5 = float(combined_pnl[combined_pnl <= np.quantile(combined_pnl, 0.05)].mean())
        n_rounds = 5000
        ruin_z = (0.0 - mean * n_rounds) / max(std * np.sqrt(n_rounds), 1e-9)
        ruin_prob = float(stats.norm.cdf(ruin_z))
        lil_bound = std * np.sqrt(2.0 * n_rounds * np.log(np.log(n_rounds)))
        rows.append(
            {
                "strategy": name,
                "round_mean": mean,
                "round_std": std,
                "cvar_5pct": cvar_5,
                "evt_shape": shape,
                "evt_99_loss": evt_99_loss,
                "ruin_prob_5000": ruin_prob,
                "lil_5000_bound": lil_bound,
                "worst_scenario_mean": float(np.min(scenario_means)),
            }
        )
    return pd.DataFrame(rows).sort_values("ruin_prob_5000")


def paired_objective_differences(raw: pd.DataFrame, baseline: str) -> pd.DataFrame:
    wide = raw.pivot(index="seed", columns="strategy", values="objective")
    rows = []
    if baseline not in wide:
        return pd.DataFrame(rows)
    for challenger in wide.columns:
        if challenger == baseline:
            continue
        diff = wide[challenger] - wide[baseline]
        rows.append(
            {
                "challenger": challenger,
                "baseline": baseline,
                "mean_diff": float(diff.mean()),
                "se_diff": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def halton_normal(size: int, dim: int, seed: int) -> np.ndarray:
    sampler = qmc.Halton(d=dim, scramble=True, seed=seed)
    u = sampler.random(size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(u)


def variance_reduction_report(
    table: pd.DataFrame,
    df: pd.DataFrame,
    bot_models: list[robust.BotQuoteModel],
    fill_models: sim.FillModels,
    n_rounds: int,
    replications: int = 8,
    first_seed: int = 9000,
) -> pd.DataFrame:
    rows = []
    for method in ("plain_mc", "antithetic_signal", "halton_signal"):
        estimates = []
        for offset in range(replications):
            seed = first_seed + offset
            fields = robust.build_scenario_fields(
                bot_models,
                n_rounds=n_rounds,
                seed=seed,
                scenario_tables=robust.scenario_tables(),
            )
            if method != "plain_mc":
                # Keep the same simulated competitor field and only variance-reduce our private signal.
                for idx, field in enumerate(fields):
                    if method == "antithetic_signal":
                        rng = np.random.default_rng(seed + 10_000 + idx)
                        half = (n_rounds + 1) // 2
                        noise = rng.normal(0.0, 50.0, half)
                        signal = field.true_value + np.r_[noise, -noise][:n_rounds]
                    else:
                        z = halton_normal(n_rounds, 1, seed + 20_000 + idx)[:, 0]
                        signal = field.true_value + 50.0 * z
                    fields[idx] = robust.ScenarioField(
                        name=field.name,
                        weight=field.weight,
                        true_value=field.true_value,
                        our_signal=signal,
                        best_bid=field.best_bid,
                        best_ask=field.best_ask,
                        best_bid_count=field.best_bid_count,
                        best_ask_count=field.best_ask_count,
                    )
            estimates.append(robust.evaluate_table_on_fields(table, fields, fill_models).objective)
        values = np.asarray(estimates, dtype=float)
        rows.append(
            {
                "method": method,
                "objective_mean": float(values.mean()),
                "objective_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                "se_vs_plain": np.nan,
            }
        )
    plain_se = rows[0]["objective_se"]
    for row in rows:
        row["se_vs_plain"] = row["objective_se"] / plain_se if plain_se > 0 else np.nan
    return pd.DataFrame(rows)


def compact_table_format(df: pd.DataFrame) -> str:
    display = df.copy()
    percent_cols = [
        column
        for column in display.columns
        if column.endswith("best")
        or column.endswith("prob")
        or column.startswith("lambda")
        or column.endswith("win")
        or column.endswith("reduction")
        or column == "coverage"
        or column == "variance_reduction"
        or column == "mse_reduction"
        or column == "ruin_prob_5000"
    ]
    for column in display.columns:
        if column in {"strategy", "scenario", "side", "bot"}:
            continue
        if column in percent_cols:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    return display.to_string(index=False)


def print_block(title: str, content: str) -> None:
    print()
    print(title)
    print("=" * len(title))
    print(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced math experiments for strategy selection.")
    parser.add_argument("--csv", type=Path, default=Path("auction_history.csv"))
    parser.add_argument("--search-n", type=int, default=70_000)
    parser.add_argument("--eval-n", type=int, default=100_000)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--search-seed", type=int, default=20260518)
    parser.add_argument("--eval-seed", type=int, default=20260618)
    parser.add_argument(
        "--run-next-steps",
        action="store_true",
        help="Run the sensitivity audit, Chebyshev deg5/6, asymmetric degrees, crowding sweep, CSV diagnostics, and ensemble hedge checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_next_steps = bool(args.run_next_steps)
    df = sim.load_history(args.csv)
    fill_models = sim.build_fill_models(df)
    bot_models = robust.fit_bot_quote_models(df)
    fields = robust.build_scenario_fields(
        bot_models,
        n_rounds=args.search_n,
        seed=args.search_seed,
        scenario_tables=robust.scenario_tables(),
    )

    print_block("Posterior boundary correction", compact_table_format(posterior_boundary_report()))
    print_block("Information-theoretic signal limits", compact_table_format(information_theory_report()))
    print_block("KL update by submitted signal bucket", compact_table_format(posterior_kl_report()))
    print_block("Rate-distortion knot diagnostics", compact_table_format(rate_distortion_report()))
    print_block("Direct competitive quote regressions", compact_table_format(competitive_regression_report(df)))
    print_block("Huber robust quote regressions", compact_table_format(robust_regression_report(df)))
    print_block("Conformal competitor quote intervals", compact_table_format(conformal_competitor_report(df)))
    print_block("Glosten-Milgrom adverse-selection proxy", compact_table_format(gm_adverse_selection_report(df)))
    print_block("PIN / Kyle / Roll / Amihud proxies", compact_table_format(microstructure_risk_report(df)))

    tables = robust.candidate_tables()
    tables["robust_weighted_prior"] = robust_weighted_prior_table()
    tables["previous_best_refined"] = previous_best_table()
    tables["avellaneda_stoikov"] = avellaneda_stoikov_table(fill_models)
    tables["gp_smooth_previous"] = gp_smooth_table(tables["previous_best_refined"])
    tables["kde_isotonic"] = kde_local_best_response_table(df, fill_models)

    print_block(
        "Baseline table comparison on advanced search field",
        compact_table_format(summarize_objective_candidates(tables, fields, fill_models, df)),
    )
    print_block(
        "Mechanism-design / game-theory competition proxies",
        compact_table_format(game_theory_proxy_report(tables, fields, fill_models)),
    )

    optimized_tables: dict[str, pd.DataFrame] = {}
    params_rows = []
    for spec in OBJECTIVES:
        best_value = -np.inf
        best_table: pd.DataFrame | None = None
        best_params: robust.SmoothParams | None = None
        for start_name in ("fresh_linear", "posterior"):
            params, table, value, _ = optimize_for_objective(
                fields=fields,
                fill_models=fill_models,
                objective=spec.objective,
                start=robust.initial_params(start_name),
                steps=[20.0, 10.0, 5.0, 2.0, 1.0],
                max_passes_per_step=3,
            )
            print(f"{spec.name} from {start_name}: objective={value:.4f}")
            if value > best_value:
                best_value = value
                best_table = table
                best_params = params
        if best_table is None or best_params is None:
            raise RuntimeError(f"No table optimized for {spec.name}")
        optimized_tables[f"opt_{spec.name}"] = best_table
        for idx, signal in enumerate(robust.KNOT_SIGNALS.astype(int)):
            params_rows.append(
                {
                    "objective": spec.name,
                    "signal": signal,
                    "mid_shift": best_params.mid_shift[idx],
                    "h_bid": best_params.h_bid[idx],
                        "h_ask": best_params.h_ask[idx],
                    }
                )

    poly_rows = []
    cheb_degrees = (4, 5, 6) if run_next_steps else (4,)
    legendre_degrees = (4, 5, 6) if run_next_steps else (4,)
    for basis in ("chebyshev", "legendre"):
        degrees = cheb_degrees if basis == "chebyshev" else legendre_degrees
        for degree in degrees:
            params, table, value, _ = optimize_poly_params(
                fields=fields,
                fill_models=fill_models,
                objective=weighted_risk_objective,
                start_table=tables["previous_best_refined"],
                basis=basis,
                degree=degree,
            )
            name = f"poly_{basis}_deg{degree}"
            optimized_tables[name] = table
            print(f"{name}: weighted_risk={value:.4f}")
            for idx, coef in enumerate(params.mid_coef):
                poly_rows.append({"basis": basis, "degree": degree, "term": f"mid_{idx}", "coef": coef})
            for idx, coef in enumerate(params.h_bid_coef):
                poly_rows.append({"basis": basis, "degree": degree, "term": f"h_bid_{idx}", "coef": coef})
            for idx, coef in enumerate(params.h_ask_coef):
                poly_rows.append({"basis": basis, "degree": degree, "term": f"h_ask_{idx}", "coef": coef})

    asym_rows = []
    if run_next_steps:
        asym_specs = [
            ("poly_chebyshev_asym_m4_b3_a5", 4, 3, 5),
            ("poly_chebyshev_asym_m4_b3_a6", 4, 3, 6),
        ]
        for name, degree_mid, degree_bid, degree_ask in asym_specs:
            params, table, value, _ = optimize_asym_poly_params(
                fields=fields,
                fill_models=fill_models,
                objective=weighted_risk_objective,
                start_table=tables["previous_best_refined"],
                basis="chebyshev",
                degree_mid=degree_mid,
                degree_bid=degree_bid,
                degree_ask=degree_ask,
            )
            optimized_tables[name] = table
            print(f"{name}: weighted_risk={value:.4f}")
            for idx, coef in enumerate(params.mid_coef):
                asym_rows.append(
                    {
                        "variant": name,
                        "term": f"mid_{idx}",
                        "coef": coef,
                    }
                )
            for idx, coef in enumerate(params.h_bid_coef):
                asym_rows.append(
                    {
                        "variant": name,
                        "term": f"h_bid_{idx}",
                        "coef": coef,
                    }
                )
            for idx, coef in enumerate(params.h_ask_coef):
                asym_rows.append(
                    {
                        "variant": name,
                        "term": f"h_ask_{idx}",
                        "coef": coef,
                    }
                )

    consensus = consensus_table_from_fields(fields, bandwidth=60.0)
    base_legendre = optimized_tables.get("poly_legendre_deg4")
    if base_legendre is not None:
        optimized_tables["legendre4_boundary_undercut_exclusive"] = boundary_micro_undercut_table(
            base_legendre,
            consensus,
            include_edges=False,
            low_cutoff=150.0,
            high_cutoff=850.0,
            eps_bid=1.0,
            eps_ask=1.0,
        )
        optimized_tables["legendre4_boundary_undercut_inclusive"] = boundary_micro_undercut_table(
            base_legendre,
            consensus,
            include_edges=True,
            low_cutoff=150.0,
            high_cutoff=850.0,
            eps_bid=1.0,
            eps_ask=1.0,
        )

    all_tables = {**tables, **optimized_tables}
    print_block(
        "Optimized objective comparison on search field",
        compact_table_format(summarize_objective_candidates(all_tables, fields, fill_models, df)),
    )
    print_block("Optimized smooth parameter knots", compact_table_format(pd.DataFrame(params_rows)))
    print_block("Optimized polynomial coefficients", compact_table_format(pd.DataFrame(poly_rows)))
    if run_next_steps and asym_rows:
        print_block("Asymmetric polynomial coefficients", compact_table_format(pd.DataFrame(asym_rows)))

    risk_tables = {
        name: all_tables[name]
        for name in (
            "fresh_linear_990_0_h95",
            "robust_weighted_prior",
            "previous_best_refined",
            "avellaneda_stoikov",
            "gp_smooth_previous",
            "kde_isotonic",
            *optimized_tables.keys(),
        )
        if name in all_tables
    }
    print_block(
        "Probability / tail-risk diagnostics",
        compact_table_format(risk_theory_report(risk_tables, fields, fill_models)),
    )

    for name in ("fresh_linear_990_0_h95", "robust_weighted_prior", "previous_best_refined", *optimized_tables.keys()):
        decomp = fill_decomposition(all_tables[name], fields, fill_models, scenario="no_clones")
        print_block(f"Fill-rate x edge decomposition: {name}", compact_table_format(decomp))

    finalists = {
        "fresh_linear_990_0_h95": all_tables["fresh_linear_990_0_h95"],
        "robust_weighted_prior": all_tables["robust_weighted_prior"],
        "previous_best_refined": all_tables["previous_best_refined"],
        "avellaneda_stoikov": all_tables["avellaneda_stoikov"],
        "gp_smooth_previous": all_tables["gp_smooth_previous"],
        "kde_isotonic": all_tables["kde_isotonic"],
        **optimized_tables,
    }
    print_block(
        "Variance-reduction check on previous best",
        compact_table_format(
            variance_reduction_report(
                finalists["previous_best_refined"],
                df=df,
                bot_models=bot_models,
                fill_models=fill_models,
                n_rounds=max(5_000, args.eval_n // 4),
            )
        ),
    )

    raw, seed_summary = robust.evaluate_across_seeds(
        finalists,
        df=df,
        bot_models=bot_models,
        fill_models=fill_models,
        n_rounds=args.eval_n,
        seeds=args.eval_seeds,
        first_seed=args.eval_seed,
    )
    print_block(
        f"Paired final evaluation across {args.eval_seeds} seeds x {args.eval_n:,} rounds",
        compact_table_format(seed_summary),
    )
    print_block(
        "Paired objective differences vs previous best",
        compact_table_format(paired_objective_differences(raw, "previous_best_refined")),
    )

    best_name = str(seed_summary.iloc[0]["strategy"])
    best_table = finalists[best_name]
    robust.validate_table(best_table)

    if run_next_steps:
        base_fields_by_seed = build_fields_for_seeds(
            bot_models,
            n_rounds=args.eval_n,
            seeds=args.eval_seeds,
            first_seed=args.eval_seed,
            scenario_tables=robust.scenario_tables(),
        )
        sensitivity = sensitivity_audit_table(
            best_table,
            base_fields_by_seed,
            fill_models,
            weighted_risk_objective,
            deltas=[1.0, 3.0, 5.0],
        )
        top_sensitivity = sensitivity[sensitivity["valid"]].sort_values("abs_diff", ascending=False).head(20)
        top_sensitivity = top_sensitivity.copy()
        top_sensitivity["valid"] = top_sensitivity["valid"].map(lambda value: "yes" if value else "no")
        top_sensitivity["clipped"] = top_sensitivity["clipped"].map(lambda value: "yes" if value else "no")
        print_block("Sensitivity audit (top 20 absolute objective deltas)", compact_table_format(top_sensitivity))

        kkt_table = optimized_tables.get("poly_legendre_deg4", best_table)
        grad_report = finite_difference_gradient_report(
            kkt_table,
            fields,
            fill_models,
            epsilon=1.0,
            objective_kind="weighted_pnl",
        )
        grad_valid = grad_report[grad_report["valid_plus"] | grad_report["valid_minus"]]
        top_grad = grad_valid.sort_values("abs_grad", ascending=False).head(20)
        print_block("KKT gradient check (weighted pnl, top 20 abs grads)", compact_table_format(top_grad))

        bid_zero_rows = []
        base_bid0 = float(kkt_table.loc[kkt_table["signal"] == 0, "bid"].iloc[0])
        for bid0 in (base_bid0, 35.0, 36.0, 38.0):
            patched = adjust_bid_at_signal(kkt_table, 0, bid0)
            score = robust.evaluate_table_on_fields(patched, fields, fill_models)
            bid_zero_rows.append(
                {
                    "bid0": bid0,
                    "objective": score.objective,
                    "weighted_pnl": score.weighted_pnl,
                    "scenario_std": score.scenario_std,
                }
            )
        print_block("Bid-at-zero sweep (Legendre deg4)", compact_table_format(pd.DataFrame(bid_zero_rows)))

        crowding = crowding_sweep_report(
            best_table,
            bot_models,
            fill_models,
            n_rounds=args.eval_n,
            seeds=args.eval_seeds,
            first_seed=args.eval_seed,
            clone_table=best_table,
            max_clones=10,
        )
        print_block("Crowding sweep vs Chebyshev-like clones", compact_table_format(crowding))

        rank_candidates = {
            name: all_tables[name]
            for name in (
                "poly_legendre_deg4",
                "poly_legendre_deg5",
                "poly_legendre_deg6",
                "poly_chebyshev_deg4",
                "poly_chebyshev_deg5",
                "poly_chebyshev_deg6",
                "previous_best_refined",
                "fresh_linear_990_0_h95",
            )
            if name in all_tables
        }
        if len(rank_candidates) >= 2:
            rank_report = median_rank_random_fields(
                rank_candidates,
                bot_models,
                fill_models,
                n_rounds=min(50_000, args.eval_n),
                n_fields=50,
                opponents_per_field=6,
                seed=args.eval_seed + 300,
            )
            print_block("Median rank vs random opponent fields", compact_table_format(rank_report))

        counter_table = shift_table(best_table, bid_shift=1.0, ask_shift=-1.0, min_spread=0.0)
        counter_fields = build_fields_for_seeds(
            bot_models,
            n_rounds=args.eval_n,
            seeds=args.eval_seeds,
            first_seed=args.eval_seed,
            scenario_tables={"counter": [counter_table]},
            scenario_weights={"counter": 1.0},
        )
        counter_summary = summarize_single_scenario_metrics(best_table, counter_fields, fill_models)
        print_block(
            "Smart-opponent stress test (bid+1/ask-1)",
            compact_table_format(pd.DataFrame([{"scenario": "counter", **counter_summary}])),
        )

        print_block("Bot dominance (overall)", compact_table_format(bot_dominance_report(df)))
        print_block("Bot dominance by block", compact_table_format(bot_dominance_by_block_report(df)))
        print_block("Nonlinear quote fit (quadratic vs linear)", compact_table_format(nonlinear_fit_report(df)))
        print_block("Regime shift summary (mid + spread)", compact_table_format(bot_regime_shift_summary(df)))

        top_names = [name for name in seed_summary["strategy"].head(5) if name in finalists]
        ensemble_tables = {}
        if len(top_names) >= 3:
            ensemble_tables["ensemble_top3"] = average_tables([finalists[name] for name in top_names[:3]])
        if len(top_names) >= 5:
            ensemble_tables["ensemble_top5"] = average_tables([finalists[name] for name in top_names[:5]])
        if ensemble_tables:
            ensemble_candidates = {name: finalists[name] for name in top_names}
            ensemble_candidates.update(ensemble_tables)
            _, ensemble_summary = robust.evaluate_across_seeds(
                ensemble_candidates,
                df=df,
                bot_models=bot_models,
                fill_models=fill_models,
                n_rounds=args.eval_n,
                seeds=args.eval_seeds,
                first_seed=args.eval_seed,
            )
            print_block("Ensemble hedge evaluation", compact_table_format(ensemble_summary))

    print_block(f"Recommended table by advanced pass: {best_name}", best_table.to_string(index=False))
    print_block("Copy-paste rows", best_table.to_csv(index=False, sep="\t", lineterminator="\n").strip())


if __name__ == "__main__":
    main()
