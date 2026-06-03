"""Robust final-score strategy search for Round 2.

This script optimizes for fresh simulated scoring rounds instead of only the
1,000 historical replay rows. It leaves historical_simulator.py unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import erf
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import historical_simulator as sim


KNOT_SIGNALS = np.array([0.0, 100.0, 200.0, 400.0, 600.0, 800.0, 900.0, 1000.0])
SEARCH_STEPS = [20.0, 10.0, 5.0, 2.0, 1.0]
SCENARIO_WEIGHTS = {
    "no_clones": 0.35,
    "coord_x1": 0.15,
    "coord_x3": 0.20,
    "coord_x5": 0.10,
    "fresh_x3": 0.10,
    "mixed_coord2_fresh2": 0.10,
}
MID_SHIFT_BOUNDS = (-120.0, 120.0)
HALF_SPREAD_BOUNDS = (40.0, 240.0)
OBJECTIVE_RISK_PENALTY = 0.5
DEFAULT_HISTORY_CSV = Path(__file__).with_name("auction_history.csv")


PUBLISHED_COORD_TABLE = pd.DataFrame(
    {
        "signal": sim.OFFICIAL_SIGNALS.astype(int),
        "bid": [
            1.00,
            4.00,
            11.00,
            90.75,
            116.50,
            143.25,
            204.00,
            243.75,
            330.50,
            312.25,
            425.00,
            419.75,
            500.50,
            583.25,
            595.00,
            663.75,
            667.50,
            731.25,
            782.00,
            778.75,
            869.50,
        ],
        "ask": [
            170.50,
            244.25,
            165.00,
            271.75,
            338.50,
            357.25,
            464.00,
            404.75,
            509.50,
            556.25,
            608.00,
            678.75,
            622.50,
            800.25,
            804.00,
            891.75,
            907.50,
            935.25,
            982.00,
            1022.75,
            1014.50,
        ],
    }
)


@dataclass(frozen=True)
class BotQuoteModel:
    name: str
    mid_coef: np.ndarray
    spread_mean: float
    residuals: np.ndarray


@dataclass(frozen=True)
class ScenarioField:
    name: str
    weight: float
    true_value: np.ndarray
    our_signal: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    best_bid_count: np.ndarray
    best_ask_count: np.ndarray


@dataclass(frozen=True)
class ScoreResult:
    objective: float
    weighted_pnl: float
    scenario_std: float
    metrics: pd.DataFrame


@dataclass(frozen=True)
class SmoothParams:
    mid_shift: np.ndarray
    h_bid: np.ndarray
    h_ask: np.ndarray

    def copy(self) -> "SmoothParams":
        return SmoothParams(self.mid_shift.copy(), self.h_bid.copy(), self.h_ask.copy())


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def posterior_mean_uniform_normal(signals: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """E[V | signal] for V~Uniform(0,1000), signal=V+N(0,sigma^2)."""
    signals = np.asarray(signals, dtype=float)
    lower = (0.0 - signals) / sigma
    upper = (1000.0 - signals) / sigma
    mass = norm_cdf(upper) - norm_cdf(lower)
    return signals + sigma * (norm_pdf(lower) - norm_pdf(upper)) / mass


def validate_table(table: pd.DataFrame, require_monotone: bool = False) -> None:
    if len(table) != len(sim.OFFICIAL_SIGNALS):
        raise ValueError("Table must have exactly 21 rows")
    if not np.allclose(table["signal"].to_numpy(dtype=float), sim.OFFICIAL_SIGNALS):
        raise ValueError("Table signals must be 0,50,...,1000")
    bid = table["bid"].to_numpy(dtype=float)
    ask = table["ask"].to_numpy(dtype=float)
    sim.validate_quotes(bid, ask)
    if require_monotone:
        if np.any(np.diff(bid) < -sim.TIE_TOL):
            raise ValueError("Generated bid table is not monotone nondecreasing")
        if np.any(np.diff(ask) < -sim.TIE_TOL):
            raise ValueError("Generated ask table is not monotone nondecreasing")


def formula_table(alpha: float, intercept: float, half_spread: float) -> pd.DataFrame:
    return sim.generate_formula_table(alpha, intercept, half_spread)


def posterior_table(offset: float, half_spread: float) -> pd.DataFrame:
    mid = posterior_mean_uniform_normal(sim.OFFICIAL_SIGNALS) + offset
    return pd.DataFrame(
        {
            "signal": sim.OFFICIAL_SIGNALS.astype(int),
            "bid": np.round(np.clip(mid - half_spread, 0.0, 1200.0), 2),
            "ask": np.round(np.clip(mid + half_spread, 0.0, 1200.0), 2),
        }
    )


def fit_bot_quote_models(df: pd.DataFrame) -> list[BotQuoteModel]:
    true_value = df["true_value"].to_numpy(dtype=float)
    design = np.column_stack([true_value, np.ones(len(df))])
    models: list[BotQuoteModel] = []
    for name in sim.BOT_NAMES:
        bid = df[f"bot_{name}_bid"].to_numpy(dtype=float)
        ask = df[f"bot_{name}_ask"].to_numpy(dtype=float)
        mid = (bid + ask) / 2.0
        spread = ask - bid
        mid_coef = np.linalg.lstsq(design, mid, rcond=None)[0]
        residuals = np.column_stack([mid - design @ mid_coef, spread - spread.mean()])
        models.append(
            BotQuoteModel(
                name=name,
                mid_coef=mid_coef,
                spread_mean=float(spread.mean()),
                residuals=residuals,
            )
        )
    return models


def simulate_bot_quotes(
    true_value: np.ndarray,
    bot_models: list[BotQuoteModel],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bids: list[np.ndarray] = []
    asks: list[np.ndarray] = []
    n_rounds = len(true_value)
    for model in bot_models:
        residual_ids = rng.integers(0, len(model.residuals), n_rounds)
        residuals = model.residuals[residual_ids]
        mid = model.mid_coef[0] * true_value + model.mid_coef[1] + residuals[:, 0]
        spread = np.maximum(0.0, model.spread_mean + residuals[:, 1])
        bids.append(np.clip(mid - spread / 2.0, 0.0, 1200.0))
        asks.append(np.clip(mid + spread / 2.0, 0.0, 1200.0))
    return np.column_stack(bids), np.column_stack(asks)


def build_scenario_fields(
    bot_models: list[BotQuoteModel],
    n_rounds: int,
    seed: int,
    scenario_tables: dict[str, list[pd.DataFrame]] | None = None,
    scenario_weights: dict[str, float] = SCENARIO_WEIGHTS,
) -> list[ScenarioField]:
    rng = np.random.default_rng(seed)
    true_value = rng.uniform(0.0, 1000.0, n_rounds)
    our_signal = true_value + rng.normal(0.0, 50.0, n_rounds)
    bot_bids, bot_asks = simulate_bot_quotes(true_value, bot_models, rng)
    if scenario_tables is None:
        scenario_tables = _default_scenario_tables()

    fields: list[ScenarioField] = []
    for name, extra_tables in scenario_tables.items():
        bid_columns = [bot_bids]
        ask_columns = [bot_asks]
        for table in extra_tables:
            extra_signal = true_value + rng.normal(0.0, 50.0, n_rounds)
            extra_bid, extra_ask = sim.table_quotes(extra_signal, table)
            bid_columns.append(extra_bid[:, None])
            ask_columns.append(extra_ask[:, None])
        all_bids = np.column_stack(bid_columns)
        all_asks = np.column_stack(ask_columns)
        best_bid = all_bids.max(axis=1)
        best_ask = all_asks.min(axis=1)
        best_bid_count = np.sum(np.abs(all_bids - best_bid[:, None]) <= sim.TIE_TOL, axis=1)
        best_ask_count = np.sum(np.abs(all_asks - best_ask[:, None]) <= sim.TIE_TOL, axis=1)
        fields.append(
            ScenarioField(
                name=name,
                weight=scenario_weights[name],
                true_value=true_value,
                our_signal=our_signal,
                best_bid=best_bid,
                best_ask=best_ask,
                best_bid_count=best_bid_count.astype(int),
                best_ask_count=best_ask_count.astype(int),
            )
        )
    return fields


def _default_scenario_tables() -> dict[str, list[pd.DataFrame]]:
    fresh = formula_table(0.990, 0.0, 95.0)
    coord = PUBLISHED_COORD_TABLE
    return {
        "no_clones": [],
        "coord_x1": [coord],
        "coord_x3": [coord, coord, coord],
        "coord_x5": [coord, coord, coord, coord, coord],
        "fresh_x3": [fresh, fresh, fresh],
        "mixed_coord2_fresh2": [coord, coord, fresh, fresh],
    }


def scenario_tables() -> dict[str, list[pd.DataFrame]]:
    return _default_scenario_tables()


def tie_aware_share(
    candidate: np.ndarray,
    best: np.ndarray,
    best_count: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=float)
    best = np.asarray(best, dtype=float)
    count = np.asarray(best_count, dtype=float)
    if side == "ask":
        won = candidate <= best + sim.TIE_TOL
        strict = candidate < best - sim.TIE_TOL
    elif side == "bid":
        won = candidate >= best - sim.TIE_TOL
        strict = candidate > best + sim.TIE_TOL
    else:
        raise ValueError(f"Unknown side: {side}")
    share = np.where(strict, 1.0, np.where(won, 1.0 / (1.0 + count), 0.0))
    return share, won


def evaluate_table_on_fields(
    table: pd.DataFrame,
    fields: list[ScenarioField],
    fill_models: sim.FillModels,
) -> ScoreResult:
    rows: list[dict[str, float | str]] = []
    scenario_pnls: list[float] = []
    weights: list[float] = []
    for field in fields:
        bid, ask = sim.table_quotes(field.our_signal, table)
        ask_share, ask_won = tie_aware_share(ask, field.best_ask, field.best_ask_count, side="ask")
        bid_share, bid_won = tie_aware_share(bid, field.best_bid, field.best_bid_count, side="bid")
        ask_edges = np.maximum(field.best_ask - ask, 0.0)
        bid_edges = np.maximum(bid - field.best_bid, 0.0)
        adverse_ask = field.true_value > ask
        adverse_bid = field.true_value < bid
        buys = ask_share * fill_models.ask.predict(ask_edges, adverse_ask)
        sells = bid_share * fill_models.bid.predict(bid_edges, adverse_bid)
        pnl = buys * (ask - field.true_value) + sells * (field.true_value - bid)
        pnl_mean = float(pnl.mean())
        scenario_pnls.append(pnl_mean)
        weights.append(field.weight)
        rows.append(
            {
                "scenario": field.name,
                "weight": field.weight,
                "pnl": pnl_mean,
                "pnl_se": float(pnl.std(ddof=1) / np.sqrt(len(pnl))),
                "fills": float((buys + sells).mean()),
                "ask_win": float(ask_won.mean()),
                "bid_win": float(bid_won.mean()),
                "adv_ask_win": sim.safe_fraction((ask_won & adverse_ask).sum(), ask_won.sum()),
                "adv_bid_win": sim.safe_fraction((bid_won & adverse_bid).sum(), bid_won.sum()),
            }
        )

    pnl_array = np.asarray(scenario_pnls, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weighted_pnl = float(np.sum(weight_array * pnl_array))
    scenario_std = float(np.sqrt(np.sum(weight_array * (pnl_array - weighted_pnl) ** 2)))
    objective = weighted_pnl - OBJECTIVE_RISK_PENALTY * scenario_std
    return ScoreResult(
        objective=objective,
        weighted_pnl=weighted_pnl,
        scenario_std=scenario_std,
        metrics=pd.DataFrame(rows),
    )


def historical_replay_pnl(
    table: pd.DataFrame,
    df: pd.DataFrame,
    fill_models: sim.FillModels,
) -> float:
    bid, ask = sim.table_quotes(df["mm_signal"].to_numpy(dtype=float), table)
    return sim.evaluate_quotes(df, bid, ask, fill_models)["avg_pnl_per_round"]


def make_smooth_table(params: SmoothParams, require_monotone: bool = True) -> pd.DataFrame:
    base_mid = posterior_mean_uniform_normal(sim.OFFICIAL_SIGNALS)
    mid_shift = np.interp(sim.OFFICIAL_SIGNALS, KNOT_SIGNALS, params.mid_shift)
    h_bid = np.interp(sim.OFFICIAL_SIGNALS, KNOT_SIGNALS, params.h_bid)
    h_ask = np.interp(sim.OFFICIAL_SIGNALS, KNOT_SIGNALS, params.h_ask)
    bid = np.clip(base_mid + mid_shift - h_bid, 0.0, 1200.0)
    ask = np.clip(base_mid + mid_shift + h_ask, 0.0, 1200.0)
    table = pd.DataFrame(
        {
            "signal": sim.OFFICIAL_SIGNALS.astype(int),
            "bid": np.round(bid, 2),
            "ask": np.round(ask, 2),
        }
    )
    validate_table(table, require_monotone=require_monotone)
    return table


def initial_params(name: str) -> SmoothParams:
    base_mid = posterior_mean_uniform_normal(KNOT_SIGNALS)
    if name == "posterior":
        return SmoothParams(
            mid_shift=np.full(len(KNOT_SIGNALS), -5.0),
            h_bid=np.full(len(KNOT_SIGNALS), 90.0),
            h_ask=np.full(len(KNOT_SIGNALS), 90.0),
        )
    if name == "fresh_linear":
        return SmoothParams(
            mid_shift=0.990 * KNOT_SIGNALS - base_mid,
            h_bid=np.full(len(KNOT_SIGNALS), 95.0),
            h_ask=np.full(len(KNOT_SIGNALS), 95.0),
        )
    if name == "v2_formula":
        return SmoothParams(
            mid_shift=0.955 * KNOT_SIGNALS + 22.5 - base_mid,
            h_bid=np.full(len(KNOT_SIGNALS), 100.0),
            h_ask=np.full(len(KNOT_SIGNALS), 100.0),
        )
    raise ValueError(f"Unknown start: {name}")


def bounded_update(params: SmoothParams, array_name: str, index: int, delta: float) -> SmoothParams:
    candidate = params.copy()
    target = getattr(candidate, array_name)
    if array_name == "mid_shift":
        lower, upper = MID_SHIFT_BOUNDS
    else:
        lower, upper = HALF_SPREAD_BOUNDS
    target[index] = float(np.clip(target[index] + delta, lower, upper))
    return candidate


def optimize_smooth_params(
    fields: list[ScenarioField],
    fill_models: sim.FillModels,
    start: SmoothParams,
    steps: Iterable[float] = SEARCH_STEPS,
    max_passes_per_step: int = 4,
) -> tuple[SmoothParams, pd.DataFrame, ScoreResult]:
    params = start
    table = make_smooth_table(params)
    best = evaluate_table_on_fields(table, fields, fill_models)

    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for array_name in ("mid_shift", "h_bid", "h_ask"):
                for index in range(len(KNOT_SIGNALS)):
                    for direction in (1.0, -1.0):
                        candidate_params = bounded_update(params, array_name, index, direction * step)
                        try:
                            candidate_table = make_smooth_table(candidate_params)
                        except ValueError:
                            continue
                        candidate_score = evaluate_table_on_fields(candidate_table, fields, fill_models)
                        if candidate_score.objective > best.objective + 1e-9:
                            params = candidate_params
                            table = candidate_table
                            best = candidate_score
                            improved = True
            if not improved:
                break
    return params, table, best


def bot_reproduction_report(
    df: pd.DataFrame,
    bot_models: list[BotQuoteModel],
    n_rounds: int,
    seed: int,
) -> pd.DataFrame:
    actual_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    actual_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    actual_best_bid = actual_bids.max(axis=1)
    actual_best_ask = actual_asks.min(axis=1)

    rng = np.random.default_rng(seed)
    true_value = rng.uniform(0.0, 1000.0, n_rounds)
    sim_bids, sim_asks = simulate_bot_quotes(true_value, bot_models, rng)
    simulated_best_bid = sim_bids.max(axis=1)
    simulated_best_ask = sim_asks.min(axis=1)

    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(sim.BOT_NAMES):
        actual_ask = float(np.mean(np.abs(actual_asks[:, idx] - actual_best_ask) <= sim.TIE_TOL))
        actual_bid = float(np.mean(np.abs(actual_bids[:, idx] - actual_best_bid) <= sim.TIE_TOL))
        simulated_ask = float(np.mean(np.abs(sim_asks[:, idx] - simulated_best_ask) <= sim.TIE_TOL))
        simulated_bid = float(np.mean(np.abs(sim_bids[:, idx] - simulated_best_bid) <= sim.TIE_TOL))
        rows.append(
            {
                "bot": name,
                "actual_ask_best": actual_ask,
                "sim_ask_best": simulated_ask,
                "ask_diff": simulated_ask - actual_ask,
                "actual_bid_best": actual_bid,
                "sim_bid_best": simulated_bid,
                "bid_diff": simulated_bid - actual_bid,
            }
        )
    return pd.DataFrame(rows)


def candidate_tables() -> dict[str, pd.DataFrame]:
    return {
        "published_coord": PUBLISHED_COORD_TABLE,
        "v2_955_22.5_h100": formula_table(0.955, 22.5, 100.0),
        "fresh_linear_990_0_h95": formula_table(0.990, 0.0, 95.0),
        "posterior_offset_-5_h90": posterior_table(-5.0, 90.0),
    }


def format_metric_table(df: pd.DataFrame) -> str:
    display = df.copy()
    percent_cols = [col for col in display.columns if col.endswith("win") or col.endswith("best") or col.endswith("diff")]
    for column in display.columns:
        if column in {"strategy", "scenario", "bot"}:
            continue
        if column in percent_cols:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    return display.to_string(index=False)


def summarize_candidates(
    tables: dict[str, pd.DataFrame],
    fields: list[ScenarioField],
    df: pd.DataFrame,
    fill_models: sim.FillModels,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, table in tables.items():
        result = evaluate_table_on_fields(table, fields, fill_models)
        no_clone = result.metrics.loc[result.metrics["scenario"] == "no_clones"].iloc[0]
        coord_x3 = result.metrics.loc[result.metrics["scenario"] == "coord_x3"].iloc[0]
        rows.append(
            {
                "strategy": name,
                "objective": result.objective,
                "weighted_pnl": result.weighted_pnl,
                "scenario_std": result.scenario_std,
                "hist_pnl": historical_replay_pnl(table, df, fill_models),
                "fresh_no_clone": no_clone["pnl"],
                "coord_x3": coord_x3["pnl"],
                "fills_no_clone": no_clone["fills"],
            }
        )
    return pd.DataFrame(rows).sort_values("objective", ascending=False)


def evaluate_across_seeds(
    tables: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    bot_models: list[BotQuoteModel],
    fill_models: sim.FillModels,
    n_rounds: int,
    seeds: int,
    first_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    for seed_offset in range(seeds):
        fields = build_scenario_fields(
            bot_models,
            n_rounds=n_rounds,
            seed=first_seed + seed_offset,
            scenario_tables=scenario_tables(),
        )
        for name, table in tables.items():
            result = evaluate_table_on_fields(table, fields, fill_models)
            rows.append(
                {
                    "seed": seed_offset,
                    "strategy": name,
                    "objective": result.objective,
                    "weighted_pnl": result.weighted_pnl,
                    "scenario_std": result.scenario_std,
                }
            )

    raw = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | str]] = []
    for name, group in raw.groupby("strategy", sort=False):
        objective = group["objective"].to_numpy(dtype=float)
        weighted = group["weighted_pnl"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "strategy": name,
                "objective_mean": float(objective.mean()),
                "objective_se": float(objective.std(ddof=1) / np.sqrt(len(objective))),
                "weighted_pnl_mean": float(weighted.mean()),
                "weighted_pnl_se": float(weighted.std(ddof=1) / np.sqrt(len(weighted))),
            }
        )
    return raw, pd.DataFrame(summary_rows).sort_values("objective_mean", ascending=False)


def paired_difference_report(raw: pd.DataFrame, challenger: str, baseline: str) -> dict[str, float]:
    wide = raw.pivot(index="seed", columns="strategy", values="objective")
    diff = wide[challenger] - wide[baseline]
    return {
        "mean_diff": float(diff.mean()),
        "se_diff": float(diff.std(ddof=1) / np.sqrt(len(diff))),
    }


def print_table(title: str, table: pd.DataFrame) -> None:
    print()
    print(title)
    print("=" * len(title))
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust final-score strategy search.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--search-n", type=int, default=60_000)
    parser.add_argument("--eval-n", type=int, default=100_000)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--search-seed", type=int, default=20260508)
    parser.add_argument("--eval-seed", type=int, default=20260601)
    parser.add_argument("--skip-search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = sim.load_history(args.csv)
    fill_models = sim.build_fill_models(df)
    bot_models = fit_bot_quote_models(df)

    reproduction = bot_reproduction_report(df, bot_models, n_rounds=100_000, seed=args.search_seed)
    max_abs_diff = float(
        reproduction[["ask_diff", "bid_diff"]].abs().to_numpy(dtype=float).max()
    )
    print("Bot simulator reproduction")
    print("==========================")
    print(format_metric_table(reproduction))
    if max_abs_diff > 0.02:
        print(f"WARNING: best-price share drift exceeds 2 percentage points: {max_abs_diff:.2%}")
    else:
        print(f"PASS: max best-price share drift is {max_abs_diff:.2%}")

    search_fields = build_scenario_fields(
        bot_models,
        n_rounds=args.search_n,
        seed=args.search_seed + 1,
        scenario_tables=scenario_tables(),
    )

    tables = candidate_tables()
    baseline_summary = summarize_candidates(tables, search_fields, df, fill_models)
    print()
    print("Baseline robust scores on search field")
    print("======================================")
    print(format_metric_table(baseline_summary))

    if args.skip_search:
        robust_table = tables["fresh_linear_990_0_h95"]
        robust_name = "fresh_linear_990_0_h95"
    else:
        best_name = ""
        best_params: SmoothParams | None = None
        robust_table = tables["fresh_linear_990_0_h95"]
        best_score = evaluate_table_on_fields(robust_table, search_fields, fill_models)
        starts = ("posterior", "fresh_linear", "v2_formula")
        for start_name in starts:
            print()
            print(f"Searching from {start_name} start...")
            params, table, score = optimize_smooth_params(
                search_fields,
                fill_models,
                initial_params(start_name),
            )
            print(
                f"  objective={score.objective:.4f}, "
                f"weighted_pnl={score.weighted_pnl:.4f}, scenario_std={score.scenario_std:.4f}"
            )
            if score.objective > best_score.objective:
                best_name = f"optimized_from_{start_name}"
                best_params = params
                robust_table = table
                best_score = score

        if best_params is not None:
            params_display = pd.DataFrame(
                {
                    "signal": KNOT_SIGNALS.astype(int),
                    "mid_shift": np.round(best_params.mid_shift, 2),
                    "h_bid": np.round(best_params.h_bid, 2),
                    "h_ask": np.round(best_params.h_ask, 2),
                }
            )
            print_table(f"Best smooth parameters ({best_name})", params_display)
            robust_name = best_name
        else:
            robust_name = "fresh_linear_990_0_h95"

        tables[f"{robust_name}"] = robust_table
        post_search = summarize_candidates(tables, search_fields, df, fill_models)
        print()
        print("Post-search robust scores on search field")
        print("=========================================")
        print(format_metric_table(post_search))

    final_tables = {
        "published_coord": PUBLISHED_COORD_TABLE,
        "fresh_linear_990_0_h95": tables["fresh_linear_990_0_h95"],
        robust_name: robust_table,
    }
    final_tables = dict(final_tables.items())
    raw, seed_summary = evaluate_across_seeds(
        final_tables,
        df=df,
        bot_models=bot_models,
        fill_models=fill_models,
        n_rounds=args.eval_n,
        seeds=args.eval_seeds,
        first_seed=args.eval_seed,
    )
    print()
    print(f"Final paired evaluation across {args.eval_seeds} seeds x {args.eval_n:,} rounds")
    print("================================================================")
    print(format_metric_table(seed_summary))

    if robust_name != "fresh_linear_990_0_h95":
        diff = paired_difference_report(raw, robust_name, "fresh_linear_990_0_h95")
        print()
        print("Optimized-vs-fresh-linear paired objective difference")
        print("====================================================")
        print(f"mean_diff={diff['mean_diff']:.4f}, se_diff={diff['se_diff']:.4f}")
        if diff["mean_diff"] > diff["se_diff"]:
            recommended_name = robust_name
        else:
            recommended_name = "fresh_linear_990_0_h95"
    else:
        recommended_name = "fresh_linear_990_0_h95"

    recommended = final_tables[recommended_name]
    validate_table(recommended)
    print_table(f"Recommended final table: {recommended_name}", recommended)
    print()
    print("Copy-paste rows")
    print("===============")
    print(recommended.to_csv(index=False, sep="\t", lineterminator="\n").strip())


if __name__ == "__main__":
    main()
