"""Run second-pass strategy experiments for Round 2.

This script keeps the original historical_simulator.py untouched and layers on:
- even/odd holdout validation,
- raw 21-point coordinate ascent with different step schedules,
- a smoother piecewise-linear spread parameterization,
- simple crowding stress tests with synthetic participant clones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import historical_simulator as sim


ALPHA = 0.955
INTERCEPT = 22.5
BASE_HALF_SPREAD = 100.0
MULTI_STARTS = 5
MULTI_START_SEED = 20260509
RANDOM_ALPHA_RANGE = (0.93, 1.02)
RANDOM_INTERCEPT_RANGE = (-40.0, 60.0)
RANDOM_HALF_SPREAD_RANGE = (70.0, 150.0)
RANDOM_JITTER = 8.0

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
class StrategyResult:
    name: str
    metrics: dict[str, float]


def table_score(df: pd.DataFrame, models: sim.FillModels, table: pd.DataFrame) -> float:
    bid, ask = sim.table_quotes(df["mm_signal"].to_numpy(dtype=float), table)
    return sim.evaluate_quotes(df, bid, ask, models)["avg_pnl_per_round"]


def evaluate_table(
    df: pd.DataFrame,
    models: sim.FillModels,
    name: str,
    table: pd.DataFrame,
) -> StrategyResult:
    bid, ask = sim.table_quotes(df["mm_signal"].to_numpy(dtype=float), table)
    return StrategyResult(name, sim.evaluate_quotes(df, bid, ask, models))


def evaluate_formula(
    df: pd.DataFrame,
    models: sim.FillModels,
    name: str,
    alpha: float,
    intercept: float,
    half_spread: float,
) -> StrategyResult:
    bid, ask = sim.formula_quotes(
        df["mm_signal"].to_numpy(dtype=float),
        alpha,
        intercept,
        half_spread,
    )
    return StrategyResult(name, sim.evaluate_quotes(df, bid, ask, models))


def formula_table(alpha: float, intercept: float, half_spread: float) -> pd.DataFrame:
    return sim.generate_formula_table(alpha, intercept, half_spread)


def valid_table(table: pd.DataFrame) -> bool:
    bid = table["bid"].to_numpy(dtype=float)
    ask = table["ask"].to_numpy(dtype=float)
    return bool(
        np.all(bid >= -sim.TIE_TOL)
        and np.all(ask <= 1200.0 + sim.TIE_TOL)
        and np.all(bid <= ask + sim.TIE_TOL)
    )


def jitter_table(
    table: pd.DataFrame,
    rng: np.random.Generator,
    bid_jitter: float = RANDOM_JITTER,
    ask_jitter: float = RANDOM_JITTER,
) -> pd.DataFrame:
    jittered = table.copy(deep=True)
    bids = jittered["bid"].to_numpy(dtype=float) + rng.uniform(-bid_jitter, bid_jitter, len(jittered))
    asks = jittered["ask"].to_numpy(dtype=float) + rng.uniform(-ask_jitter, ask_jitter, len(jittered))
    bids = np.clip(bids, 0.0, 1200.0)
    asks = np.clip(asks, 0.0, 1200.0)
    asks = np.maximum(asks, bids)
    jittered["bid"] = bids
    jittered["ask"] = asks
    if not valid_table(jittered):
        raise ValueError("Jittered table violates bid/ask constraints")
    return jittered


def random_start_table(
    rng: np.random.Generator,
    alpha_range: tuple[float, float] = RANDOM_ALPHA_RANGE,
    intercept_range: tuple[float, float] = RANDOM_INTERCEPT_RANGE,
    half_spread_range: tuple[float, float] = RANDOM_HALF_SPREAD_RANGE,
    jitter: float = RANDOM_JITTER,
) -> tuple[pd.DataFrame, dict[str, float]]:
    alpha = float(rng.uniform(*alpha_range))
    intercept = float(rng.uniform(*intercept_range))
    half_spread = float(rng.uniform(*half_spread_range))
    table = formula_table(alpha, intercept, half_spread)
    if jitter > 0:
        table = jitter_table(table, rng, bid_jitter=jitter, ask_jitter=jitter)
    return table, {"alpha": alpha, "intercept": intercept, "half_spread": half_spread}


def multi_start_coordinate_ascent(
    df: pd.DataFrame,
    full_models: sim.FillModels,
    steps: list[float],
    starts: int = MULTI_STARTS,
    seed: int = MULTI_START_SEED,
    max_passes_per_step: int = 3,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    even = df[df["period"] % 2 == 0].reset_index(drop=True)
    odd = df[df["period"] % 2 == 1].reset_index(drop=True)
    folds = (("even", even, "odd", odd), ("odd", odd, "even", even))

    results = []
    best_start = None
    best_cv = -np.inf
    for start_idx in range(starts):
        start_table, params = random_start_table(rng)
        fold_scores = []
        for train_name, train_df, test_name, test_df in folds:
            train_models = sim.build_fill_models(train_df)
            optimized, _ = coordinate_ascent_table(
                train_df,
                train_models,
                start_table,
                steps=steps,
                max_passes_per_step=max_passes_per_step,
            )
            test_score = table_score(test_df, train_models, optimized)
            fold_scores.append(test_score)
        cv_mean = float(np.mean(fold_scores))
        results.append(
            {
                "start_id": start_idx + 1,
                "alpha": params["alpha"],
                "intercept": params["intercept"],
                "half_spread": params["half_spread"],
                "cv_mean": cv_mean,
                "cv_even_to_odd": float(fold_scores[0]),
                "cv_odd_to_even": float(fold_scores[1]),
            }
        )
        if cv_mean > best_cv:
            best_cv = cv_mean
            best_start = start_table

    if best_start is None:
        raise RuntimeError("Multi-start coordinate ascent produced no valid starts")

    best_table, best_score = coordinate_ascent_table(
        df,
        full_models,
        best_start,
        steps=steps,
        max_passes_per_step=max_passes_per_step,
    )
    return best_table, best_score, pd.DataFrame(results)


def coordinate_ascent_table(
    df: pd.DataFrame,
    models: sim.FillModels,
    start_table: pd.DataFrame,
    steps: list[float],
    max_passes_per_step: int = 4,
) -> tuple[pd.DataFrame, float]:
    table = start_table.copy(deep=True).reset_index(drop=True)
    best = table_score(df, models, table)

    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for row in range(len(table)):
                for column in ("bid", "ask"):
                    for direction in (1.0, -1.0):
                        candidate = table.copy(deep=True)
                        candidate.loc[row, column] = candidate.loc[row, column] + direction * step
                        candidate.loc[row, "bid"] = float(np.clip(candidate.loc[row, "bid"], 0.0, 1200.0))
                        candidate.loc[row, "ask"] = float(np.clip(candidate.loc[row, "ask"], 0.0, 1200.0))
                        if not valid_table(candidate):
                            continue
                        score = table_score(df, models, candidate)
                        if score > best + 1e-9:
                            table = candidate
                            best = score
                            improved = True
            if not improved:
                break

    table[["bid", "ask"]] = table[["bid", "ask"]].round(2)
    return table, table_score(df, models, table)


def optimize_piecewise_spreads(
    df: pd.DataFrame,
    models: sim.FillModels,
    steps: list[float],
    knot_signals: np.ndarray | None = None,
    max_passes_per_step: int = 5,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    if knot_signals is None:
        knot_signals = np.array([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])

    bid_h = np.full(len(knot_signals), BASE_HALF_SPREAD, dtype=float)
    ask_h = np.full(len(knot_signals), BASE_HALF_SPREAD, dtype=float)

    def make_table(bid_spreads: np.ndarray, ask_spreads: np.ndarray) -> pd.DataFrame:
        mid = ALPHA * sim.OFFICIAL_SIGNALS + INTERCEPT
        official_bid_h = np.interp(sim.OFFICIAL_SIGNALS, knot_signals, bid_spreads)
        official_ask_h = np.interp(sim.OFFICIAL_SIGNALS, knot_signals, ask_spreads)
        bid = np.clip(mid - official_bid_h, 0.0, 1200.0)
        ask = np.clip(mid + official_ask_h, 0.0, 1200.0)
        table = pd.DataFrame(
            {
                "signal": sim.OFFICIAL_SIGNALS.astype(int),
                "bid": bid,
                "ask": ask,
            }
        )
        if not valid_table(table):
            raise ValueError("Invalid piecewise spread table")
        return table

    table = make_table(bid_h, ask_h)
    best = table_score(df, models, table)

    for step in steps:
        for _ in range(max_passes_per_step):
            improved = False
            for side in ("bid", "ask"):
                spreads = bid_h if side == "bid" else ask_h
                for idx in range(len(spreads)):
                    for direction in (1.0, -1.0):
                        next_bid_h = bid_h.copy()
                        next_ask_h = ask_h.copy()
                        target = next_bid_h if side == "bid" else next_ask_h
                        target[idx] = float(np.clip(target[idx] + direction * step, 0.0, 500.0))
                        candidate = make_table(next_bid_h, next_ask_h)
                        score = table_score(df, models, candidate)
                        if score > best + 1e-9:
                            bid_h = next_bid_h
                            ask_h = next_ask_h
                            table = candidate
                            best = score
                            improved = True
            if not improved:
                break

    table[["bid", "ask"]] = table[["bid", "ask"]].round(2)
    knots = pd.DataFrame({"signal": knot_signals.astype(int), "bid_h": bid_h, "ask_h": ask_h})
    return table, table_score(df, models, table), knots.round(2)


def conservative_table(base: pd.DataFrame, widen: float) -> pd.DataFrame:
    table = base.copy(deep=True)
    table["bid"] = np.clip(table["bid"].to_numpy(dtype=float) - widen, 0.0, 1200.0)
    table["ask"] = np.clip(table["ask"].to_numpy(dtype=float) + widen, 0.0, 1200.0)
    table[["bid", "ask"]] = table[["bid", "ask"]].round(2)
    return table


def evaluate_with_extra_competitors(
    df: pd.DataFrame,
    models: sim.FillModels,
    table: pd.DataFrame,
    extra_tables: list[pd.DataFrame],
    rng: np.random.Generator,
) -> dict[str, float]:
    our_bid, our_ask = sim.table_quotes(df["mm_signal"].to_numpy(dtype=float), table)
    true_value = df["true_value"].to_numpy(dtype=float)
    extra_bids = []
    extra_asks = []
    for extra_table in extra_tables:
        signals = np.clip(true_value + rng.normal(0.0, 50.0, size=len(df)), 0.0, 1000.0)
        bid, ask = sim.table_quotes(signals, extra_table)
        extra_bids.append(bid)
        extra_asks.append(ask)

    bot_bids = df.loc[:, sim.BOT_BID_COLS].to_numpy(dtype=float)
    bot_asks = df.loc[:, sim.BOT_ASK_COLS].to_numpy(dtype=float)
    all_bids = np.column_stack([bot_bids, *extra_bids])
    all_asks = np.column_stack([bot_asks, *extra_asks])

    ask_share, best_ask = sim.best_side_share(our_ask, all_asks, side="ask")
    bid_share, best_bid = sim.best_side_share(our_bid, all_bids, side="bid")
    ask_edges = np.maximum(best_ask - our_ask, 0.0)
    bid_edges = np.maximum(our_bid - best_bid, 0.0)
    adverse_ask = true_value > our_ask
    adverse_bid = true_value < our_bid
    buys = ask_share * models.ask.predict(ask_edges, adverse_ask)
    sells = bid_share * models.bid.predict(bid_edges, adverse_bid)
    pnl = buys * (our_ask - true_value) + sells * (true_value - our_bid)
    ask_won = ask_share > 0.0
    bid_won = bid_share > 0.0

    return {
        "avg_pnl_per_round": float(pnl.sum() / len(df)),
        "ask_win_frequency": float(ask_won.mean()),
        "bid_win_frequency": float(bid_won.mean()),
        "adverse_ask_given_win": sim.safe_fraction((ask_won & adverse_ask).sum(), ask_won.sum()),
        "adverse_bid_given_win": sim.safe_fraction((bid_won & adverse_bid).sum(), bid_won.sum()),
        "fills_per_round": float((buys.sum() + sells.sum()) / len(df)),
    }


def format_results(results: list[StrategyResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        m = result.metrics
        rows.append(
            {
                "strategy": result.name,
                "pnl": m["avg_pnl_per_round"],
                "ask_win": m["ask_win_frequency"],
                "bid_win": m["bid_win_frequency"],
                "adv_ask_win": m["adverse_ask_given_win"],
                "adv_bid_win": m["adverse_bid_given_win"],
                "fills": m["expected_buys_per_round"] + m["expected_sells_per_round"],
            }
        )
    return pd.DataFrame(rows)


def print_metric_table(title: str, df: pd.DataFrame) -> None:
    display = df.copy()
    for col in ("pnl", "fills"):
        display[col] = display[col].map(lambda value: f"{value:.2f}")
    for col in ("ask_win", "bid_win", "adv_ask_win", "adv_bid_win"):
        display[col] = display[col].map(lambda value: f"{value:.1%}")
    print()
    print(title)
    print("=" * len(title))
    print(display.to_string(index=False))


def print_submission_table(title: str, table: pd.DataFrame) -> None:
    print()
    print(title)
    print("=" * len(title))
    print(table.to_string(index=False))


def main() -> None:
    df = sim.load_history(Path("auction_history.csv"))
    full_models = sim.build_fill_models(df)

    base_formula = formula_table(ALPHA, INTERCEPT, BASE_HALF_SPREAD)
    conservative_formula = formula_table(ALPHA, INTERCEPT, 105.0)

    full_results = [
        evaluate_formula(df, full_models, "formula h=100", ALPHA, INTERCEPT, 100.0),
        evaluate_formula(df, full_models, "formula h=105", ALPHA, INTERCEPT, 105.0),
        evaluate_table(df, full_models, "published coord", PUBLISHED_COORD_TABLE),
        evaluate_table(df, full_models, "published coord +5 wide", conservative_table(PUBLISHED_COORD_TABLE, 5.0)),
        evaluate_table(df, full_models, "published coord +10 wide", conservative_table(PUBLISHED_COORD_TABLE, 10.0)),
    ]
    print_metric_table("Full-history baseline comparison", format_results(full_results))

    print()
    print("Running raw coordinate ascent variants...")
    full_raw_big, full_raw_big_score = coordinate_ascent_table(
        df,
        full_models,
        base_formula,
        steps=[50, 25, 15, 10, 5, 3, 2, 1],
    )
    full_raw_small, full_raw_small_score = coordinate_ascent_table(
        df,
        full_models,
        base_formula,
        steps=[10, 5, 3, 2, 1],
    )
    full_refined, full_refined_score = coordinate_ascent_table(
        df,
        full_models,
        PUBLISHED_COORD_TABLE,
        steps=[10, 5, 3, 2, 1],
    )
    print(f"raw big-step score:       {full_raw_big_score:.4f}")
    print(f"raw small-step score:     {full_raw_small_score:.4f}")
    print(f"published refined score:  {full_refined_score:.4f}")

    print()
    print("Running smoother piecewise spread optimizer...")
    smooth_table, smooth_score, smooth_knots = optimize_piecewise_spreads(
        df,
        full_models,
        steps=[50, 25, 10, 5, 2, 1],
    )
    print(f"smooth piecewise score:   {smooth_score:.4f}")
    print("smooth spread knots:")
    print(smooth_knots.to_string(index=False))

    even = df[df["period"] % 2 == 0].reset_index(drop=True)
    odd = df[df["period"] % 2 == 1].reset_index(drop=True)
    fold_rows = []
    for train_name, train_df, test_name, test_df in (
        ("even", even, "odd", odd),
        ("odd", odd, "even", even),
    ):
        train_models = sim.build_fill_models(train_df)
        train_raw, _ = coordinate_ascent_table(
            train_df,
            train_models,
            base_formula,
            steps=[10, 5, 3, 2, 1],
            max_passes_per_step=3,
        )
        train_smooth, _, _ = optimize_piecewise_spreads(
            train_df,
            train_models,
            steps=[25, 10, 5, 2, 1],
            max_passes_per_step=3,
        )
        candidates = [
            ("formula h=100", base_formula),
            ("formula h=105", conservative_formula),
            ("published coord", PUBLISHED_COORD_TABLE),
            ("raw opt on " + train_name, train_raw),
            ("smooth opt on " + train_name, train_smooth),
        ]
        for strategy, table in candidates:
            train_score = table_score(train_df, train_models, table)
            test_score = table_score(test_df, train_models, table)
            fold_rows.append(
                {
                    "train": train_name,
                    "test": test_name,
                    "strategy": strategy,
                    "train_pnl": train_score,
                    "test_pnl": test_score,
                    "gap": train_score - test_score,
                }
            )

    holdout = pd.DataFrame(fold_rows)
    holdout_display = holdout.copy()
    for col in ("train_pnl", "test_pnl", "gap"):
        holdout_display[col] = holdout_display[col].map(lambda value: f"{value:.2f}")
    print()
    print("Even/Odd holdout validation")
    print("===========================")
    print(holdout_display.to_string(index=False))

    if MULTI_STARTS > 0:
        print()
        print(f"Running multi-start coordinate ascent ({MULTI_STARTS} starts)...")
        multi_table, multi_score, multi_summary = multi_start_coordinate_ascent(
            df,
            full_models,
            steps=[10, 5, 3, 2, 1],
            starts=MULTI_STARTS,
            seed=MULTI_START_SEED,
            max_passes_per_step=3,
        )
        multi_display = multi_summary.copy()
        for col in ("cv_mean", "cv_even_to_odd", "cv_odd_to_even"):
            multi_display[col] = multi_display[col].map(lambda value: f"{value:.2f}")
        print(multi_display.to_string(index=False))
        print(f"best multi-start full-history score: {multi_score:.4f}")
        print_submission_table("Best multi-start raw table", multi_table)

    rng_seed = 20260508
    stress_rows = []
    stress_tables = [
        ("published coord", PUBLISHED_COORD_TABLE),
        ("coord +5 wide", conservative_table(PUBLISHED_COORD_TABLE, 5.0)),
        ("coord +10 wide", conservative_table(PUBLISHED_COORD_TABLE, 10.0)),
        ("formula h=105", conservative_formula),
        ("smooth piecewise", smooth_table),
    ]
    for clones in (1, 3, 5, 10):
        for name, table in stress_tables:
            metrics = evaluate_with_extra_competitors(
                df,
                full_models,
                table,
                [PUBLISHED_COORD_TABLE] * clones,
                np.random.default_rng(rng_seed + clones),
            )
            stress_rows.append({"clones": clones, "strategy": name, **metrics})

    stress = pd.DataFrame(stress_rows)
    stress_display = stress.copy()
    stress_display["avg_pnl_per_round"] = stress_display["avg_pnl_per_round"].map(lambda value: f"{value:.2f}")
    stress_display["fills_per_round"] = stress_display["fills_per_round"].map(lambda value: f"{value:.2f}")
    for col in ("ask_win_frequency", "bid_win_frequency", "adverse_ask_given_win", "adverse_bid_given_win"):
        stress_display[col] = stress_display[col].map(lambda value: f"{value:.1%}")
    print()
    print("Crowding stress test against synthetic coord-table clones")
    print("=========================================================")
    print(stress_display.to_string(index=False))

    best_name = "published refined" if full_refined_score >= full_raw_big_score else "raw big-step"
    best_table = full_refined if best_name == "published refined" else full_raw_big
    print_submission_table(f"Best full-history raw table ({best_name})", best_table)
    print_submission_table("Smooth piecewise table", smooth_table)


if __name__ == "__main__":
    main()
