"""Replay historical auction rounds against the five house operators.

This is a deterministic expected-PnL evaluator. It learns expected fill
counts from the reference booth in auction_history.csv, then scores candidate
quotes by replaying the historical bot quotes and true values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BOT_NAMES = ("tight", "wide", "skewed", "noisy", "fade")
BOT_BID_COLS = tuple(f"bot_{name}_bid" for name in BOT_NAMES)
BOT_ASK_COLS = tuple(f"bot_{name}_ask" for name in BOT_NAMES)
REQUIRED_COLUMNS = {
    "period",
    "true_value",
    "mm_signal",
    "mm_bid",
    "mm_ask",
    "mm_num_buys",
    "mm_num_sells",
    "mm_pnl",
    *BOT_BID_COLS,
    *BOT_ASK_COLS,
}

TIE_TOL = 1e-9
EDGE_BIN_UPPER_BOUNDS = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf])
SHRINKAGE_STRENGTH = 20.0
OFFICIAL_SIGNALS = np.arange(0.0, 1000.0 + 50.0, 50.0)


@dataclass(frozen=True)
class SideFillModel:
    side: str
    global_mean: float
    estimates: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(
        cls,
        side: str,
        edges: np.ndarray,
        adverse: np.ndarray,
        fills: np.ndarray,
        shrinkage: float = SHRINKAGE_STRENGTH,
    ) -> "SideFillModel":
        if len(edges) == 0:
            raise ValueError(f"Cannot fit {side} fill model with no observations")

        clean_edges = np.maximum(np.asarray(edges, dtype=float), 0.0)
        clean_adverse = np.asarray(adverse, dtype=bool)
        clean_fills = np.asarray(fills, dtype=float)

        global_mean = float(clean_fills.mean())
        bin_ids = edge_bin_ids(clean_edges)
        estimates = np.full((len(EDGE_BIN_UPPER_BOUNDS), 2), global_mean, dtype=float)
        counts = np.zeros((len(EDGE_BIN_UPPER_BOUNDS), 2), dtype=int)

        for bin_id in range(len(EDGE_BIN_UPPER_BOUNDS)):
            for adverse_id in (0, 1):
                mask = (bin_ids == bin_id) & (clean_adverse == bool(adverse_id))
                count = int(mask.sum())
                counts[bin_id, adverse_id] = count
                if count:
                    total = float(clean_fills[mask].sum())
                    estimates[bin_id, adverse_id] = (
                        total + shrinkage * global_mean
                    ) / (count + shrinkage)

        return cls(
            side=side,
            global_mean=global_mean,
            estimates=estimates,
            counts=counts,
        )

    def predict(self, edges: np.ndarray, adverse: np.ndarray) -> np.ndarray:
        bin_ids = edge_bin_ids(np.maximum(np.asarray(edges, dtype=float), 0.0))
        adverse_ids = np.asarray(adverse, dtype=bool).astype(int)
        return self.estimates[bin_ids, adverse_ids]


@dataclass(frozen=True)
class FillModels:
    ask: SideFillModel
    bid: SideFillModel


def edge_bin_ids(edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGE_BIN_UPPER_BOUNDS, edges, side="left")


def load_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_fill_models(df: pd.DataFrame) -> FillModels:
    bot_asks = df.loc[:, BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, BOT_BID_COLS].to_numpy(dtype=float)
    true_value = df["true_value"].to_numpy(dtype=float)
    mm_ask = df["mm_ask"].to_numpy(dtype=float)
    mm_bid = df["mm_bid"].to_numpy(dtype=float)

    ask_edges = bot_asks.min(axis=1) - mm_ask
    bid_edges = mm_bid - bot_bids.max(axis=1)
    # Historical reference fill averages in the roadmap use strict best-price
    # rounds. Candidate simulation still handles exact ties by splitting flow.
    ask_won = ask_edges > TIE_TOL
    bid_won = bid_edges > TIE_TOL

    ask_model = SideFillModel.fit(
        side="ask",
        edges=ask_edges[ask_won],
        adverse=true_value[ask_won] > mm_ask[ask_won],
        fills=df.loc[ask_won, "mm_num_buys"].to_numpy(dtype=float),
    )
    bid_model = SideFillModel.fit(
        side="bid",
        edges=bid_edges[bid_won],
        adverse=true_value[bid_won] < mm_bid[bid_won],
        fills=df.loc[bid_won, "mm_num_sells"].to_numpy(dtype=float),
    )
    return FillModels(ask=ask_model, bid=bid_model)


def formula_quotes(
    signals: np.ndarray,
    alpha: float,
    intercept: float,
    half_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    mid = alpha * np.asarray(signals, dtype=float) + intercept
    bid = np.clip(mid - half_spread, 0.0, 1200.0)
    ask = np.clip(mid + half_spread, 0.0, 1200.0)
    validate_quotes(bid, ask)
    return bid, ask


def table_quotes(
    signals: np.ndarray,
    table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    table_signals = table["signal"].to_numpy(dtype=float)
    table_bids = table["bid"].to_numpy(dtype=float)
    table_asks = table["ask"].to_numpy(dtype=float)
    bid = np.interp(signals, table_signals, table_bids)
    ask = np.interp(signals, table_signals, table_asks)
    validate_quotes(bid, ask)
    return bid, ask


def validate_quotes(bid: np.ndarray, ask: np.ndarray) -> None:
    if bid.shape != ask.shape:
        raise ValueError("Bid and ask arrays must have the same shape")
    if np.any(bid < -TIE_TOL):
        raise ValueError("Bid quote below 0")
    if np.any(ask > 1200.0 + TIE_TOL):
        raise ValueError("Ask quote above 1200")
    if np.any(bid > ask + TIE_TOL):
        raise ValueError("Bid quote above ask")


def load_submission_table(path: Path) -> pd.DataFrame:
    first_pass = pd.read_csv(path)
    column_lookup = {str(col).strip().lower(): col for col in first_pass.columns}

    if {"signal", "bid", "ask"}.issubset(column_lookup):
        table = first_pass[
            [
                column_lookup["signal"],
                column_lookup["bid"],
                column_lookup["ask"],
            ]
        ].copy()
        table.columns = ["signal", "bid", "ask"]
    else:
        table = pd.read_csv(path, header=None)
        if table.shape[1] < 3:
            raise ValueError("Submission table must have signal, bid, and ask columns")
        table = table.iloc[:, :3].copy()
        table.columns = ["signal", "bid", "ask"]

    for column in ("signal", "bid", "ask"):
        table[column] = pd.to_numeric(table[column], errors="raise")

    table = table.sort_values("signal").reset_index(drop=True)
    if len(table) != len(OFFICIAL_SIGNALS):
        raise ValueError("Submission table must contain exactly 21 rows")
    if not np.allclose(table["signal"].to_numpy(dtype=float), OFFICIAL_SIGNALS):
        expected = ", ".join(str(int(value)) for value in OFFICIAL_SIGNALS)
        raise ValueError(f"Submission signals must be exactly: {expected}")

    validate_quotes(
        table["bid"].to_numpy(dtype=float),
        table["ask"].to_numpy(dtype=float),
    )
    return table


def best_side_share(
    candidate: np.ndarray,
    competitors: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if side == "ask":
        best_competitor = competitors.min(axis=1)
        is_best = candidate <= best_competitor + TIE_TOL
    elif side == "bid":
        best_competitor = competitors.max(axis=1)
        is_best = candidate >= best_competitor - TIE_TOL
    else:
        raise ValueError(f"Unknown side: {side}")

    tied_competitors = np.sum(np.abs(competitors - candidate[:, None]) <= TIE_TOL, axis=1)
    share = np.where(is_best, 1.0 / (1.0 + tied_competitors), 0.0)
    return share, best_competitor


def evaluate_quotes(
    df: pd.DataFrame,
    bid: np.ndarray,
    ask: np.ndarray,
    fill_models: FillModels,
) -> dict[str, float]:
    validate_quotes(bid, ask)
    n_rounds = len(df)
    true_value = df["true_value"].to_numpy(dtype=float)
    bot_asks = df.loc[:, BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, BOT_BID_COLS].to_numpy(dtype=float)

    ask_share, best_bot_ask = best_side_share(ask, bot_asks, side="ask")
    bid_share, best_bot_bid = best_side_share(bid, bot_bids, side="bid")

    ask_edges = np.maximum(best_bot_ask - ask, 0.0)
    bid_edges = np.maximum(bid - best_bot_bid, 0.0)
    adverse_ask = true_value > ask
    adverse_bid = true_value < bid

    expected_buys = ask_share * fill_models.ask.predict(ask_edges, adverse_ask)
    expected_sells = bid_share * fill_models.bid.predict(bid_edges, adverse_bid)
    ask_pnl = expected_buys * (ask - true_value)
    bid_pnl = expected_sells * (true_value - bid)
    total_pnl = ask_pnl + bid_pnl

    ask_won = ask_share > 0.0
    bid_won = bid_share > 0.0
    ask_adverse_won = ask_won & adverse_ask
    bid_adverse_won = bid_won & adverse_bid
    avg_pnl_per_round = float(total_pnl.sum() / n_rounds)

    return {
        "rounds": float(n_rounds),
        "avg_pnl_per_round": avg_pnl_per_round,
        "pnl_mean": avg_pnl_per_round,
        "pnl": avg_pnl_per_round,
        "ask_pnl_per_round": float(ask_pnl.sum() / n_rounds),
        "bid_pnl_per_round": float(bid_pnl.sum() / n_rounds),
        "ask_win_frequency": float(ask_won.mean()),
        "bid_win_frequency": float(bid_won.mean()),
        "adverse_ask_hit_frequency": float(ask_adverse_won.mean()),
        "adverse_bid_hit_frequency": float(bid_adverse_won.mean()),
        "adverse_ask_given_win": safe_fraction(ask_adverse_won.sum(), ask_won.sum()),
        "adverse_bid_given_win": safe_fraction(bid_adverse_won.sum(), bid_won.sum()),
        "expected_buys_total": float(expected_buys.sum()),
        "expected_sells_total": float(expected_sells.sum()),
        "expected_buys_per_round": float(expected_buys.sum() / n_rounds),
        "expected_sells_per_round": float(expected_sells.sum() / n_rounds),
        "mean_bid": float(bid.mean()),
        "mean_ask": float(ask.mean()),
    }


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def run_grid(
    df: pd.DataFrame,
    fill_models: FillModels,
    alpha: float,
    intercept: float,
    grid: tuple[float, float, float],
) -> pd.DataFrame:
    start, stop, step = grid
    if step <= 0:
        raise ValueError("Grid step must be positive")
    if stop < start:
        raise ValueError("Grid stop must be greater than or equal to start")

    spreads = np.arange(start, stop + step * 0.5, step)
    rows: list[dict[str, float]] = []
    signals = df["mm_signal"].to_numpy(dtype=float)
    for half_spread in spreads:
        bid, ask = formula_quotes(signals, alpha, intercept, half_spread)
        metrics = evaluate_quotes(df, bid, ask, fill_models)
        metrics["half_spread"] = float(half_spread)
        rows.append(metrics)

    result = pd.DataFrame(rows)
    return result.sort_values("avg_pnl_per_round", ascending=False).reset_index(drop=True)


def generate_formula_table(alpha: float, intercept: float, half_spread: float) -> pd.DataFrame:
    bid, ask = formula_quotes(OFFICIAL_SIGNALS, alpha, intercept, half_spread)
    return pd.DataFrame(
        {
            "signal": OFFICIAL_SIGNALS.astype(int),
            "bid": np.round(bid, 4),
            "ask": np.round(ask, 4),
        }
    )


def print_single_report(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print("=" * len(title))
    print(f"Average PnL / round:      {metrics['avg_pnl_per_round']:.4f}")
    print(f"Ask-side PnL / round:     {metrics['ask_pnl_per_round']:.4f}")
    print(f"Bid-side PnL / round:     {metrics['bid_pnl_per_round']:.4f}")
    print(f"Best ask frequency:       {metrics['ask_win_frequency']:.2%}")
    print(f"Best bid frequency:       {metrics['bid_win_frequency']:.2%}")
    print(f"Adverse ask frequency:    {metrics['adverse_ask_hit_frequency']:.2%}")
    print(f"Adverse bid frequency:    {metrics['adverse_bid_hit_frequency']:.2%}")
    print(f"Adverse ask | ask won:    {metrics['adverse_ask_given_win']:.2%}")
    print(f"Adverse bid | bid won:    {metrics['adverse_bid_given_win']:.2%}")
    print(f"Expected buys / round:    {metrics['expected_buys_per_round']:.4f}")
    print(f"Expected sells / round:   {metrics['expected_sells_per_round']:.4f}")
    print(f"Expected buys total:      {metrics['expected_buys_total']:.2f}")
    print(f"Expected sells total:     {metrics['expected_sells_total']:.2f}")


def print_grid_report(result: pd.DataFrame, top: int) -> None:
    columns = [
        "half_spread",
        "avg_pnl_per_round",
        "ask_pnl_per_round",
        "bid_pnl_per_round",
        "ask_win_frequency",
        "bid_win_frequency",
        "adverse_ask_hit_frequency",
        "adverse_bid_hit_frequency",
        "expected_buys_per_round",
        "expected_sells_per_round",
    ]
    display = result.loc[:, columns].head(top).copy()
    percent_cols = [
        "ask_win_frequency",
        "bid_win_frequency",
        "adverse_ask_hit_frequency",
        "adverse_bid_hit_frequency",
    ]
    for column in percent_cols:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in set(display.columns).difference(percent_cols):
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    print(f"Top {min(top, len(display))} half-spreads by average PnL")
    print("=" * 43)
    print(display.to_string(index=False))


def run_self_test(csv_path: Path) -> None:
    df = load_history(csv_path)
    fill_models = build_fill_models(df)
    signals = df["mm_signal"].to_numpy(dtype=float)

    historical_mean = float(df["mm_pnl"].mean())
    assert_close(historical_mean, -50.58504, 1e-9, "historical mean mm_pnl")

    mm_bid = df["mm_bid"].to_numpy(dtype=float)
    mm_ask = df["mm_ask"].to_numpy(dtype=float)
    bot_asks = df.loc[:, BOT_ASK_COLS].to_numpy(dtype=float)
    bot_bids = df.loc[:, BOT_BID_COLS].to_numpy(dtype=float)
    best_ask_rounds = int((mm_ask < bot_asks.min(axis=1) - TIE_TOL).sum())
    best_bid_rounds = int((mm_bid > bot_bids.max(axis=1) + TIE_TOL).sum())
    if best_ask_rounds != 192:
        raise AssertionError(f"Expected 192 reference best-ask rounds, got {best_ask_rounds}")
    if best_bid_rounds != 169:
        raise AssertionError(f"Expected 169 reference best-bid rounds, got {best_bid_rounds}")

    assert_close(fill_models.ask.global_mean, 14.5625, 0.05, "ask fill global mean")
    assert_close(fill_models.bid.global_mean, 15.8343195266, 0.05, "bid fill global mean")

    bid, ask = formula_quotes(signals, alpha=0.9709, intercept=14.56, half_spread=90.0)
    validate_quotes(bid, ask)

    table = generate_formula_table(alpha=0.9709, intercept=14.56, half_spread=90.0)
    table_bid, table_ask = table_quotes(OFFICIAL_SIGNALS, table)
    if not np.allclose(table_bid, table["bid"].to_numpy(dtype=float)):
        raise AssertionError("Table interpolation changed exact bid values at official signals")
    if not np.allclose(table_ask, table["ask"].to_numpy(dtype=float)):
        raise AssertionError("Table interpolation changed exact ask values at official signals")

    print("Self-test passed.")
    print(f"Historical mean mm_pnl: {historical_mean:.5f}")
    print(f"Reference best-ask rounds: {best_ask_rounds}")
    print(f"Reference best-bid rounds: {best_bid_rounds}")
    print(f"Learned ask fill mean: {fill_models.ask.global_mean:.4f}")
    print(f"Learned bid fill mean: {fill_models.bid.global_mean:.4f}")


def assert_close(actual: float, expected: float, tol: float, label: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: expected {expected:.10f}, got {actual:.10f}, tolerance {tol}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay historical auction rounds against fixed house operators.",
    )
    parser.add_argument("--csv", type=Path, default=Path("auction_history.csv"))
    parser.add_argument("--alpha", type=float, default=0.9709)
    parser.add_argument("--intercept", type=float, default=14.56)
    parser.add_argument("--half-spread", type=float, default=90.0)
    parser.add_argument(
        "--grid",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help="Evaluate formula strategies across a half-spread grid.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        help="Evaluate a 21-row CSV table with signal,bid,ask columns.",
    )
    parser.add_argument(
        "--submission-table",
        action="store_true",
        help="Print the 21-row formula table for the current alpha/intercept/spread.",
    )
    parser.add_argument("--top", type=int, default=10, help="Rows to show for grid search.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test(args.csv)
        return

    if args.table and args.grid:
        raise SystemExit("--grid cannot be combined with --table")

    df = load_history(args.csv)
    fill_models = build_fill_models(df)

    if args.submission_table:
        table = generate_formula_table(args.alpha, args.intercept, args.half_spread)
        print("Formula submission table")
        print("========================")
        print(table.to_string(index=False))
        print()

    if args.grid:
        result = run_grid(
            df,
            fill_models,
            alpha=args.alpha,
            intercept=args.intercept,
            grid=tuple(args.grid),
        )
        print_grid_report(result, top=args.top)
        return

    signals = df["mm_signal"].to_numpy(dtype=float)
    if args.table:
        table = load_submission_table(args.table)
        bid, ask = table_quotes(signals, table)
        title = f"Historical evaluation for table: {args.table}"
    else:
        bid, ask = formula_quotes(
            signals,
            alpha=args.alpha,
            intercept=args.intercept,
            half_spread=args.half_spread,
        )
        title = (
            "Historical evaluation for formula "
            f"alpha={args.alpha:g}, intercept={args.intercept:g}, h={args.half_spread:g}"
        )

    metrics = evaluate_quotes(df, bid, ask, fill_models)
    print_single_report(title, metrics)


if __name__ == "__main__":
    main()
