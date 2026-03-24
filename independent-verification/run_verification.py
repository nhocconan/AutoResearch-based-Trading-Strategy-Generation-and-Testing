#!/usr/bin/env python3
"""
Independent verification runner for strategy audits.

This script intentionally avoids the project's existing backtest engine.
It loads each strategy, generates signals directly from the strategy code,
runs an independent t+1-open backtest, checks for look-ahead via prefix
stability, compares train/test claims from results.tsv, and renders a
standalone Markdown report plus static HTML dashboard.
"""

from __future__ import annotations

import argparse
import builtins
import concurrent.futures
import contextlib
import importlib.util
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
STRATEGIES_DIR = ROOT / "strategies"
RESULTS_FILE = ROOT / "results.tsv"
KLINES_DIR = ROOT / "data" / "processed" / "klines"
FUNDING_DIR = ROOT / "data" / "processed" / "funding"
DEFAULT_OUTPUT_DIR = ROOT / "independent-verification" / "output"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FULL_PERIOD_LABEL = "full"
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
START_DATE = pd.Timestamp("2021-01-01", tz="UTC")
FEE_PCT_PER_SIDE = 0.0004
SLIPPAGE_PCT_PER_SIDE = 0.0001
COST_PER_SIDE = FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE
INITIAL_CAPITAL = 10_000.0
RISK_FREE_RATE = 0.05
EPS = 1e-12

TIMEFRAME_TO_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

TIMEFRAME_BARS_PER_YEAR = {
    "1m": 365.25 * 24 * 60,
    "5m": 365.25 * 24 * 12,
    "15m": 365.25 * 24 * 4,
    "1h": 365.25 * 24,
    "4h": 365.25 * 6,
    "1d": 365.25,
}

@dataclass
class StrategyContext:
    file_name: str
    path: Path
    module_name: str
    strategy_name: str
    timeframe: str
    leverage: float
    static_flags: dict[str, Any]


@dataclass
class PrefixFailure:
    checkpoint: int
    max_abs_diff: float
    tail_window: int


@dataclass
class SignalRun:
    signals: np.ndarray
    io_attempts: list[str]


@dataclass
class TradeRecord:
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    pnl: float
    pnl_pct: float
    fee_cost: float
    funding_cost: float
    bars: int


PRICE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
GAP_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
CLAIMS_DF: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent strategy verification.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated artifacts.")
    parser.add_argument("--start-date", default="2021-01-01", help="Inclusive start date for full-period verification.")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS), help="Symbols to verify.")
    parser.add_argument("--strategies", nargs="*", default=None, help="Optional list of strategy file stems to verify.")
    parser.add_argument("--lookahead-symbol", default="ALL", help="Run prefix look-ahead audit on ALL symbols or one specific symbol.")
    parser.add_argument("--tail-check", type=int, default=3, help="Tail window size for prefix stability checks.")
    parser.add_argument("--max-prefix-checks", type=int, default=4, help="Maximum prefix checkpoints per strategy.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="Parallel workers across strategies.")
    return parser.parse_args()


def load_claims() -> pd.DataFrame:
    global CLAIMS_DF
    if CLAIMS_DF is None:
        claims = pd.read_csv(RESULTS_FILE, sep="\t")
        current = {path.stem for path in STRATEGIES_DIR.glob("*.py")}
        claims = claims[claims["strategy"].isin(current)].copy()
        CLAIMS_DF = claims
    return CLAIMS_DF.copy()


def strategy_files(filter_names: list[str] | None) -> list[Path]:
    paths = sorted(path for path in STRATEGIES_DIR.glob("*.py") if path.is_file())
    if not filter_names:
        return paths
    allowed = {name.removesuffix(".py") for name in filter_names}
    return [path for path in paths if path.stem in allowed]


def import_strategy(path: Path) -> StrategyContext:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    timeframe = getattr(module, "timeframe", None)
    if timeframe not in TIMEFRAME_TO_FREQ:
        raise ValueError(f"{path.name}: unsupported timeframe {timeframe!r}")
    strategy_name = getattr(module, "name", path.stem)
    leverage = float(getattr(module, "leverage", 1.0))
    static_flags = inspect_strategy_source(path)
    return StrategyContext(
        file_name=path.name,
        path=path,
        module_name=path.stem,
        strategy_name=strategy_name,
        timeframe=timeframe,
        leverage=leverage,
        static_flags=static_flags,
    )


def inspect_strategy_source(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    future_pattern = bool(
        re.search(r"shift\(\s*-\d+", text)
        or re.search(r"iloc\[\s*i\s*\+\s*1\s*\]", text)
        or re.search(r"\[\s*i\s*\+\s*1\s*\]", text)
    )
    resample_freqs = re.findall(r"resample\(\s*['\"]([^'\"]+)['\"]", text)
    external_data_patterns = bool(
        re.search(r"read_parquet|read_csv|read_json|read_pickle|read_excel|open\(", text)
    )
    mentions_cross_asset = bool(re.search(r"cross-asset|cross asset|market leader|leader", text, re.I))
    flags = {
        "uses_open_time": "open_time" in text,
        "fixed_date_range": (
            "pd.date_range(start='2021-01-01'" in text
            or 'pd.date_range(start="2021-01-01"' in text
        ),
        "synthetic_open_from_close": ("'open': close" in text or '"open": close' in text),
        "future_pattern_found": future_pattern,
        "reads_funding_rate": "funding_rate" in text,
        "has_resample": "resample(" in text,
        "resample_freqs": resample_freqs,
        "mentions_cross_asset": mentions_cross_asset,
        "external_data_patterns": external_data_patterns,
        "source_line_count": text.count("\n") + 1,
    }
    return flags


def load_prices(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    if key in PRICE_CACHE:
        return PRICE_CACHE[key].copy()

    path = KLINES_DIR / symbol / f"{timeframe}.parquet"
    prices = pd.read_parquet(path).copy()
    prices = prices.sort_values("open_time").reset_index(drop=True)
    prices = prices[prices["open_time"] >= START_DATE].reset_index(drop=True)
    prices = merge_funding(prices, symbol)
    PRICE_CACHE[key] = prices
    return prices.copy()


def merge_funding(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    path = FUNDING_DIR / symbol / "funding_rate.parquet"
    if not path.exists():
        prices["funding_rate"] = 0.0
        prices["funding_interval_hours"] = 8
        return prices

    funding = pd.read_parquet(path).copy()
    funding = funding.sort_values("calc_time").reset_index(drop=True)
    funding = funding.rename(columns={"calc_time": "funding_time", "last_funding_rate": "funding_rate"})

    merged = pd.merge_asof(
        prices.sort_values("open_time"),
        funding[["funding_time", "funding_rate", "funding_interval_hours"]].sort_values("funding_time"),
        left_on="open_time",
        right_on="funding_time",
        direction="backward",
    )
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    merged["funding_interval_hours"] = merged["funding_interval_hours"].fillna(8).astype(int)
    merged = merged.drop(columns=["funding_time"], errors="ignore")
    return merged


def gap_stats(symbol: str, timeframe: str) -> dict[str, Any]:
    key = (symbol, timeframe)
    if key in GAP_CACHE:
        return GAP_CACHE[key]

    prices = load_prices(symbol, timeframe)
    expected = pd.Timedelta(TIMEFRAME_TO_FREQ[timeframe])
    diffs = prices["open_time"].diff().dropna()
    gaps = diffs[diffs != expected]
    stats = {
        "expected_freq": TIMEFRAME_TO_FREQ[timeframe],
        "irregular_gap_count": int(len(gaps)),
        "max_gap": str(gaps.max()) if len(gaps) else "",
        "start_time": prices["open_time"].iloc[0].isoformat() if len(prices) else "",
        "end_time": prices["open_time"].iloc[-1].isoformat() if len(prices) else "",
    }
    GAP_CACHE[key] = stats
    return stats


@contextlib.contextmanager
def sandbox_external_io() -> Any:
    attempts: list[str] = []

    def block(label: str):
        def inner(*args: Any, **kwargs: Any) -> Any:
            detail = label
            if args:
                detail = f"{label}: {args[0]}"
            attempts.append(detail)
            raise RuntimeError(f"External IO blocked during generate_signals: {detail}")
        return inner

    patches: list[tuple[Any, str, Any]] = []

    def patch(obj: Any, attr: str, replacement: Any) -> None:
        if hasattr(obj, attr):
            patches.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, replacement)

    patch(pd, "read_parquet", block("pandas.read_parquet"))
    patch(pd, "read_csv", block("pandas.read_csv"))
    patch(pd, "read_json", block("pandas.read_json"))
    patch(pd, "read_pickle", block("pandas.read_pickle"))
    patch(pd, "read_excel", block("pandas.read_excel"))
    patch(np, "load", block("numpy.load"))
    patch(Path, "open", block("Path.open"))
    patch(builtins, "open", block("builtins.open"))

    try:
        import requests  # type: ignore
        patch(requests, "request", block("requests.request"))
    except Exception:
        pass

    try:
        yield attempts
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def generate_signals(module: Any, prices: pd.DataFrame) -> SignalRun:
    with sandbox_external_io() as attempts:
        signals = np.asarray(module.generate_signals(prices.copy()), dtype=float)
    return SignalRun(signals=signals, io_attempts=attempts)


def validate_signals(signals: np.ndarray, expected_len: int) -> dict[str, Any]:
    finite_ok = bool(np.isfinite(signals).all()) if len(signals) else True
    max_abs = float(np.max(np.abs(signals))) if len(signals) else 0.0
    return {
        "len_ok": len(signals) == expected_len,
        "finite_ok": finite_ok,
        "range_ok": max_abs <= 1.0 + 1e-9,
        "signal_abs_max": max_abs,
        "signal_unique_values": int(len(np.unique(np.round(signals, 8)))) if len(signals) else 0,
    }


def timeframe_ratio(base_timeframe: str, target_freq: str) -> int | None:
    try:
        base_td = pd.Timedelta(TIMEFRAME_TO_FREQ[base_timeframe])
        target_td = pd.Timedelta(target_freq)
    except Exception:
        return None
    if target_td <= base_td:
        return None
    ratio = target_td / base_td
    if abs(ratio - round(ratio)) > 1e-9:
        return None
    return int(round(ratio))


def higher_timeframe_ratios(ctx: StrategyContext) -> list[int]:
    ratios: set[int] = set()
    for freq in ctx.static_flags.get("resample_freqs", []):
        ratio = timeframe_ratio(ctx.timeframe, freq)
        if ratio and ratio > 1:
            ratios.add(ratio)
    return sorted(ratios)


def build_prefix_checkpoints(n: int, limit: int, ratios: list[int]) -> list[int]:
    base = [1000, 5000, 20000, n // 4, n // 2]
    for ratio in ratios:
        for blocks in (32, 64, 128):
            anchor = ratio * blocks
            for offset in (-1, 1, max(1, ratio // 2)):
                point = anchor + offset
                if point > 64:
                    base.append(point)
    checkpoints = sorted({value for value in base if 64 < value < n})
    if len(checkpoints) <= limit:
        return checkpoints
    idx = np.linspace(0, len(checkpoints) - 1, num=limit, dtype=int)
    return [checkpoints[i] for i in idx]


def run_prefix_check(
    ctx: StrategyContext,
    module: Any,
    prices: pd.DataFrame,
    full_signals: np.ndarray,
    tail_check: int,
    max_prefix_checks: int,
) -> tuple[bool, list[PrefixFailure]]:
    failures: list[PrefixFailure] = []
    ratios = higher_timeframe_ratios(ctx)
    checkpoints = build_prefix_checkpoints(len(prices), max_prefix_checks, ratios)
    if not checkpoints:
        return True, failures

    effective_tail = max([tail_check, *ratios, 3])
    for checkpoint in checkpoints:
        prefix_prices = prices.iloc[:checkpoint].copy()
        prefix_run = generate_signals(module, prefix_prices)
        prefix_signals = prefix_run.signals
        tail = min(effective_tail, checkpoint)
        prefix_tail = prefix_signals[-tail:]
        full_tail = full_signals[checkpoint - tail:checkpoint]
        max_abs_diff = float(np.max(np.abs(prefix_tail - full_tail)))
        if max_abs_diff > 1e-9:
            failures.append(PrefixFailure(checkpoint=checkpoint, max_abs_diff=max_abs_diff, tail_window=tail))
    return len(failures) == 0, failures


def bars_per_year(timeframe: str) -> float:
    return TIMEFRAME_BARS_PER_YEAR.get(timeframe, 365.25 * 24)


def backtest(
    prices: pd.DataFrame,
    signals: np.ndarray,
    leverage: float,
    timeframe: str,
) -> dict[str, Any]:
    start_ts = time.perf_counter()
    if len(prices) == 0:
        raise ValueError("Empty price frame")
    if len(prices) != len(signals):
        raise ValueError(f"Signal length mismatch: {len(signals)} != {len(prices)}")

    opens = prices["open"].to_numpy(dtype=float)
    closes = prices["close"].to_numpy(dtype=float)
    open_times = prices["open_time"]

    delayed = np.zeros(len(signals), dtype=float)
    delayed[1:] = signals[:-1]

    equity_curve = np.zeros(len(signals), dtype=float)
    returns = np.zeros(len(signals), dtype=float)
    equity_curve[0] = INITIAL_CAPITAL

    position = 0.0
    current_trade: dict[str, Any] | None = None
    trades: list[TradeRecord] = []

    for i in range(1, len(signals)):
        prev_equity = equity_curve[i - 1]
        target = float(delayed[i])
        open_px = opens[i]
        close_px = closes[i]
        bar_time = open_times.iloc[i]

        if not np.isfinite(prev_equity) or prev_equity <= 0:
            equity_curve[i:] = 0.0
            returns[i:] = -1.0
            break

        if abs(position) > EPS and (abs(target) <= EPS or position * target < 0):
            exit_fee = abs(position) * leverage * COST_PER_SIDE * prev_equity
            if current_trade is not None:
                current_trade["fee_cost"] += exit_fee
                close_trade(current_trade, trades, bar_time, open_px)
                current_trade = None
        elif abs(position) > EPS and position * target > 0 and abs(target - position) > EPS and current_trade is not None:
            adjust_fee = abs(target - position) * leverage * COST_PER_SIDE * prev_equity
            current_trade["fee_cost"] += adjust_fee

        if abs(position) <= EPS and abs(target) > EPS:
            entry_fee = abs(target) * leverage * COST_PER_SIDE * prev_equity
            current_trade = {
                "direction": "LONG" if target > 0 else "SHORT",
                "entry_time": bar_time,
                "entry_price": open_px,
                "size": abs(target),
                "gross_pnl": 0.0,
                "fee_cost": entry_fee,
                "funding_cost": 0.0,
                "bars": 0,
                "entry_notional": max(prev_equity * abs(target) * leverage, EPS),
            }
        elif abs(position) > EPS and position * target < 0 and abs(target) > EPS:
            entry_fee = abs(target) * leverage * COST_PER_SIDE * prev_equity
            current_trade = {
                "direction": "LONG" if target > 0 else "SHORT",
                "entry_time": bar_time,
                "entry_price": open_px,
                "size": abs(target),
                "gross_pnl": 0.0,
                "fee_cost": entry_fee,
                "funding_cost": 0.0,
                "bars": 0,
                "entry_notional": max(prev_equity * abs(target) * leverage, EPS),
            }
        elif abs(position) > EPS and position * target > 0 and abs(target - position) > EPS and current_trade is not None:
            current_trade["size"] = abs(target)

        fee = abs(target - position) * leverage * COST_PER_SIDE * prev_equity
        equity_after_fee = prev_equity - fee

        if abs(target) > EPS and open_px > 0:
            bar_return = ((close_px - open_px) / open_px) * target * leverage
            pnl = equity_after_fee * bar_return
        else:
            pnl = 0.0

        equity = max(equity_after_fee + pnl, 0.0)
        equity_curve[i] = equity
        returns[i] = (equity / prev_equity - 1.0) if prev_equity > 0 else 0.0

        if current_trade is not None and abs(target) > EPS:
            current_trade["gross_pnl"] += pnl
            current_trade["bars"] += 1

        position = target

    if current_trade is not None and abs(position) > EPS:
        final_fee = abs(position) * leverage * COST_PER_SIDE * equity_curve[-1]
        equity_curve[-1] = max(equity_curve[-1] - final_fee, 0.0)
        if len(equity_curve) > 1 and equity_curve[-2] > 0:
            returns[-1] = equity_curve[-1] / equity_curve[-2] - 1.0
        current_trade["fee_cost"] += final_fee
        close_trade(current_trade, trades, open_times.iloc[-1], closes[-1])

    metrics = compute_metrics(equity_curve, returns, trades, timeframe, time.perf_counter() - start_ts)
    metrics["final_equity"] = float(equity_curve[-1])
    metrics["period_start"] = open_times.iloc[0].isoformat()
    metrics["period_end"] = open_times.iloc[-1].isoformat()
    return metrics


def close_trade(current_trade: dict[str, Any], trades: list[TradeRecord], exit_time: pd.Timestamp, exit_price: float) -> None:
    net_pnl = current_trade["gross_pnl"] - current_trade["fee_cost"] - current_trade["funding_cost"]
    trades.append(
        TradeRecord(
            direction=current_trade["direction"],
            entry_time=current_trade["entry_time"].isoformat(),
            exit_time=exit_time.isoformat(),
            entry_price=float(current_trade["entry_price"]),
            exit_price=float(exit_price),
            size=float(current_trade["size"]),
            gross_pnl=float(current_trade["gross_pnl"]),
            pnl=float(net_pnl),
            pnl_pct=float(net_pnl / current_trade["entry_notional"] * 100.0),
            fee_cost=float(current_trade["fee_cost"]),
            funding_cost=float(current_trade["funding_cost"]),
            bars=int(current_trade["bars"]),
        )
    )


def compute_metrics(
    equity_curve: np.ndarray,
    returns: np.ndarray,
    trades: list[TradeRecord],
    timeframe: str,
    runtime_s: float,
) -> dict[str, Any]:
    bpy = bars_per_year(timeframe)
    rf_per_bar = RISK_FREE_RATE / bpy
    initial = float(equity_curve[0]) if len(equity_curve) else INITIAL_CAPITAL
    final = float(equity_curve[-1]) if len(equity_curve) else INITIAL_CAPITAL
    total_return_pct = (final / initial - 1.0) * 100.0 if initial > 0 else 0.0
    n_years = len(equity_curve) / bpy if bpy > 0 else 0.0
    cagr_pct = ((final / initial) ** (1 / n_years) - 1.0) * 100.0 if initial > 0 and final > 0 and n_years > 0 else 0.0

    ret_std = float(np.std(returns))
    mean_excess = float(np.mean(returns - rf_per_bar))
    sharpe = mean_excess / ret_std * math.sqrt(bpy) if ret_std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) else 0.0
    sortino = mean_excess / downside_std * math.sqrt(bpy) if downside_std > 0 else (float("inf") if mean_excess > 0 else 0.0)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = np.where(peak > 0, (equity_curve - peak) / peak, 0.0)
    max_dd_pct = float(np.min(drawdown) * 100.0)
    calmar = cagr_pct / abs(max_dd_pct) if max_dd_pct < 0 else (float("inf") if cagr_pct > 0 else 0.0)

    pnls = np.array([trade.pnl for trade in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(trades) * 100.0) if trades else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    exposure_pct = float(np.mean(np.abs(returns) > EPS) * 100.0) if len(returns) else 0.0
    total_fees = float(sum(trade.fee_cost for trade in trades))

    return {
        "return_pct": float(total_return_pct),
        "cagr_pct": float(cagr_pct),
        "sharpe": float(sharpe),
        "sortino": float(sortino if np.isfinite(sortino) else 999.0),
        "calmar": float(calmar if np.isfinite(calmar) else 999.0),
        "max_dd_pct": float(max_dd_pct),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999.0),
        "trades": int(len(trades)),
        "exposure_pct": float(exposure_pct),
        "total_fees": float(total_fees),
        "runtime_s": float(runtime_s),
        "num_bars": int(len(equity_curve)),
    }


def subset_period(prices: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "train":
        return prices[prices["open_time"] < TRAIN_END].reset_index(drop=True)
    if period == "test":
        return prices[prices["open_time"] >= TRAIN_END].reset_index(drop=True)
    if period == FULL_PERIOD_LABEL:
        return prices.reset_index(drop=True)
    raise ValueError(f"Unknown period {period}")


def period_slice(values: np.ndarray, prices: pd.DataFrame, period: str) -> np.ndarray:
    if period == FULL_PERIOD_LABEL:
        return values
    if period == "train":
        mask = prices["open_time"] < TRAIN_END
        return values[mask.to_numpy()]
    if period == "test":
        mask = prices["open_time"] >= TRAIN_END
        return values[mask.to_numpy()]
    raise ValueError(f"Unknown period {period}")


def lookup_claim(claims: pd.DataFrame, strategy_name: str, symbol: str, period: str) -> dict[str, Any]:
    rows = claims[
        (claims["strategy"] == strategy_name)
        & (claims["symbol"] == symbol)
        & (claims["period"] == period)
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "claimed_sharpe": float(row["sharpe"]),
        "claimed_return_pct": float(row["return_pct"]),
        "claimed_max_dd_pct": float(row["max_dd_pct"]),
        "claimed_trades": int(row["trades"]),
        "claimed_status": str(row["status"]),
        "claimed_description": str(row["description"]),
    }


def concerns_from_row(row: dict[str, Any]) -> list[str]:
    concerns: list[str] = []
    if row.get("error"):
        concerns.append("execution_error")
    if not row.get("signal_len_ok", True):
        concerns.append("signal_length_mismatch")
    if not row.get("signal_finite_ok", True):
        concerns.append("non_finite_signal_values")
    if not row.get("signal_range_ok", True):
        concerns.append("signal_outside_minus1_plus1")
    if row.get("lookahead_pass") is False:
        concerns.append("prefix_lookahead_failure")
    if row.get("uses_fixed_date_range"):
        concerns.append("synthetic_date_index")
    if row.get("uses_synthetic_open_from_close"):
        concerns.append("synthetic_open_from_close")
    if row.get("uses_future_pattern_found"):
        concerns.append("static_future_pattern")
    if not row.get("io_sandbox_pass", True):
        concerns.append("external_io_attempt")
    if row.get("mentions_cross_asset"):
        concerns.append("cross_asset_claim")
    if row.get("irregular_gap_count", 0) > 0 and row.get("uses_fixed_date_range"):
        concerns.append("synthetic_resample_on_gappy_data")
    if row.get("return_match") is False:
        concerns.append("claim_return_mismatch")
    return concerns


def verify_strategy_symbol(
    ctx: StrategyContext,
    module: Any,
    symbol: str,
    claims: pd.DataFrame,
    lookahead_symbol: str,
    tail_check: int,
    max_prefix_checks: int,
) -> list[dict[str, Any]]:
    prices = load_prices(symbol, ctx.timeframe)
    rows: list[dict[str, Any]] = []
    gap_info = gap_stats(symbol, ctx.timeframe)

    full_prices = subset_period(prices, FULL_PERIOD_LABEL)
    full_run = generate_signals(module, full_prices)
    full_signals = full_run.signals
    signal_validation = validate_signals(full_signals, len(full_prices))
    io_sandbox_pass = len(full_run.io_attempts) == 0

    lookahead_pass = True
    prefix_failures: list[PrefixFailure] = []
    if lookahead_symbol in {"ALL", "*"} or symbol == lookahead_symbol:
        lookahead_pass, prefix_failures = run_prefix_check(
            ctx,
            module,
            full_prices,
            full_signals,
            tail_check=tail_check,
            max_prefix_checks=max_prefix_checks,
        )

    for period in (FULL_PERIOD_LABEL, "train", "test"):
        period_prices = subset_period(prices, period)
        if len(period_prices) < 2:
            row = build_base_row(ctx, symbol, period, gap_info, signal_validation, lookahead_pass, prefix_failures, io_sandbox_pass, full_run.io_attempts)
            row["error"] = f"Not enough bars in {period}"
            row["concerns"] = ";".join(concerns_from_row(row))
            rows.append(row)
            continue

        try:
            period_signals = period_slice(full_signals, prices, period)
            validation = signal_validation if period == FULL_PERIOD_LABEL else validate_signals(period_signals, len(period_prices))
            metrics = backtest(period_prices, period_signals, ctx.leverage, ctx.timeframe)

            row = build_base_row(ctx, symbol, period, gap_info, validation, lookahead_pass, prefix_failures, io_sandbox_pass, full_run.io_attempts)
            row.update(
                {
                    "period_start": metrics["period_start"],
                    "period_end": metrics["period_end"],
                    "num_bars": metrics["num_bars"],
                    "ind_return_pct": metrics["return_pct"],
                    "ind_cagr_pct": metrics["cagr_pct"],
                    "ind_sharpe": metrics["sharpe"],
                    "ind_sortino": metrics["sortino"],
                    "ind_calmar": metrics["calmar"],
                    "ind_max_dd_pct": metrics["max_dd_pct"],
                    "ind_win_rate": metrics["win_rate"],
                    "ind_profit_factor": metrics["profit_factor"],
                    "ind_trades": metrics["trades"],
                    "ind_exposure_pct": metrics["exposure_pct"],
                    "ind_final_equity": metrics["final_equity"],
                    "ind_total_fees": metrics["total_fees"],
                    "backtest_runtime_s": metrics["runtime_s"],
                    "signal_history_mode": "full_history_slice",
                }
            )

            if period in {"train", "test"}:
                claim = lookup_claim(claims, ctx.strategy_name, symbol, period)
                row.update(claim)
                if claim:
                    row["return_diff_pct_pts"] = row["ind_return_pct"] - row["claimed_return_pct"]
                    row["sharpe_diff"] = row["ind_sharpe"] - row["claimed_sharpe"]
                    row["dd_diff_pct_pts"] = row["ind_max_dd_pct"] - row["claimed_max_dd_pct"]
                    row["trades_diff"] = row["ind_trades"] - row["claimed_trades"]
                    row["return_match"] = abs(row["return_diff_pct_pts"]) <= 5.0
                    row["sharpe_match"] = abs(row["sharpe_diff"]) <= 0.5
                else:
                    row["return_match"] = None
                    row["sharpe_match"] = None

            row["concerns"] = ";".join(concerns_from_row(row))
            rows.append(row)
        except Exception as exc:
            row = build_base_row(ctx, symbol, period, gap_info, signal_validation, lookahead_pass, prefix_failures, io_sandbox_pass, full_run.io_attempts)
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=5)
            row["concerns"] = ";".join(concerns_from_row(row))
            rows.append(row)
    return rows


def verify_strategy_path(
    path_str: str,
    symbols: list[str],
    lookahead_symbol: str,
    tail_check: int,
    max_prefix_checks: int,
) -> list[dict[str, Any]]:
    path = Path(path_str)
    try:
        ctx = import_strategy(path)
        spec = importlib.util.spec_from_file_location(ctx.module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not reload {path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        claims = load_claims()
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            rows.extend(
                verify_strategy_symbol(
                    ctx=ctx,
                    module=module,
                    symbol=symbol,
                    claims=claims,
                    lookahead_symbol=lookahead_symbol,
                    tail_check=tail_check,
                    max_prefix_checks=max_prefix_checks,
                )
            )
        return rows
    except Exception as exc:
        return [
            {
                "strategy_file": path.name,
                "strategy_name": path.stem,
                "module_name": path.stem,
                "symbol": "",
                "timeframe": "",
                "period": "",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=10),
                "concerns": "strategy_import_error",
            }
        ]


def build_base_row(
    ctx: StrategyContext,
    symbol: str,
    period: str,
    gap_info: dict[str, Any],
    validation: dict[str, Any],
    lookahead_pass: bool,
    prefix_failures: list[PrefixFailure],
    io_sandbox_pass: bool,
    io_attempts: list[str],
) -> dict[str, Any]:
    return {
        "strategy_file": ctx.file_name,
        "strategy_name": ctx.strategy_name,
        "module_name": ctx.module_name,
        "symbol": symbol,
        "timeframe": ctx.timeframe,
        "leverage": ctx.leverage,
        "period": period,
        "uses_open_time": ctx.static_flags["uses_open_time"],
        "uses_fixed_date_range": ctx.static_flags["fixed_date_range"],
        "uses_synthetic_open_from_close": ctx.static_flags["synthetic_open_from_close"],
        "uses_future_pattern_found": ctx.static_flags["future_pattern_found"],
        "reads_funding_rate": ctx.static_flags["reads_funding_rate"],
        "has_resample": ctx.static_flags["has_resample"],
        "mentions_cross_asset": ctx.static_flags["mentions_cross_asset"],
        "external_data_patterns": ctx.static_flags["external_data_patterns"],
        "resample_freqs": json.dumps(ctx.static_flags["resample_freqs"]),
        "higher_tf_ratios": json.dumps(higher_timeframe_ratios(ctx)),
        "signal_len_ok": validation["len_ok"],
        "signal_finite_ok": validation["finite_ok"],
        "signal_range_ok": validation["range_ok"],
        "signal_abs_max": validation["signal_abs_max"],
        "signal_unique_values": validation["signal_unique_values"],
        "lookahead_pass": lookahead_pass,
        "lookahead_failure_count": len(prefix_failures),
        "lookahead_failures": json.dumps([failure.__dict__ for failure in prefix_failures]),
        "io_sandbox_pass": io_sandbox_pass,
        "io_attempt_count": len(io_attempts),
        "io_attempts": json.dumps(io_attempts),
        "irregular_gap_count": gap_info["irregular_gap_count"],
        "max_gap": gap_info["max_gap"],
        "data_start_time": gap_info["start_time"],
        "data_end_time": gap_info["end_time"],
        "return_match": None,
        "sharpe_match": None,
        "error": "",
        "traceback": "",
    }


def build_strategy_summary(results: pd.DataFrame) -> pd.DataFrame:
    full_rows = results[results["period"] == FULL_PERIOD_LABEL].copy()
    summaries: list[dict[str, Any]] = []
    for strategy_name, group in full_rows.groupby("strategy_name", sort=True):
        summary = {
            "strategy_name": strategy_name,
            "timeframe": group["timeframe"].iloc[0],
            "lookahead_pass": bool(group["lookahead_pass"].all()),
            "signal_range_ok": bool(group["signal_range_ok"].all()),
            "io_sandbox_pass": bool(group["io_sandbox_pass"].all()),
            "uses_fixed_date_range": bool(group["uses_fixed_date_range"].any()),
            "uses_synthetic_open_from_close": bool(group["uses_synthetic_open_from_close"].any()),
            "mentions_cross_asset": bool(group["mentions_cross_asset"].any()),
            "irregular_gap_count_max": int(group["irregular_gap_count"].max()),
            "avg_full_return_pct": float(group["ind_return_pct"].mean()),
            "avg_full_sharpe": float(group["ind_sharpe"].mean()),
            "avg_full_max_dd_pct": float(group["ind_max_dd_pct"].mean()),
            "avg_full_trades": float(group["ind_trades"].mean()),
            "error_count": int((group["error"] != "").sum()),
        }
        summary["severity"] = classify_strategy_severity(group)
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame(
            columns=[
                "strategy_name",
                "timeframe",
                "lookahead_pass",
                "signal_range_ok",
                "io_sandbox_pass",
                "uses_fixed_date_range",
                "uses_synthetic_open_from_close",
                "mentions_cross_asset",
                "irregular_gap_count_max",
                "avg_full_return_pct",
                "avg_full_sharpe",
                "avg_full_max_dd_pct",
                "avg_full_trades",
                "error_count",
                "severity",
            ]
        )
    return pd.DataFrame(summaries).sort_values(["severity", "avg_full_sharpe"], ascending=[True, False]).reset_index(drop=True)


def classify_strategy_severity(group: pd.DataFrame) -> str:
    if (group["error"] != "").any():
        return "error"
    if not group["lookahead_pass"].all():
        return "critical"
    if not group["io_sandbox_pass"].all():
        return "critical"
    if not group["signal_range_ok"].all():
        return "warning"
    if group["uses_fixed_date_range"].any() and group["irregular_gap_count"].max() > 0:
        return "warning"
    return "ok"


def build_dataset_summary(symbols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in ("15m", "1h", "4h"):
            gap = gap_stats(symbol, timeframe)
            prices = load_prices(symbol, timeframe)
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": int(len(prices)),
                    "start_time": gap["start_time"],
                    "end_time": gap["end_time"],
                    "irregular_gap_count": gap["irregular_gap_count"],
                    "max_gap": gap["max_gap"],
                }
            )
    return rows


def render_report(results: pd.DataFrame, strategy_summary: pd.DataFrame, dataset_summary: list[dict[str, Any]]) -> str:
    full_rows = results[results["period"] == FULL_PERIOD_LABEL].copy()
    compare_rows = results[results["period"].isin(["train", "test"])].copy()

    total_strategies = int(strategy_summary["strategy_name"].nunique())
    lookahead_fail_strats = int((strategy_summary["lookahead_pass"] == False).sum())  # noqa: E712
    signal_range_fail_strats = int((strategy_summary["signal_range_ok"] == False).sum())  # noqa: E712
    synthetic_index_strats = int(strategy_summary["uses_fixed_date_range"].sum())
    mismatch_rows = compare_rows[compare_rows["return_match"] == False].copy()  # noqa: E712

    top_mismatches = mismatch_rows.assign(abs_return_diff=np.abs(mismatch_rows["return_diff_pct_pts"])).sort_values(
        "abs_return_diff", ascending=False
    ).head(15)
    top_full = full_rows.sort_values("ind_sharpe", ascending=False).head(15)

    lines = []
    lines.append("# Independent Verification Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Train/test comparisons use signal slices from the full-history run, so indicators keep realistic warmup instead of resetting at period boundaries.")
    lines.append(f"- Engine execution model: signal[t] -> fill at bar[t+1] open, taker fee 0.04% + slippage 0.01% per side.")
    lines.append(f"- Strategies verified: {total_strategies}")
    lines.append(f"- Full-period rows: {len(full_rows)}")
    lines.append(f"- Train/test comparison rows: {len(compare_rows)}")
    lines.append(f"- Look-ahead failures: {lookahead_fail_strats} strategies")
    lines.append(f"- Signal range violations: {signal_range_fail_strats} strategies")
    lines.append(f"- External IO violations inside generate_signals(): {int((strategy_summary['io_sandbox_pass'] == False).sum())} strategies")
    lines.append(f"- Synthetic date-index strategies: {synthetic_index_strats} strategies")
    lines.append(f"- Claim return mismatches (>5 pct-pts): {len(mismatch_rows)} rows")
    lines.append("")
    lines.append("## Dataset Notes")
    lines.append("")
    for item in dataset_summary:
        lines.append(
            f"- {item['symbol']} {item['timeframe']}: {item['rows']} bars, "
            f"{item['start_time']} -> {item['end_time']}, irregular_gaps={item['irregular_gap_count']}, max_gap={item['max_gap'] or 'none'}"
        )
    lines.append("")
    lines.append("## Top Full-Period Independent Results")
    lines.append("")
    lines.append("| Strategy | Symbol | TF | Return | Sharpe | Max DD | Trades | Concerns |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for _, row in top_full.iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['symbol']} | {row['timeframe']} | "
            f"{row['ind_return_pct']:+.2f}% | {row['ind_sharpe']:.3f} | {row['ind_max_dd_pct']:.2f}% | "
            f"{int(row['ind_trades'])} | {row['concerns'] or '-'} |"
        )
    lines.append("")
    lines.append("## Largest Claim Mismatches")
    lines.append("")
    lines.append("| Strategy | Symbol | Period | Independent Return | Claimed Return | Diff | Independent Sharpe | Claimed Sharpe |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, row in top_mismatches.iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['symbol']} | {row['period']} | "
            f"{row['ind_return_pct']:+.2f}% | {row['claimed_return_pct']:+.2f}% | {row['return_diff_pct_pts']:+.2f} | "
            f"{row['ind_sharpe']:.3f} | {row['claimed_sharpe']:.3f} |"
        )
    lines.append("")
    lines.append("## Highest-Risk Strategy Flags")
    lines.append("")
    risk_rows = strategy_summary.sort_values(["severity", "avg_full_sharpe"], ascending=[True, False])
    lines.append("| Strategy | Severity | TF | Look-ahead | Signal Range | Synthetic Index | Avg Full Sharpe | Avg Full Return |")
    lines.append("|---|---|---:|---|---|---|---:|---:|")
    for _, row in risk_rows.head(25).iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['severity']} | {row['timeframe']} | "
            f"{'pass' if row['lookahead_pass'] else 'fail'} | "
            f"{'pass' if row['signal_range_ok'] else 'fail'} | "
            f"{'yes' if row['uses_fixed_date_range'] else 'no'} | "
            f"{row['avg_full_sharpe']:.3f} | {row['avg_full_return_pct']:+.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def render_dashboard(results: pd.DataFrame, strategy_summary: pd.DataFrame, dataset_summary: list[dict[str, Any]]) -> str:
    compare_rows = results[results["period"].isin(["train", "test"])].copy()
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_count": int(strategy_summary["strategy_name"].nunique()),
        "full_rows": int((results["period"] == FULL_PERIOD_LABEL).sum()),
        "comparison_rows": int(len(compare_rows)),
        "lookahead_fail_strategies": int((strategy_summary["lookahead_pass"] == False).sum()),  # noqa: E712
        "signal_range_fail_strategies": int((strategy_summary["signal_range_ok"] == False).sum()),  # noqa: E712
        "synthetic_index_strategies": int(strategy_summary["uses_fixed_date_range"].sum()),
        "claim_return_mismatches": int((compare_rows["return_match"] == False).sum()),  # noqa: E712
    }

    full_rows = results[results["period"] == FULL_PERIOD_LABEL].copy()
    top_full = full_rows.sort_values("ind_sharpe", ascending=False).head(20)
    top_mismatch = compare_rows.assign(abs_return_diff=np.abs(compare_rows["return_diff_pct_pts"].fillna(0.0))).sort_values(
        "abs_return_diff", ascending=False
    ).head(20)

    payload = {
        "summary": summary,
        "dataset": dataset_summary,
        "full_rows": [{k: json_safe(v) for k, v in row.items()} for row in full_rows.to_dict(orient="records")],
        "compare_rows": [{k: json_safe(v) for k, v in row.items()} for row in compare_rows.to_dict(orient="records")],
        "strategy_summary": [{k: json_safe(v) for k, v in row.items()} for row in strategy_summary.to_dict(orient="records")],
        "top_full": [{k: json_safe(v) for k, v in row.items()} for row in top_full.to_dict(orient="records")],
        "top_mismatch": [{k: json_safe(v) for k, v in row.items()} for row in top_mismatch.to_dict(orient="records")],
    }
    payload_json = json.dumps(payload, ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Independent Verification Dashboard</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --panel: #fffaf0;
      --ink: #1e1a16;
      --muted: #6b6258;
      --line: #d8cdbd;
      --good: #146c43;
      --warn: #b25f15;
      --bad: #9d2b25;
      --accent: #0f766e;
      --shadow: 0 12px 40px rgba(32, 24, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(178,95,21,0.12), transparent 22%),
        linear-gradient(180deg, #f9f4ea 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15,118,110,0.95), rgba(7,52,63,0.95));
      color: white;
      border-radius: 20px;
      padding: 28px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 5vw, 44px);
      line-height: 1.05;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 880px;
      color: rgba(255,255,255,0.86);
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .metric-value {{
      font-size: 34px;
      font-weight: 700;
      letter-spacing: -0.04em;
      margin-top: 8px;
    }}
    .metric-label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
    }}
    .section {{
      margin-top: 24px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: -0.02em;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1.4fr 1fr 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }}
    .toolbar input, .toolbar select {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .mono {{
      font-family: "SFMono-Regular", ui-monospace, Menlo, monospace;
      font-size: 12px;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      border: 1px solid currentColor;
      margin-right: 6px;
      margin-bottom: 4px;
      white-space: nowrap;
    }}
    .good {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    .scroll {{
      overflow-x: auto;
    }}
    .tiny {{
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .toolbar {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Independent Verification Dashboard</h1>
      <p>Separate audit of every strategy in <span class="mono">strategies/</span> using raw parquet OHLCV, optional funding merge, prefix look-ahead checks, and a custom t+1 open execution model. This dashboard is generated without the project's existing backtest engine.</p>
      <div class="grid" id="summary-grid"></div>
    </section>

    <section class="section">
      <div class="grid">
        <div class="card">
          <h2>Dataset</h2>
          <div class="scroll">
            <table id="dataset-table"></table>
          </div>
        </div>
        <div class="card">
          <h2>Top Full-Period Results</h2>
          <div class="scroll">
            <table id="top-full-table"></table>
          </div>
        </div>
      </div>
    </section>

    <section class="section card">
      <h2>Largest Claim Mismatches</h2>
      <div class="scroll">
        <table id="mismatch-table"></table>
      </div>
    </section>

    <section class="section card">
      <h2>Strategy Summary</h2>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Filter by strategy name or concern">
        <select id="severity-filter">
          <option value="all">All severities</option>
          <option value="error">error</option>
          <option value="critical">critical</option>
          <option value="warning">warning</option>
          <option value="ok">ok</option>
        </select>
        <select id="lookahead-filter">
          <option value="all">All look-ahead states</option>
          <option value="pass">look-ahead pass</option>
          <option value="fail">look-ahead fail</option>
        </select>
        <select id="synthetic-filter">
          <option value="all">All index modes</option>
          <option value="yes">synthetic index</option>
          <option value="no">real open_time</option>
        </select>
      </div>
      <p class="tiny" style="margin-top:10px;">The summary table is aggregated from full-period independent runs only. Detailed row-level output remains in the CSV/JSON artifacts next to this dashboard.</p>
      <div class="scroll">
        <table id="strategy-summary-table"></table>
      </div>
    </section>
  </main>

  <script>
    const payload = {payload_json};

    function esc(value) {{
      return String(value ?? "").replace(/[&<>"]/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}}[ch]));
    }}

    function badge(text, cls) {{
      return `<span class="badge ${{cls}}">${{esc(text)}}</span>`;
    }}

    function renderSummary() {{
      const entries = [
        ["Strategies", payload.summary.strategy_count, "good"],
        ["Full Rows", payload.summary.full_rows, "good"],
        ["Claim Mismatches", payload.summary.claim_return_mismatches, payload.summary.claim_return_mismatches ? "warn" : "good"],
        ["Look-ahead Fails", payload.summary.lookahead_fail_strategies, payload.summary.lookahead_fail_strategies ? "bad" : "good"],
        ["Signal Range Fails", payload.summary.signal_range_fail_strategies, payload.summary.signal_range_fail_strategies ? "warn" : "good"],
        ["Synthetic Index", payload.summary.synthetic_index_strategies, payload.summary.synthetic_index_strategies ? "warn" : "good"],
      ];
      document.getElementById("summary-grid").innerHTML = entries.map(([label, value, cls]) => `
        <div class="card">
          <div class="metric-label">${{esc(label)}}</div>
          <div class="metric-value ${{cls}}">${{esc(value)}}</div>
        </div>
      `).join("");
    }}

    function renderTable(elId, headers, rows) {{
      const table = document.getElementById(elId);
      table.innerHTML = `
        <thead><tr>${{headers.map((h) => `<th>${{esc(h.label)}}</th>`).join("")}}</tr></thead>
        <tbody>${{rows.map((row) => `<tr>${{headers.map((h) => `<td>${{h.render(row)}}</td>`).join("")}}</tr>`).join("")}}</tbody>
      `;
    }}

    function renderDataset() {{
      renderTable("dataset-table", [
        {{ label: "Symbol", render: (r) => esc(r.symbol) }},
        {{ label: "TF", render: (r) => esc(r.timeframe) }},
        {{ label: "Rows", render: (r) => esc(r.rows) }},
        {{ label: "Range", render: (r) => `<span class="mono">${{esc(r.start_time)}}<br>${{esc(r.end_time)}}</span>` }},
        {{ label: "Gaps", render: (r) => r.irregular_gap_count ? badge(`${{r.irregular_gap_count}} gap(s)`, "warn") + `<span class="mono">${{esc(r.max_gap)}}</span>` : badge("none", "good") }},
      ], payload.dataset);
    }}

    function renderTopFull() {{
      renderTable("top-full-table", [
        {{ label: "Strategy", render: (r) => `<div><strong>${{esc(r.strategy_name)}}</strong><div class="tiny">${{esc(r.symbol)}} · ${{esc(r.timeframe)}}</div></div>` }},
        {{ label: "Return", render: (r) => `<span class="${{r.ind_return_pct >= 0 ? "good" : "bad"}}">${{r.ind_return_pct >= 0 ? "+" : ""}}${{Number(r.ind_return_pct).toFixed(2)}}%</span>` }},
        {{ label: "Sharpe", render: (r) => `<span class="${{r.ind_sharpe >= 0 ? "good" : "bad"}}">${{Number(r.ind_sharpe).toFixed(3)}}</span>` }},
        {{ label: "DD", render: (r) => `<span class="${{r.ind_max_dd_pct >= -20 ? "good" : "warn"}}">${{Number(r.ind_max_dd_pct).toFixed(2)}}%</span>` }},
        {{ label: "Concerns", render: (r) => (r.concerns ? r.concerns.split(";").map((item) => badge(item, "warn")).join("") : badge("none", "good")) }},
      ], payload.top_full);
    }}

    function renderMismatch() {{
      renderTable("mismatch-table", [
        {{ label: "Strategy", render: (r) => `<strong>${{esc(r.strategy_name)}}</strong><div class="tiny">${{esc(r.symbol)}} · ${{esc(r.period)}}</div>` }},
        {{ label: "Independent Return", render: (r) => `${{r.ind_return_pct >= 0 ? "+" : ""}}${{Number(r.ind_return_pct).toFixed(2)}}%` }},
        {{ label: "Claimed Return", render: (r) => `${{r.claimed_return_pct >= 0 ? "+" : ""}}${{Number(r.claimed_return_pct).toFixed(2)}}%` }},
        {{ label: "Diff", render: (r) => `<span class="${{Math.abs(Number(r.return_diff_pct_pts)) > 5 ? "bad" : "good"}}">${{Number(r.return_diff_pct_pts).toFixed(2)}} pts</span>` }},
        {{ label: "Sharpe Diff", render: (r) => `${{Number(r.sharpe_diff).toFixed(3)}}` }},
        {{ label: "Concerns", render: (r) => (r.concerns ? r.concerns.split(";").map((item) => badge(item, "warn")).join("") : badge("none", "good")) }},
      ], payload.top_mismatch);
    }}

    function renderStrategySummary() {{
      const search = document.getElementById("search").value.trim().toLowerCase();
      const severityFilter = document.getElementById("severity-filter").value;
      const lookaheadFilter = document.getElementById("lookahead-filter").value;
      const syntheticFilter = document.getElementById("synthetic-filter").value;

      const rows = payload.strategy_summary.filter((row) => {{
        if (severityFilter !== "all" && row.severity !== severityFilter) return false;
        if (lookaheadFilter === "pass" && !row.lookahead_pass) return false;
        if (lookaheadFilter === "fail" && row.lookahead_pass) return false;
        if (syntheticFilter === "yes" && !row.uses_fixed_date_range) return false;
        if (syntheticFilter === "no" && row.uses_fixed_date_range) return false;
        if (!search) return true;
        const hay = `${{row.strategy_name}} ${{row.severity}} ${{row.timeframe}}`.toLowerCase();
        return hay.includes(search);
      }});

      renderTable("strategy-summary-table", [
        {{ label: "Strategy", render: (r) => `<strong>${{esc(r.strategy_name)}}</strong><div class="tiny">${{esc(r.timeframe)}}</div>` }},
        {{ label: "Severity", render: (r) => badge(r.severity, r.severity === "ok" ? "good" : (r.severity === "warning" ? "warn" : "bad")) }},
        {{ label: "Look-ahead", render: (r) => badge(r.lookahead_pass ? "pass" : "fail", r.lookahead_pass ? "good" : "bad") }},
        {{ label: "Signal Range", render: (r) => badge(r.signal_range_ok ? "pass" : "fail", r.signal_range_ok ? "good" : "warn") }},
        {{ label: "Index Mode", render: (r) => badge(r.uses_fixed_date_range ? "synthetic" : "open_time", r.uses_fixed_date_range ? "warn" : "good") }},
        {{ label: "Avg Full Sharpe", render: (r) => Number(r.avg_full_sharpe).toFixed(3) }},
        {{ label: "Avg Full Return", render: (r) => `${{r.avg_full_return_pct >= 0 ? "+" : ""}}${{Number(r.avg_full_return_pct).toFixed(2)}}%` }},
        {{ label: "Avg Full DD", render: (r) => `${{Number(r.avg_full_max_dd_pct).toFixed(2)}}%` }},
      ], rows);
    }}

    renderSummary();
    renderDataset();
    renderTopFull();
    renderMismatch();
    renderStrategySummary();
    ["search", "severity-filter", "lookahead-filter", "synthetic-filter"].forEach((id) => {{
      document.getElementById(id).addEventListener("input", renderStrategySummary);
      document.getElementById(id).addEventListener("change", renderStrategySummary);
    }});
  </script>
</body>
</html>
"""


def run(args: argparse.Namespace) -> None:
    global START_DATE
    START_DATE = pd.Timestamp(args.start_date, tz="UTC")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_paths = strategy_files(args.strategies)
    load_claims()

    results_rows: list[dict[str, Any]] = []
    completed = 0

    if args.workers <= 1:
        for path in strategy_paths:
            results_rows.extend(
                verify_strategy_path(
                    str(path),
                    args.symbols,
                    args.lookahead_symbol,
                    args.tail_check,
                    args.max_prefix_checks,
                )
            )
            completed += 1
            print(f"[{completed}/{len(strategy_paths)}] {path.stem}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    verify_strategy_path,
                    str(path),
                    args.symbols,
                    args.lookahead_symbol,
                    args.tail_check,
                    args.max_prefix_checks,
                ): path
                for path in strategy_paths
            }
            for future in concurrent.futures.as_completed(future_map):
                path = future_map[future]
                results_rows.extend(future.result())
                completed += 1
                print(f"[{completed}/{len(strategy_paths)}] {path.stem}", flush=True)

    results = pd.DataFrame(results_rows)

    results = results.sort_values(["strategy_name", "symbol", "period"]).reset_index(drop=True)
    strategy_summary = build_strategy_summary(results.copy())
    dataset_summary = build_dataset_summary(args.symbols)

    report_md = render_report(results, strategy_summary, dataset_summary)
    dashboard_html = render_dashboard(results, strategy_summary, dataset_summary)

    results.to_csv(output_dir / "verification_results.csv", index=False)
    results.to_json(output_dir / "verification_results.json", orient="records", indent=2)
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "strategies_requested": len(strategy_paths),
        "workers": args.workers,
        "symbols": args.symbols,
        "rows": int(len(results)),
        "errors": int((results.get("error", "") != "").sum()) if len(results) else 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run(parse_args())
