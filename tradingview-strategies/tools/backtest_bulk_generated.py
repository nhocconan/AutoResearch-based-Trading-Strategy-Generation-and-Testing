#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import BacktestConfig, BacktestResult, run_backtest
from evaluate import compute_metrics
from prepare import load_config, load_funding_rate, load_klines
from tv_backtest_settings import (
    FIXED_ORDER_FRACTION,
    FIXED_ORDER_SIZE_USD,
    INITIAL_CAPITAL_USD,
    POSITION_SIZING_LABEL,
    apply_fixed_order_size,
)


TV_ROOT = ROOT / "tradingview-strategies"
STRATEGY_DIR = TV_ROOT / "python-strategies" / "bulk"
OUT_PATH = TV_ROOT / "results" / "bulk-backtests.json"
ERRORS_PATH = TV_ROOT / "results" / "bulk-backtest-errors.json"
START_DATE = "2021-01-01"
LOAD_START = "2020-01-01"
LOAD_END = "2030-01-01"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def flush_outputs(results: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ERRORS_PATH.write_text(json.dumps(errors, indent=2), encoding="utf-8")


def main() -> None:
    config = load_config()
    bt_config = BacktestConfig.from_config(config)
    results = []
    errors = []
    strategy_paths = sorted(STRATEGY_DIR.glob("*.py"))
    total_paths = len(strategy_paths)

    for idx, path in enumerate(strategy_paths, start=1):
        try:
            mod = load_module(path)
            timeframe = getattr(mod, "timeframe")
            leverage = float(getattr(mod, "leverage", 1.0))
            strategy_name = getattr(mod, "name", path.stem)
        except Exception as exc:
            errors.append({
                "python_file": str(path.relative_to(TV_ROOT)),
                "symbol": None,
                "error": f"import failed: {exc}",
            })
            flush_outputs(results, errors)
            print(f"[{idx}/{total_paths}] import failed {path.name}: {exc}", flush=True)
            continue
        for symbol in ("BTCUSDT", "ETHUSDT"):
            try:
                print(f"[{idx}/{total_paths}] backtesting {path.name} {symbol} {timeframe}", flush=True)
                prices_full = load_klines(symbol, timeframe, LOAD_START, LOAD_END, config)
                signals_full = np.asarray(mod.generate_signals(prices_full), dtype=np.float64)
                mask = prices_full["open_time"] >= pd.Timestamp(START_DATE, tz="UTC")
                prices = prices_full.loc[mask].reset_index(drop=True)
                signals = apply_fixed_order_size(signals_full[mask.to_numpy()])
                funding = load_funding_rate(symbol, START_DATE, LOAD_END, config)
                equity, returns, trades = run_backtest(signals, prices, funding, bt_config, leverage=leverage)
                result = BacktestResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    period="2021-now",
                    equity_curve=equity,
                    returns=returns,
                    trades=trades,
                    backtest_duration_s=0.0,
                    num_bars=len(prices),
                )
                metrics = clean_metrics(compute_metrics(result))
                results.append({
                    "python_file": str(path.relative_to(TV_ROOT)),
                    "strategy_name": result.strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "position_sizing": POSITION_SIZING_LABEL,
                    "position_size_fraction": FIXED_ORDER_FRACTION,
                    "position_size_usd": FIXED_ORDER_SIZE_USD,
                    "initial_capital_usd": INITIAL_CAPITAL_USD,
                    "metrics": metrics,
                })
                flush_outputs(results, errors)
                print(
                    f"[{idx}/{total_paths}] ok {path.name} {symbol} trades={len(trades)} "
                    f"return={metrics.get('total_return_pct')}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "python_file": str(path.relative_to(TV_ROOT)),
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": str(exc),
                })
                flush_outputs(results, errors)
                print(f"[{idx}/{total_paths}] error {path.name} {symbol}: {exc}", flush=True)

    flush_outputs(results, errors)
    print(json.dumps({"results": len(results), "errors": len(errors), "output": str(OUT_PATH), "errors_output": str(ERRORS_PATH)}, indent=2))


if __name__ == "__main__":
    main()
