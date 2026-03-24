#!/usr/bin/env python3
"""
Run converted TradingView strategies against repo parquet data.

This script is isolated to tradingview-strategies/ and reuses the repo's
immutable loaders/backtest/evaluation modules without modifying them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
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
STRATEGY_DIR = TV_ROOT / "python-strategies"
RESULTS_DIR = TV_ROOT / "results"
REPORTS_DIR = TV_ROOT / "reports"

START_DATE = "2021-01-01"
LOAD_START_DATE = "2020-01-01"
LOAD_END_DATE = "2030-01-01"


@dataclass(frozen=True)
class StrategySpec:
    slug: str
    strategy_path: Path
    source_url: str
    pine_file: str
    compatibility: str
    notes: list[str]


SUPPORTED_STRATEGIES = [
    StrategySpec(
        slug="kinetic_kalman_breakout",
        strategy_path=STRATEGY_DIR / "kinetic_kalman_breakout.py",
        source_url="https://www.tradingview.com/script/nd8EpyQ5-Kinetic-Kalman-Breakout/",
        pine_file="nd8EpyQ5-Kinetic-Kalman-Breakout.pine",
        compatibility="direct",
        notes=[
            "Two-state Kalman filter and MAE bands were translated directly.",
            "Signal model preserves always-in-market flips between long and short.",
        ],
    ),
    StrategySpec(
        slug="macd_trend_enhancement",
        strategy_path=STRATEGY_DIR / "macd_trend_enhancement.py",
        source_url="https://www.tradingview.com/script/2o1g6Qo5/",
        pine_file="2o1g6Qo5.pine",
        compatibility="partial",
        notes=[
            "Broker-managed stop and trailing behavior were approximated as bar-triggered exit signals.",
            "All fills still occur at next bar open under the repo backtester.",
        ],
    ),
    StrategySpec(
        slug="quant_trend_engine_long_only_v3",
        strategy_path=STRATEGY_DIR / "quant_trend_engine_long_only_v3.py",
        source_url="https://www.tradingview.com/script/bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe/",
        pine_file="bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe.pine",
        compatibility="partial",
        notes=[
            "Friday close and stop behavior were adapted to next-bar execution.",
            "Signal logic, scoring, cooldown, and session gating were preserved.",
        ],
    ),
]

UNSUPPORTED_STRATEGIES = [
    {
        "slug": "btc_re_entry_alpha_1h",
        "source_url": "https://www.tradingview.com/script/w45uet8E/",
        "pine_file": "w45uet8E.pine",
        "compatibility": "unsupported",
        "reason": "Uses request.security(..., lookahead=barmerge.lookahead_on), which leaks higher-timeframe future information into lower-timeframe bars.",
    }
]


def load_strategy(strategy_path: Path):
    spec = importlib.util.spec_from_file_location(strategy_path.stem, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialize_metric(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run_single_backtest(strategy_module, symbol: str, config: dict) -> dict[str, Any]:
    timeframe = getattr(strategy_module, "timeframe")
    leverage = float(getattr(strategy_module, "leverage", 1.0))
    bt_config = BacktestConfig.from_config(config)

    prices_full = load_klines(symbol, timeframe, LOAD_START_DATE, LOAD_END_DATE, config)
    if len(prices_full) == 0:
        raise ValueError(f"No data available for {symbol} {timeframe}")

    signals_full = strategy_module.generate_signals(prices_full)
    if len(signals_full) != len(prices_full):
        raise ValueError(
            f"Signal length mismatch for {strategy_module.name}: "
            f"{len(signals_full)} vs {len(prices_full)}"
        )

    trimmed_mask = prices_full["open_time"] >= pd.Timestamp(START_DATE, tz="UTC")
    prices = prices_full.loc[trimmed_mask].reset_index(drop=True)
    signals = np.asarray(signals_full[trimmed_mask.to_numpy()], dtype=np.float64)
    signals = apply_fixed_order_size(signals)

    if len(prices) == 0:
        raise ValueError(f"No trimmed data available for {symbol} {timeframe}")

    data_end = prices["open_time"].max()
    funding_df = None
    funding_end = None
    if bt_config.include_funding:
        funding_df = load_funding_rate(symbol, START_DATE, LOAD_END_DATE, config)
        if len(funding_df) > 0:
            funding_end = funding_df["calc_time"].max()

    equity, returns, trades = run_backtest(
        signals=signals,
        prices=prices,
        funding_df=funding_df,
        bt_config=bt_config,
        leverage=leverage,
    )
    result = BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_module.name,
        period="2021-now",
        equity_curve=equity,
        returns=returns,
        trades=trades,
        backtest_duration_s=0.0,
        num_bars=len(prices),
    )
    metrics = compute_metrics(result)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_name": strategy_module.name,
        "leverage": leverage,
        "position_sizing": POSITION_SIZING_LABEL,
        "position_size_fraction": FIXED_ORDER_FRACTION,
        "position_size_usd": FIXED_ORDER_SIZE_USD,
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "start_date": START_DATE,
        "data_end": data_end.isoformat(),
        "funding_end": funding_end.isoformat() if funding_end is not None else None,
        "num_signals_nonzero": int(np.count_nonzero(signals)),
        "metrics": {k: serialize_metric(v) for k, v in metrics.items()},
    }


def write_strategy_report(spec: StrategySpec, rows: list[dict[str, Any]]) -> None:
    report_path = REPORTS_DIR / f"{spec.slug}.md"
    lines = [
        f"# {rows[0]['strategy_name']}",
        "",
        f"- Source URL: {spec.source_url}",
        f"- Pine file: `raw-pine/{spec.pine_file}`",
        f"- Python file: `python-strategies/{spec.strategy_path.name}`",
        f"- Compatibility: `{spec.compatibility}`",
        f"- Position sizing: `{POSITION_SIZING_LABEL}` (`${FIXED_ORDER_SIZE_USD:,.0f}` per trade on `${INITIAL_CAPITAL_USD:,.0f}` initial capital)",
        "",
        "## Adaptation Notes",
        "",
    ]
    for note in spec.notes:
        lines.append(f"- {note}")
    lines += [
        "",
        "## Backtest Results",
        "",
        "| Symbol | Timeframe | Data End | Funding End | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {symbol} | {timeframe} | {data_end} | {funding_end} | {ret:.2f} | {sharpe:.3f} | {dd:.2f} | {trades} | {win:.1f} | {pf:.2f} |".format(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                data_end=row["data_end"][:19],
                funding_end=(row["funding_end"][:19] if row["funding_end"] else "n/a"),
                ret=metrics["total_return_pct"] or 0.0,
                sharpe=metrics["sharpe_ratio"] or 0.0,
                dd=metrics["max_drawdown_pct"] or 0.0,
                trades=metrics["num_trades"] or 0,
                win=metrics["win_rate"] or 0.0,
                pf=metrics["profit_factor"] or 0.0,
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unsupported_report(item: dict[str, Any]) -> None:
    report_path = REPORTS_DIR / f"{item['slug']}.md"
    lines = [
        "# Unsupported Conversion",
        "",
        f"- Source URL: {item['source_url']}",
        f"- Pine file: `raw-pine/{item['pine_file']}`",
        f"- Compatibility: `{item['compatibility']}`",
        "",
        "## Reason",
        "",
        f"- {item['reason']}",
        "",
        "## Decision",
        "",
        "- No Python strategy was generated because that would require a dishonest lookahead-dependent translation.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(results: list[dict[str, Any]]) -> None:
    summary_path = REPORTS_DIR / "summary.md"
    lines = [
        "# TradingView Strategy Conversion Summary",
        "",
        f"- Tested period: `{START_DATE}` through latest locally available bars",
        "- Symbols: `BTCUSDT`, `ETHUSDT`",
        f"- Position sizing: `{POSITION_SIZING_LABEL}` (`${FIXED_ORDER_SIZE_USD:,.0f}` fixed order size on `${INITIAL_CAPITAL_USD:,.0f}` initial capital)",
        "- Funding data is included where present in repo parquet data.",
        "",
        "## Converted Strategies",
        "",
        "| Strategy | Compatibility | Symbol | Return % | Sharpe | Max DD % | Trades |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            "| {strategy} | {compat} | {symbol} | {ret:.2f} | {sharpe:.3f} | {dd:.2f} | {trades} |".format(
                strategy=result["strategy_name"],
                compat=result["compatibility"],
                symbol=result["symbol"],
                ret=metrics["total_return_pct"] or 0.0,
                sharpe=metrics["sharpe_ratio"] or 0.0,
                dd=metrics["max_drawdown_pct"] or 0.0,
                trades=metrics["num_trades"] or 0,
            )
        )
    lines += [
        "",
        "## Unsupported",
        "",
    ]
    for item in UNSUPPORTED_STRATEGIES:
        lines.append(f"- `{item['slug']}`: {item['reason']}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    all_results: list[dict[str, Any]] = []

    for spec in SUPPORTED_STRATEGIES:
        strategy_module = load_strategy(spec.strategy_path)
        strategy_rows = []
        for symbol in ("BTCUSDT", "ETHUSDT"):
            row = run_single_backtest(strategy_module, symbol, config)
            row["source_url"] = spec.source_url
            row["pine_file"] = spec.pine_file
            row["compatibility"] = spec.compatibility
            strategy_rows.append(row)
            all_results.append(row)

        write_strategy_report(spec, strategy_rows)

    for item in UNSUPPORTED_STRATEGIES:
        write_unsupported_report(item)

    results_payload = {
        "supported": all_results,
        "unsupported": UNSUPPORTED_STRATEGIES,
    }
    (RESULTS_DIR / "backtest_results.json").write_text(
        json.dumps(results_payload, indent=2),
        encoding="utf-8",
    )
    write_summary(all_results)


if __name__ == "__main__":
    main()
