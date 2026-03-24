#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
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
STAGE2_SUMMARY_PATH = TV_ROOT / "results" / "stage2-local" / "summary.json"
STAGE3_RESULTS_PATH = TV_ROOT / "results" / "stage3-local-backtests.json"
STAGE3_ERRORS_PATH = TV_ROOT / "results" / "stage3-local-backtest-errors.json"
STAGE3_STATE_PATH = TV_ROOT / "results" / "stage3-local-backtest-state.json"
STAGE3_REPORTS_DIR = TV_ROOT / "reports" / "stage3-local"
LOG_PATH = TV_ROOT / "logs" / "stage3-local-backtests.log"

START_DATE = "2021-01-01"
LOAD_START = "2020-01-01"
LOAD_END = "2030-01-01"
SYMBOLS = ("BTCUSDT", "ETHUSDT")

TIMEFRAME_PRIORITY = {
    "1d": 0,
    "12h": 1,
    "6h": 2,
    "4h": 3,
    "1h": 4,
    "30m": 5,
    "15m": 6,
    "5m": 7,
    "1m": 8,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log_line(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)):
            out[key] = None if (np.isnan(value) or np.isinf(value)) else float(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def result_key(python_file: str, symbol: str) -> str:
    return f"{python_file}::{symbol}"


def uses_lookahead_blacklist(item: dict[str, Any]) -> bool:
    texts = [
        str(item.get("reason") or ""),
        str(item.get("last_error") or ""),
        " ".join(str(x) for x in (item.get("adaptations") or [])),
        " ".join(str(x) for x in (item.get("attempt_log") or [])),
    ]
    pine_file = item.get("pine_file")
    if pine_file:
        pine_path = TV_ROOT / pine_file
        if pine_path.exists():
            texts.append(pine_path.read_text(encoding="utf-8", errors="ignore"))
    lowered = "\n".join(texts).lower()
    patterns = (
        "lookahead_on",
        "lookahead = true",
        "lookahead=true",
        "security() with lookahead=true",
        "uses higher-timeframe lookahead_on",
    )
    return any(pattern in lowered for pattern in patterns)


def load_stage2_candidates() -> list[dict[str, Any]]:
    summary = load_json(STAGE2_SUMMARY_PATH, [])
    rows = []
    for item in summary:
        if item.get("status") != "converted":
            continue
        python_file = item.get("python_file")
        timeframe = item.get("timeframe")
        if not python_file or not timeframe:
            continue
        if not (TV_ROOT / python_file).exists():
            continue
        if uses_lookahead_blacklist(item):
            continue
        rows.append(item)
    rows.sort(
        key=lambda row: (
            TIMEFRAME_PRIORITY.get(str(row.get("timeframe") or ""), 99),
            int(row.get("queue_rank") or 10**9),
            row.get("slug") or "",
        )
    )
    return rows


def run_one(module, candidate: dict[str, Any], symbol: str, config: dict[str, Any], bt_config: BacktestConfig) -> dict[str, Any]:
    timeframe = getattr(module, "timeframe")
    leverage = float(getattr(module, "leverage", 1.0))
    strategy_name = getattr(module, "name", candidate.get("slug") or Path(candidate["python_file"]).stem)

    prices_full = load_klines(symbol, timeframe, LOAD_START, LOAD_END, config)
    signals_full = np.asarray(module.generate_signals(prices_full), dtype=np.float64)
    if len(signals_full) != len(prices_full):
        raise RuntimeError(f"Signal length mismatch: {len(signals_full)} != {len(prices_full)}")

    mask = prices_full["open_time"] >= pd.Timestamp(START_DATE, tz="UTC")
    prices = prices_full.loc[mask].reset_index(drop=True)
    signals = apply_fixed_order_size(signals_full[mask.to_numpy()])
    funding_df = load_funding_rate(symbol, START_DATE, LOAD_END, config)

    equity, returns, trades = run_backtest(signals, prices, funding_df, bt_config, leverage=leverage)
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
    funding_end = funding_df["calc_time"].max().isoformat() if len(funding_df) else None
    data_end = prices["open_time"].max().isoformat() if len(prices) else None

    return {
        "chart_url": candidate.get("chart_url"),
        "slug": candidate.get("slug"),
        "queue_rank": candidate.get("queue_rank"),
        "batch": "stage2-local",
        "classification": candidate.get("classification"),
        "python_file": candidate["python_file"],
        "strategy_name": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "position_sizing": POSITION_SIZING_LABEL,
        "position_size_fraction": FIXED_ORDER_FRACTION,
        "position_size_usd": FIXED_ORDER_SIZE_USD,
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "data_start": START_DATE,
        "data_end": data_end,
        "funding_end": funding_end,
        "metrics": metrics,
        "backtested_at": now_iso(),
    }


def write_report(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    report_path = STAGE3_REPORTS_DIR / f"{candidate['slug']}.md"
    lines = [
        f"# {rows[0]['strategy_name']}",
        "",
        f"- Source URL: {candidate.get('chart_url')}",
        f"- Python file: `{candidate.get('python_file')}`",
        f"- Classification: `{candidate.get('classification')}`",
        f"- Position sizing: `{POSITION_SIZING_LABEL}` (`${FIXED_ORDER_SIZE_USD:,.0f}` per trade on `${INITIAL_CAPITAL_USD:,.0f}` initial capital)",
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
                data_end=(row["data_end"] or "n/a")[:19],
                funding_end=(row["funding_end"] or "n/a")[:19],
                ret=metrics.get("total_return_pct") or 0.0,
                sharpe=metrics.get("sharpe_ratio") or 0.0,
                dd=metrics.get("max_drawdown_pct") or 0.0,
                trades=metrics.get("num_trades") or 0,
                win=metrics.get("win_rate") or 0.0,
                pf=metrics.get("profit_factor") or 0.0,
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Stage 2 local converted strategies and persist resumable state.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of strategy files to process this run. 0 means no limit.")
    parser.add_argument("--force", action="store_true", help="Re-backtest even if an item is already recorded.")
    args = parser.parse_args()

    config = load_config()
    bt_config = BacktestConfig.from_config(config)
    candidates = load_stage2_candidates()

    existing_results = load_json(STAGE3_RESULTS_PATH, [])
    existing_errors = load_json(STAGE3_ERRORS_PATH, [])
    results_by_key = {result_key(item["python_file"], item["symbol"]): item for item in existing_results}
    errors_by_key = {result_key(item["python_file"], item["symbol"]): item for item in existing_errors}

    processed_files = 0
    for candidate in candidates:
        if args.limit and processed_files >= args.limit:
            break
        python_path = TV_ROOT / candidate["python_file"]
        if not python_path.exists():
            continue
        module = None
        report_rows = []
        should_run_file = args.force
        if not should_run_file:
            for symbol in SYMBOLS:
                if result_key(candidate["python_file"], symbol) not in results_by_key:
                    should_run_file = True
                    break
        if not should_run_file:
            continue

        save_json(
            STAGE3_STATE_PATH,
            {
                "generated_at": now_iso(),
                "converted_candidates": len(candidates),
                "strategy_files_backtested": len({row["python_file"] for row in results_by_key.values()}),
                "result_rows": len(results_by_key),
                "error_rows": len(errors_by_key),
                "symbols_per_strategy": list(SYMBOLS),
                "status": "running",
                "current_slug": candidate.get("slug"),
                "current_python_file": candidate.get("python_file"),
                "processed_files_this_run": processed_files,
            },
        )
        log_line(f"backtesting {candidate['slug']} file={candidate['python_file']}")
        processed_files += 1
        try:
            module = load_module(python_path)
        except Exception as exc:
            for symbol in SYMBOLS:
                key = result_key(candidate["python_file"], symbol)
                errors_by_key[key] = {
                    "chart_url": candidate.get("chart_url"),
                    "slug": candidate.get("slug"),
                    "python_file": candidate["python_file"],
                    "symbol": symbol,
                    "timeframe": candidate.get("timeframe"),
                    "error": f"import failed: {exc}",
                    "backtested_at": now_iso(),
                }
            save_json(STAGE3_RESULTS_PATH, sorted(results_by_key.values(), key=lambda row: (row.get("queue_rank") or 10**9, row.get("python_file"), row.get("symbol"))))
            save_json(STAGE3_ERRORS_PATH, sorted(errors_by_key.values(), key=lambda row: (row.get("slug") or "", row.get("symbol") or "")))
            continue

        for symbol in SYMBOLS:
            key = result_key(candidate["python_file"], symbol)
            if not args.force and key in results_by_key:
                report_rows.append(results_by_key[key])
                continue
            try:
                row = run_one(module, candidate, symbol, config, bt_config)
                results_by_key[key] = row
                errors_by_key.pop(key, None)
                report_rows.append(row)
                log_line(
                    f"ok slug={candidate['slug']} symbol={symbol} trades={row['metrics'].get('num_trades')} "
                    f"return={row['metrics'].get('total_return_pct')}"
                )
            except Exception as exc:
                errors_by_key[key] = {
                    "chart_url": candidate.get("chart_url"),
                    "slug": candidate.get("slug"),
                    "python_file": candidate["python_file"],
                    "symbol": symbol,
                    "timeframe": candidate.get("timeframe"),
                    "error": str(exc),
                    "backtested_at": now_iso(),
                }
                log_line(f"error slug={candidate['slug']} symbol={symbol} error={exc}")

        if report_rows:
            write_report(candidate, sorted(report_rows, key=lambda row: row["symbol"]))

        merged_results = sorted(results_by_key.values(), key=lambda row: (row.get("queue_rank") or 10**9, row.get("python_file"), row.get("symbol")))
        merged_errors = sorted(errors_by_key.values(), key=lambda row: (row.get("slug") or "", row.get("symbol") or ""))
        save_json(STAGE3_RESULTS_PATH, merged_results)
        save_json(STAGE3_ERRORS_PATH, merged_errors)
        save_json(
            STAGE3_STATE_PATH,
            {
                "generated_at": now_iso(),
                "converted_candidates": len(candidates),
                "strategy_files_backtested": len({row["python_file"] for row in merged_results}),
                "result_rows": len(merged_results),
                "error_rows": len(merged_errors),
                "symbols_per_strategy": list(SYMBOLS),
                "status": "running",
                "last_processed_slug": candidate.get("slug"),
            },
        )

    final_results = load_json(STAGE3_RESULTS_PATH, [])
    final_errors = load_json(STAGE3_ERRORS_PATH, [])
    print(
        json.dumps(
            {
                "converted_candidates": len(candidates),
                "strategy_files_backtested": len({row["python_file"] for row in final_results}),
                "result_rows": len(final_results),
                "error_rows": len(final_errors),
                "state_path": str(STAGE3_STATE_PATH),
            },
            indent=2,
        )
    )
    save_json(
        STAGE3_STATE_PATH,
        {
            "generated_at": now_iso(),
            "converted_candidates": len(candidates),
            "strategy_files_backtested": len({row["python_file"] for row in final_results}),
            "result_rows": len(final_results),
            "error_rows": len(final_errors),
            "symbols_per_strategy": list(SYMBOLS),
            "status": "idle",
            "last_processed_slug": final_results[-1]["slug"] if final_results else "",
        },
    )


if __name__ == "__main__":
    main()
