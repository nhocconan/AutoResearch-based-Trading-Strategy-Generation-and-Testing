#!/usr/bin/env python3
"""
Dashboard for TradingView strategy testing outputs.

Usage:
    ./.venv/bin/python tradingview-strategies/tools/dashboard_tv.py
    ./.venv/bin/python tradingview-strategies/tools/dashboard_tv.py --port 8890
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
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
RESULTS_DIR = TV_ROOT / "results"
REPORTS_DIR = TV_ROOT / "reports"
PY_DIR = TV_ROOT / "python-strategies"
STATE_PATH = RESULTS_DIR / "continuous-pipeline-state.json"
STAGE_PROGRESS_PATH = RESULTS_DIR / "stage-progress.json"
STAGE1_RUNNER_PATH = RESULTS_DIR / "stage1-runner.json"
STAGE2_LOCAL_SUMMARY_PATH = RESULTS_DIR / "stage2-local" / "summary.json"
STAGE2_LOCAL_PROGRESS_PATH = RESULTS_DIR / "stage2-local" / "progress.json"
STAGE2_LOCAL_ERRORS_PATH = RESULTS_DIR / "stage2-local" / "errors.json"
STAGE2_LOCAL_RUNNER_PATH = RESULTS_DIR / "stage2-local-runner.json"
STAGE3_LOCAL_RESULTS_PATH = RESULTS_DIR / "stage3-local-backtests.json"
STAGE3_LOCAL_ERRORS_PATH = RESULTS_DIR / "stage3-local-backtest-errors.json"
STAGE3_LOCAL_STATE_PATH = RESULTS_DIR / "stage3-local-backtest-state.json"
STAGE3_REPORTS_DIR = REPORTS_DIR / "stage3-local"

START_DATE = "2021-01-01"
LOAD_START = "2020-01-01"
LOAD_END = "2030-01-01"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialize_metric(value):
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def timeframe_to_tv_interval(timeframe: str) -> str:
    mapping = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "1D",
        "1w": "1W",
    }
    return mapping.get(timeframe, timeframe)


def build_tv_chart_url(symbol: str, timeframe: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe_to_tv_interval(timeframe)}"


def load_coverage_summary() -> dict:
    summary = load_json(STATE_PATH, {}).get("summary", {})
    stage_progress = load_json(STAGE_PROGRESS_PATH, {})
    stage1_runner = load_json(STAGE1_RUNNER_PATH, {})
    stage2_local_summary = load_json(STAGE2_LOCAL_SUMMARY_PATH, [])
    stage2_local_progress = load_json(STAGE2_LOCAL_PROGRESS_PATH, {})
    stage2_local_runner = load_json(STAGE2_LOCAL_RUNNER_PATH, {})
    stage2_local_errors = load_json(STAGE2_LOCAL_ERRORS_PATH, [])
    stage3_local_state = load_json(STAGE3_LOCAL_STATE_PATH, {})
    stage1 = stage_progress.get("stage_1_pine_cache", {}) if isinstance(stage_progress, dict) else {}
    stage2 = stage_progress.get("stage_2_conversion", {}) if isinstance(stage_progress, dict) else {}
    stage3 = stage_progress.get("stage_3_backtests", {}) if isinstance(stage_progress, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    stage2_runner_agg = stage2_local_runner.get("aggregate") if isinstance(stage2_local_runner, dict) else {}
    if not isinstance(stage2_runner_agg, dict):
        stage2_runner_agg = {}
    live_stage2_reports = int(stage2_runner_agg.get("processed_reports") or stage2_local_progress.get("processed_reports") or 0)
    live_stage2_terminal = int(stage2_runner_agg.get("terminal_reports") or stage2_local_progress.get("terminal_reports") or 0)
    live_stage2_converted = int(stage2_runner_agg.get("converted") or stage2_local_progress.get("converted") or 0)
    live_stage2_unsupported = int(stage2_runner_agg.get("unsupported") or stage2_local_progress.get("unsupported") or 0)
    live_stage2_errors = int(stage2_runner_agg.get("errors") or stage2_local_progress.get("errors") or 0)
    live_stage2_retryable = int(stage2_runner_agg.get("retryable") or stage2_local_progress.get("retryable") or 0)
    live_stage2_queue = int(stage2_runner_agg.get("queue_candidates_available") or stage2_local_progress.get("queue_candidates_available") or 0)
    live_stage2_remaining = int(stage2_runner_agg.get("remaining_candidates") or 0)
    if not live_stage2_remaining and live_stage2_queue:
        live_stage2_remaining = max(live_stage2_queue - live_stage2_terminal, 0)
    live_stage2_active_lanes = int(stage2_runner_agg.get("active_lanes") or stage2_local_progress.get("active_lanes") or 0)
    if isinstance(stage2_local_errors, list):
        live_stage2_recorded_errors = len(stage2_local_errors)
    else:
        live_stage2_recorded_errors = 0
    if isinstance(stage2_local_summary, list):
        stage2_summary_reports = len(stage2_local_summary)
        stage2_summary_converted = sum(1 for item in stage2_local_summary if item.get("status") == "converted")
        stage2_summary_unsupported = sum(1 for item in stage2_local_summary if item.get("status") == "unsupported")
        stage2_summary_errors = sum(1 for item in stage2_local_summary if item.get("status") == "error")
        stage2_summary_retryable = sum(1 for item in stage2_local_summary if item.get("status") == "retryable")
    else:
        stage2_summary_reports = 0
        stage2_summary_converted = 0
        stage2_summary_unsupported = 0
        stage2_summary_errors = 0
        stage2_summary_retryable = 0
    summary.update(
        {
            "stage1_total_items": int(stage1.get("total_items") or 0),
            "stage1_cached_ok": int(stage1.get("cached_ok") or 0),
            "stage1_pending": int(stage1.get("pending") or 0),
            "stage1_errors": int(stage1.get("errors") or 0),
            "stage2_reports": stage2_summary_reports or live_stage2_reports or int(stage2.get("reports") or 0),
            "stage2_unsupported": stage2_summary_unsupported or live_stage2_unsupported or int(stage2.get("unsupported") or 0),
            "stage2_converted_import_ok": stage2_summary_converted or live_stage2_converted or int(stage2.get("converted_import_ok") or 0),
            "stage2_convert_errors": stage2_summary_errors or live_stage2_errors or int(stage2.get("convert_errors") or 0),
            "stage2_retryable": stage2_summary_retryable or live_stage2_retryable,
            "stage2_queue_candidates": live_stage2_queue,
            "stage2_remaining_candidates": live_stage2_remaining,
            "stage2_recorded_error_items": live_stage2_recorded_errors,
            "stage2_active_lanes": live_stage2_active_lanes,
            "stage2_runner_status": stage2_local_runner.get("status") or "unknown",
            "stage2_runner_updated_at": stage2_local_runner.get("updated_at") or stage2_local_progress.get("generated_at") or "",
            "stage2_last_processed_slug": stage2_runner_agg.get("last_processed_slug") or stage2_local_progress.get("last_processed_slug") or "",
            "stage2_model": stage2_runner_agg.get("model") or stage2_local_progress.get("model") or "",
            "stage3_result_rows": int(stage3_local_state.get("result_rows") or stage3.get("bulk_result_rows") or 0),
            "stage3_strategy_files": int(stage3_local_state.get("strategy_files_backtested") or stage3.get("bulk_strategy_files") or 0),
            "stage3_error_rows": int(stage3_local_state.get("error_rows") or 0),
            "stage1_runner_status": stage1_runner.get("status") or "unknown",
            "stage1_retryable_errors": int(
                ((stage1_runner.get("counts_before_batch") or {}).get("retryable_errors")) or 0
            ),
            "stage1_runner_updated_at": stage1_runner.get("updated_at") or "",
        }
    )
    return summary


def load_unsupported_rows() -> list[dict]:
    path = STAGE2_LOCAL_SUMMARY_PATH
    obj = load_json(path, [])
    if not isinstance(obj, list):
        return []
    rows = []
    for item in obj:
        if item.get("status") != "unsupported":
            continue
        slug = item.get("slug") or ""
        report_path = REPORTS_DIR / "stage2-local" / f"{slug}.md"
        rows.append(
            {
                "name": slug,
                "compatibility": item.get("classification") or "unsupported",
                "reason": item.get("last_error") or "unsupported by conversion skill",
                "source_url": item.get("chart_url") or "",
                "report_file": str(report_path.relative_to(TV_ROOT)) if report_path.exists() else "",
            }
        )
    return rows


def _uses_lookahead_text(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "lookahead_on",
        "lookahead = true",
        "lookahead=true",
        "security() with lookahead=true",
        "uses higher-timeframe lookahead_on",
    )
    return any(pattern in lowered for pattern in patterns)


def load_lookahead_blacklist_rows() -> list[dict]:
    summary_rows = load_json(STAGE2_LOCAL_SUMMARY_PATH, [])
    error_rows = load_json(STAGE2_LOCAL_ERRORS_PATH, [])
    canonical_item_rows = []
    for path in (RESULTS_DIR / "stage2-local").glob("*.json"):
        if path.name in {"summary.json", "progress.json", "errors.json"}:
            continue
        payload = load_json(path, {})
        if isinstance(payload, dict):
            canonical_item_rows.append(payload)
    rows_by_slug: dict[str, dict] = {}

    def maybe_add(item: dict, source: str) -> None:
        if not isinstance(item, dict):
            return
        slug = item.get("slug") or Path(str(item.get("python_file") or "")).stem
        if not slug:
            return
        pine_rel = item.get("pine_file") or ""
        pine_text = read_text(TV_ROOT / pine_rel) if pine_rel else ""
        text_parts = [
            str(item.get("reason") or ""),
            str(item.get("last_error") or ""),
            " ".join(str(x) for x in (item.get("adaptations") or [])),
            " ".join(str(x) for x in (item.get("attempt_log") or [])),
            pine_text,
        ]
        combined = "\n".join(text_parts)
        if not _uses_lookahead_text(combined):
            return
        entry = {
            "queue_rank": item.get("queue_rank"),
            "slug": slug,
            "name": item.get("name") or slug,
            "status": item.get("status") or "",
            "classification": item.get("classification") or "",
            "timeframe": item.get("timeframe") or "",
            "source_url": item.get("chart_url") or "",
            "pine_file": pine_rel,
            "python_file": item.get("python_file") or "",
            "reason": item.get("reason") or item.get("last_error") or "",
            "detection_source": source,
        }
        existing = rows_by_slug.get(slug)
        if existing is None:
            rows_by_slug[slug] = entry
            return
        existing_status = str(existing.get("status") or "")
        new_status = str(entry.get("status") or "")
        status_rank = {"converted": 3, "unsupported": 2, "error": 1}
        if status_rank.get(new_status, 0) >= status_rank.get(existing_status, 0):
            rows_by_slug[slug] = entry

    if isinstance(summary_rows, list):
        for item in summary_rows:
            maybe_add(item, "stage2-summary")
    if isinstance(error_rows, list):
        for item in error_rows:
            maybe_add(item, "stage2-errors")
    for item in canonical_item_rows:
        maybe_add(item, "stage2-canonical")

    rows = list(rows_by_slug.values())
    rows.sort(key=lambda row: (int(row.get("queue_rank") or 10**9), row.get("slug") or ""))
    return rows


def load_rows() -> list[dict]:
    obj = load_json(STAGE3_LOCAL_RESULTS_PATH, [])
    if not isinstance(obj, list):
        return []
    rows = []
    for item in obj:
        metrics = item.get("metrics") or {}
        py_rel = item.get("python_file") or ""
        py_path = TV_ROOT / py_rel if py_rel else None
        slug = item.get("slug") or Path(py_rel).stem
        report_path = STAGE3_REPORTS_DIR / f"{slug}.md"
        rows.append(
            {
                "batch": item.get("batch") or "stage2-local",
                "strategy": item.get("strategy_name") or slug,
                "symbol": item.get("symbol") or "",
                "timeframe": item.get("timeframe") or "",
                "compatibility": item.get("classification") or "",
                "source_url": item.get("chart_url") or "",
                "python_file": py_rel,
                "report_file": str(report_path.relative_to(TV_ROOT)) if report_path.exists() else "",
                "total_return_pct": float(metrics.get("total_return_pct") or 0.0),
                "sharpe_ratio": float(metrics.get("sharpe_ratio") or 0.0),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
                "num_trades": int(metrics.get("num_trades") or 0),
                "win_rate": float(metrics.get("win_rate") or 0.0),
                "profit_factor": float(metrics.get("profit_factor") or 0.0),
                "tv_chart_url": build_tv_chart_url(item.get("symbol") or "", item.get("timeframe") or ""),
                "code": read_text(py_path),
                "report": read_text(report_path),
            }
        )
    rows.sort(key=lambda row: (row["sharpe_ratio"], row["total_return_pct"]), reverse=True)
    return rows


def build_strategy_payload(rows: list[dict]) -> str:
    grouped: dict[str, dict] = {}
    for row in rows:
        name = row["strategy"]
        grouped.setdefault(
            name,
            {
                "strategy": name,
                "batch": row["batch"],
                "compatibility": row["compatibility"],
                "source_url": row["source_url"],
                "python_file": row["python_file"],
                "report_file": row["report_file"],
                "code": row["code"],
                "report": row["report"],
                "rows": [],
            },
        )
        grouped[name]["rows"].append(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "return_pct": row["total_return_pct"],
                "sharpe_ratio": row["sharpe_ratio"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "num_trades": row["num_trades"],
                "win_rate": row["win_rate"],
                "profit_factor": row["profit_factor"],
                "tv_chart_url": row["tv_chart_url"],
            }
        )
    return json.dumps(grouped, ensure_ascii=False)


def find_strategy_row(strategy_name: str, symbol: str) -> dict | None:
    for row in load_rows():
        if row["strategy"] == strategy_name and row["symbol"] == symbol:
            return row
    return None


def run_detail_backtest(strategy_name: str, symbol: str) -> dict:
    row = find_strategy_row(strategy_name, symbol)
    if row is None:
        return {"error": f"Strategy result not found for {strategy_name} / {symbol}"}
    python_file = row.get("python_file")
    if not python_file:
        return {"error": f"Python file not found for {strategy_name}"}
    strategy_path = TV_ROOT / python_file
    if not strategy_path.exists():
        return {"error": f"Strategy path missing: {python_file}"}

    try:
        module = load_module(strategy_path)
        timeframe = getattr(module, "timeframe", row["timeframe"])
        leverage = float(getattr(module, "leverage", 1.0))
        config = load_config()
        bt_config = BacktestConfig.from_config(config)

        prices_full = load_klines(symbol, timeframe, LOAD_START, LOAD_END, config)
        if len(prices_full) == 0:
            return {"error": f"No prices for {symbol} {timeframe}"}
        signals_full = np.asarray(module.generate_signals(prices_full), dtype=np.float64)
        if len(signals_full) != len(prices_full):
            return {"error": f"Signal length mismatch: {len(signals_full)} vs {len(prices_full)}"}

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
        metrics = compute_metrics(result)

        n = len(equity)
        step = max(1, n // 500)
        eq_sampled = [round(float(v), 2) for v in equity[::step]]
        eq_labels = [f"{i * step}" for i in range(len(eq_sampled))]

        unix_times = [int(pd.Timestamp(t).timestamp()) for t in prices["open_time"].values]
        price_n = len(prices)
        price_step = max(1, price_n // 3000)
        ohlc = []
        for i in range(0, price_n, price_step):
            end = min(i + price_step, price_n)
            ohlc.append(
                {
                    "time": int(unix_times[i]),
                    "open": round(float(prices["open"].values[i]), 4),
                    "high": round(float(prices["high"].values[i:end].max()), 4),
                    "low": round(float(prices["low"].values[i:end].min()), 4),
                    "close": round(float(prices["close"].values[end - 1]), 4),
                }
            )

        close_s = prices["close"]
        ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().to_numpy(dtype=np.float64)
        ema55 = close_s.ewm(span=55, min_periods=55, adjust=False).mean().to_numpy(dtype=np.float64)
        indicators = {}
        for name, series, color in (
            ("EMA 21", ema21, "#f0883e"),
            ("EMA 55", ema55, "#a371f7"),
        ):
            ind_data = []
            for i in range(0, price_n, price_step):
                value = float(series[i])
                if not np.isnan(value):
                    ind_data.append({"time": int(unix_times[i]), "value": round(value, 4)})
            indicators[name] = {"data": ind_data, "color": color}

        signal_data = []
        for i in range(0, price_n, price_step):
            signal_data.append(
                {
                    "time": int(unix_times[i]),
                    "value": round(float(signals[i]), 4),
                }
            )

        trade_markers = []
        trades_payload = []
        for trade in result.trades:
            entry_ts = int(pd.Timestamp(trade.entry_time).timestamp())
            exit_ts = int(pd.Timestamp(trade.exit_time).timestamp())
            direction = "LONG" if trade.direction == 1 else "SHORT"
            pnl_pct = round(float(trade.pnl_pct) * 100, 3)
            trade_markers.append(
                {
                    "entry_time": entry_ts,
                    "exit_time": exit_ts,
                    "direction": direction,
                    "entry_price": round(float(trade.entry_price), 4),
                    "exit_price": round(float(trade.exit_price), 4),
                    "pnl_pct": pnl_pct,
                }
            )
            trades_payload.append(
                {
                    "entry_time": str(trade.entry_time),
                    "exit_time": str(trade.exit_time),
                    "direction": direction,
                    "entry_price": round(float(trade.entry_price), 4),
                    "exit_price": round(float(trade.exit_price), 4),
                    "size": round(float(trade.size), 4),
                    "leverage": round(float(trade.leverage), 2),
                    "pnl": round(float(trade.pnl), 2),
                    "pnl_pct": pnl_pct,
                    "fee_cost": round(float(trade.fee_cost), 2),
                    "funding_cost": round(float(trade.funding_cost), 2),
                }
            )

        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "tv_chart_url": build_tv_chart_url(symbol, timeframe),
            "position_sizing": POSITION_SIZING_LABEL,
            "position_size_fraction": FIXED_ORDER_FRACTION,
            "position_size_usd": FIXED_ORDER_SIZE_USD,
            "initial_capital_usd": INITIAL_CAPITAL_USD,
            "metrics": {
                "sharpe": round(float(serialize_metric(metrics["sharpe_ratio"]) or 0.0), 4),
                "return_pct": round(float(serialize_metric(metrics["total_return_pct"]) or 0.0), 2),
                "max_dd_pct": round(float(serialize_metric(metrics["max_drawdown_pct"]) or 0.0), 2),
                "win_rate": round(float(serialize_metric(metrics["win_rate"]) or 0.0), 1),
                "num_trades": int(serialize_metric(metrics["num_trades"]) or 0),
                "profit_factor": round(float(serialize_metric(metrics["profit_factor"]) or 0.0), 2),
                "total_fees": round(float(serialize_metric(metrics["total_fees"]) or 0.0), 2),
                "total_funding": round(float(serialize_metric(metrics["total_funding_cost"]) or 0.0), 2),
            },
            "equity_curve": eq_sampled,
            "equity_labels": eq_labels,
            "trades": trades_payload,
            "num_bars": n,
            "ohlc": ohlc,
            "indicators": indicators,
            "signals": signal_data,
            "trade_markers": trade_markers,
        }
    except Exception as exc:
        return {"error": str(exc)}


def render_html() -> str:
    rows = load_rows()
    unsupported = load_unsupported_rows()
    lookahead_blacklist = load_lookahead_blacklist_rows()
    df = pd.DataFrame(rows)
    coverage = load_coverage_summary()

    total_rows = len(df)
    total_strategies = df["strategy"].nunique() if total_rows else 0
    best_sharpe = float(df["sharpe_ratio"].max()) if total_rows else 0.0
    best_return = float(df["total_return_pct"].max()) if total_rows else 0.0
    positive_rows = int((df["total_return_pct"] > 0).sum()) if total_rows else 0
    stage1_total = int(coverage.get("stage1_total_items") or 0)
    queue_supported = stage1_total or int(coverage.get("queue_supported_timeframes") or 0)
    extracted_ok = int(coverage.get("stage1_cached_ok") or 0) or int(coverage.get("extracted_ok") or 0)
    classified = int(coverage.get("stage2_reports") or 0) or int(coverage.get("classified") or 0)
    converted_ok = int(coverage.get("stage2_converted_import_ok") or 0) or int(coverage.get("converted_import_ok") or 0)
    backtested_files = int(coverage.get("stage3_strategy_files") or 0) or int(coverage.get("backtested_strategy_files") or 0)
    stage1_errors = int(coverage.get("stage1_errors") or 0)
    stage1_pending = int(coverage.get("stage1_pending") or 0)
    stage1_retryable = int(coverage.get("stage1_retryable_errors") or 0)
    stage1_status = str(coverage.get("stage1_runner_status") or "unknown")
    stage1_updated_at = str(coverage.get("stage1_runner_updated_at") or "")
    stage2_reports = int(coverage.get("stage2_reports") or 0)
    stage2_unsupported = int(coverage.get("stage2_unsupported") or 0)
    stage2_errors = int(coverage.get("stage2_convert_errors") or 0)
    stage2_retryable = int(coverage.get("stage2_retryable") or 0)
    stage2_queue = int(coverage.get("stage2_queue_candidates") or 0)
    stage2_remaining = int(coverage.get("stage2_remaining_candidates") or 0)
    stage2_recorded_error_items = int(coverage.get("stage2_recorded_error_items") or 0)
    stage2_active_lanes = int(coverage.get("stage2_active_lanes") or 0)
    stage2_status = str(coverage.get("stage2_runner_status") or "unknown")
    stage2_updated_at = str(coverage.get("stage2_runner_updated_at") or "")
    stage2_last_slug = str(coverage.get("stage2_last_processed_slug") or "")
    stage2_model = str(coverage.get("stage2_model") or "")
    stage3_rows = int(coverage.get("stage3_result_rows") or 0)
    stage3_error_rows = int(coverage.get("stage3_error_rows") or 0)
    lookahead_blacklist_count = len(lookahead_blacklist)
    stage1_done_pct = (100.0 * extracted_ok / stage1_total) if stage1_total else 0.0
    stage2_done_pct = (100.0 * (stage2_reports - stage2_retryable) / stage2_queue) if stage2_queue else 0.0

    symbols = sorted(df["symbol"].unique().tolist()) if total_rows else []
    timeframes = sorted(df["timeframe"].unique().tolist()) if total_rows else []
    batches = sorted(df["batch"].unique().tolist()) if total_rows else []
    strategy_payload = build_strategy_payload(rows)

    table_rows = []
    for row in rows:
        table_rows.append(
            f"""
            <tr data-strategy="{_esc(row['strategy'])}" data-symbol="{_esc(row['symbol'])}" data-tf="{_esc(row['timeframe'])}" data-batch="{_esc(row['batch'])}" data-compatibility="{_esc(row['compatibility'])}" onclick="openModal('{_esc(row['strategy'])}')">
              <td>{_esc(row['strategy'])}</td>
              <td>{_esc(row['batch'])}</td>
              <td>{_esc(row['symbol'])}</td>
              <td>{_esc(row['timeframe'])}</td>
              <td>{_esc(row['compatibility'])}</td>
              <td style="color:{'#2ecc71' if row['sharpe_ratio'] > 0 else '#e74c3c'}">{row['sharpe_ratio']:.3f}</td>
              <td>{row['total_return_pct']:+.2f}%</td>
              <td>{row['max_drawdown_pct']:.2f}%</td>
              <td>{row['win_rate']:.1f}%</td>
              <td>{row['num_trades']}</td>
              <td>{row['profit_factor']:.2f}</td>
            </tr>
            """
        )

    unsupported_rows = []
    for row in unsupported:
        unsupported_rows.append(
            f"""
            <tr>
              <td>{_esc(row['name'])}</td>
              <td>{_esc(row['compatibility'])}</td>
              <td><a href="{_esc(row['source_url'])}" target="_blank" style="color:#58a6ff">source</a></td>
              <td>{_esc(row['reason'])}</td>
            </tr>
            """
        )

    lookahead_rows = []
    for row in lookahead_blacklist:
        lookahead_rows.append(
            f"""
            <tr>
              <td>{int(row['queue_rank']) if row.get('queue_rank') is not None else ''}</td>
              <td>{_esc(row['slug'])}</td>
              <td>{_esc(row['name'])}</td>
              <td>{_esc(row['status'])}</td>
              <td>{_esc(row['classification'])}</td>
              <td>{_esc(row['timeframe'])}</td>
              <td>{_esc(row['detection_source'])}</td>
              <td>{_esc(row['pine_file'])}</td>
              <td>{_esc(row['python_file'])}</td>
              <td><a href="{_esc(row['source_url'])}" target="_blank" style="color:#58a6ff">source</a></td>
              <td>{_esc(row['reason'])}</td>
            </tr>
            """
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TradingView Strategies Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #79c0ff; margin-top: 28px; }}
  .timestamp {{ color: #8b949e; font-size: 0.82em; }}
  .stats {{ display:flex; gap:15px; flex-wrap:wrap; margin:20px 0; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 18px; min-width:130px; }}
  .card .value {{ font-size:1.8em; font-weight:bold; color:#58a6ff; }}
  .card .label {{ font-size:0.78em; color:#8b949e; margin-top:4px; }}
  .filter-bar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:15px 0; }}
  .filter-label {{ color:#8b949e; font-size:0.8em; }}
  .filter-btn {{ background:#21262d; border:1px solid #30363d; border-radius:4px; color:#c9d1d9; padding:5px 10px; cursor:pointer; font-family:inherit; }}
  .filter-btn.active {{ background:#1f6feb; border-color:#1f6feb; color:#fff; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }}
  th {{ background:#21262d; color:#8b949e; text-align:left; padding:8px; font-size:0.8em; }}
  td {{ padding:7px 8px; border-top:1px solid #30363d; font-size:0.8em; }}
  tr[data-strategy]:hover td {{ background:#1c2128; cursor:pointer; }}
  .note {{ color:#8b949e; font-size:0.82em; }}
  .page-tabs {{ display:flex; gap:8px; margin:18px 0 12px; }}
  .page-tab-btn {{ background:#21262d; border:1px solid #30363d; border-radius:4px; color:#c9d1d9; padding:7px 12px; cursor:pointer; font-family:inherit; }}
  .page-tab-btn.active {{ background:#1f6feb; border-color:#1f6feb; color:#fff; }}
  .page-section {{ display:none; }}
  .page-section.active {{ display:block; }}
  pre {{ background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px; overflow:auto; max-height:420px; font-size:0.78em; }}
  .modal-overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.78); z-index:1000; overflow-y:auto; }}
  .modal {{ background:#161b22; border:1px solid #30363d; border-radius:10px; max-width:1000px; margin:40px auto; padding:25px; position:relative; }}
  .modal-close {{ position:absolute; top:14px; right:18px; background:none; border:none; color:#8b949e; font-size:1.5em; cursor:pointer; }}
  .tabs {{ display:flex; gap:8px; margin:12px 0 16px; }}
  .tab-btn {{ background:#21262d; border:1px solid #30363d; border-radius:4px; color:#c9d1d9; padding:6px 12px; cursor:pointer; font-family:inherit; }}
  .tab-btn.active {{ background:#1f6feb; border-color:#1f6feb; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}
  .detail-btn {{ background:#21262d; border:1px solid #58a6ff; border-radius:4px; color:#58a6ff; padding:4px 10px; cursor:pointer; font-family:inherit; font-size:0.78em; }}
  .detail-btn:hover {{ background:#1f6feb; color:#fff; }}
  .detail-panel {{ background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:15px; margin-top:10px; }}
  .detail-chart {{ height:220px; margin:10px 0; }}
  .trade-list {{ max-height:350px; overflow-y:auto; font-size:0.75em; }}
  .trade-list th {{ position:sticky; top:0; background:#21262d; z-index:1; }}
  .trade-long {{ color:#2ecc71; }}
  .trade-short {{ color:#e74c3c; }}
  .pnl-pos {{ color:#2ecc71; }}
  .pnl-neg {{ color:#e74c3c; }}
  a {{ color:#58a6ff; }}
</style>
</head>
<body>
<h1>TradingView Strategy Testing Dashboard</h1>
<p class="timestamp">Manual refresh only · Last updated: {now}</p>
<p class="note">Crawl completeness check: local manifest still matches the live TradingView listing end at page <b>226</b>. Requests to pages 227 and 228 return the same terminal payload as page 226, with no `next` page.</p>
<p class="note"><b>Do not read the result row count as crawl progress.</b> The table below shows only finished backtest rows. Stage progress is tracked separately below.</p>

<h2>Pipeline Progress</h2>
<p class="note">Stage 1 runner status: <b>{_esc(stage1_status)}</b>{' · updated ' + _esc(stage1_updated_at) if stage1_updated_at else ''}</p>
<p class="note">Stage 2 local runner status: <b>{_esc(stage2_status)}</b>{' · updated ' + _esc(stage2_updated_at) if stage2_updated_at else ''}{' · model ' + _esc(stage2_model) if stage2_model else ''}{' · last ' + _esc(stage2_last_slug) if stage2_last_slug else ''}</p>
<div class="stats">
  <div class="card"><div class="value">{stage1_total}</div><div class="label">Stage 1 Total Scripts</div></div>
  <div class="card"><div class="value" style="color:#2ecc71">{extracted_ok}</div><div class="label">Stage 1 Cached Pine</div></div>
  <div class="card"><div class="value" style="color:#f39c12">{stage1_errors}</div><div class="label">Stage 1 Errors</div></div>
  <div class="card"><div class="value" style="color:#f1c40f">{stage1_retryable}</div><div class="label">Retryable Errors</div></div>
  <div class="card"><div class="value">{stage2_queue}</div><div class="label">Stage 2 Queue</div></div>
  <div class="card"><div class="value">{stage2_reports}</div><div class="label">Stage 2 Processed</div></div>
  <div class="card"><div class="value">{converted_ok}</div><div class="label">Stage 2 Converted OK</div></div>
  <div class="card"><div class="value" style="color:#f39c12">{stage2_unsupported}</div><div class="label">Stage 2 Unsupported</div></div>
  <div class="card"><div class="value" style="color:{'#f1c40f' if stage2_retryable else '#58a6ff'}">{stage2_retryable}</div><div class="label">Stage 2 Retryable</div></div>
  <div class="card"><div class="value" style="color:{'#e74c3c' if stage2_errors else '#58a6ff'}">{stage2_errors}</div><div class="label">Stage 2 Hard Errors</div></div>
  <div class="card"><div class="value">{stage2_remaining}</div><div class="label">Stage 2 Remaining</div></div>
  <div class="card"><div class="value">{stage2_active_lanes}</div><div class="label">Stage 2 Active Lanes</div></div>
  <div class="card"><div class="value">{backtested_files}</div><div class="label">Stage 3 Backtested Files</div></div>
  <div class="card"><div class="value">{stage3_rows}</div><div class="label">Stage 3 Saved Test Rows</div></div>
  <div class="card"><div class="value" style="color:{'#e74c3c' if stage3_error_rows else '#58a6ff'}">{stage3_error_rows}</div><div class="label">Stage 3 Error Rows</div></div>
  <div class="card"><div class="value" style="color:{'#f1c40f' if lookahead_blacklist_count else '#58a6ff'}">{lookahead_blacklist_count}</div><div class="label">Lookahead Blacklist</div></div>
</div>
<p class="note">Stage 1 completion = <b>{stage1_done_pct:.1f}%</b> of the `5403` crawled TradingView scripts cached locally. Current split: pending=<b>{stage1_pending}</b>, errors=<b>{stage1_errors}</b>, retryable=<b>{stage1_retryable}</b>. Stage 2 local completion = <b>{stage2_done_pct:.1f}%</b> of the active queue, with attempted=<b>{stage2_reports}</b>, converted=<b>{converted_ok}</b>, unsupported=<b>{stage2_unsupported}</b>, retryable=<b>{stage2_retryable}</b>, hard errors=<b>{stage2_errors}</b>, recorded issue items=<b>{stage2_recorded_error_items}</b>, remaining=<b>{stage2_remaining}</b>. Stage 3 rows saved=<b>{stage3_rows}</b>, files backtested=<b>{backtested_files}</b>, error rows=<b>{stage3_error_rows}</b>.</p>

<div class="stats">
  <div class="card"><div class="value">{total_rows}</div><div class="label">Result Rows</div></div>
  <div class="card"><div class="value">{total_strategies}</div><div class="label">Strategies</div></div>
  <div class="card"><div class="value" style="color:#2ecc71">{positive_rows}</div><div class="label">Positive Rows</div></div>
  <div class="card"><div class="value" style="color:#58a6ff">{best_sharpe:.3f}</div><div class="label">Best Sharpe</div></div>
  <div class="card"><div class="value" style="color:#2ecc71">{best_return:+.1f}%</div><div class="label">Best Return</div></div>
  <div class="card"><div class="value">{backtested_files}</div><div class="label">Backtested Files</div></div>
</div>

<div class="page-tabs">
  <button class="page-tab-btn active" onclick="switchPageTab('results', this)">Backtests</button>
  <button class="page-tab-btn" onclick="switchPageTab('unsupported', this)">Unsupported</button>
  <button class="page-tab-btn" onclick="switchPageTab('lookahead', this)">Lookahead Blacklist</button>
</div>

<div id="page-results" class="page-section active">
<h2>Backtest Results</h2>
<div class="filter-bar">
  <span class="filter-label">Batch:</span>
  <button class="filter-btn active" data-group="batch" onclick="setFilter('batch','ALL',this)">All</button>
  {''.join(f'<button class="filter-btn" data-group="batch" onclick="setFilter(&#39;batch&#39;,&#39;{_esc(v)}&#39;,this)">{_esc(v)}</button>' for v in batches)}
  <span class="filter-label" style="margin-left:12px">Symbol:</span>
  <button class="filter-btn active" data-group="symbol" onclick="setFilter('symbol','ALL',this)">All</button>
  {''.join(f'<button class="filter-btn" data-group="symbol" onclick="setFilter(&#39;symbol&#39;,&#39;{_esc(v)}&#39;,this)">{_esc(v)}</button>' for v in symbols)}
  <span class="filter-label" style="margin-left:12px">TF:</span>
  <button class="filter-btn active" data-group="tf" onclick="setFilter('tf','ALL',this)">All</button>
  {''.join(f'<button class="filter-btn" data-group="tf" onclick="setFilter(&#39;tf&#39;,&#39;{_esc(v)}&#39;,this)">{_esc(v)}</button>' for v in timeframes)}
  <span class="filter-label" style="margin-left:12px">Compatibility:</span>
  <button class="filter-btn active" data-group="compatibility" onclick="setFilter('compatibility','ALL',this)">All</button>
  <button class="filter-btn" data-group="compatibility" onclick="setFilter('compatibility','direct',this)">direct</button>
  <button class="filter-btn" data-group="compatibility" onclick="setFilter('compatibility','partial',this)">partial</button>
</div>
<table id="results-table">
  <thead><tr><th>Strategy</th><th>Batch</th><th>Symbol</th><th>TF</th><th>Compat</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Win Rate</th><th>Trades</th><th>PF</th></tr></thead>
  <tbody>
    {''.join(table_rows) if table_rows else '<tr><td colspan="11">No data</td></tr>'}
  </tbody>
</table>
</div>

<div id="page-unsupported" class="page-section">
<h2>Unsupported</h2>
<table>
  <thead><tr><th>Name</th><th>Compatibility</th><th>Source</th><th>Reason</th></tr></thead>
  <tbody>
    {''.join(unsupported_rows) if unsupported_rows else '<tr><td colspan="4">No unsupported entries logged.</td></tr>'}
  </tbody>
</table>
</div>

<div id="page-lookahead" class="page-section">
<h2>Lookahead Blacklist</h2>
<p class="note">These strategies reference Pine `lookahead_on` or equivalent repainting behavior. They should be blacklisted, not repaired or trusted for fair testing.</p>
<table>
  <thead><tr><th>Rank</th><th>Slug</th><th>Name</th><th>Status</th><th>Compat</th><th>TF</th><th>Detected From</th><th>Pine File</th><th>Python File</th><th>Source</th><th>Reason</th></tr></thead>
  <tbody>
    {''.join(lookahead_rows) if lookahead_rows else '<tr><td colspan="11">No lookahead blacklist entries detected.</td></tr>'}
  </tbody>
</table>
</div>

<div class="modal-overlay" id="modalOverlay" onclick="closeModalIfBackground(event)">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <h2 id="modalTitle">Strategy</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('metrics', this)">Metrics</button>
      <button class="tab-btn" onclick="switchTab('detail', this)">Detail</button>
      <button class="tab-btn" onclick="switchTab('report', this)">Report</button>
      <button class="tab-btn" onclick="switchTab('code', this)">Code</button>
    </div>
    <div id="tab-metrics" class="tab-content active"></div>
    <div id="tab-detail" class="tab-content"></div>
    <div id="tab-report" class="tab-content"></div>
    <div id="tab-code" class="tab-content"></div>
  </div>
</div>

<script>
const STRATEGIES = {strategy_payload};
const FILTERS = {{ batch: 'ALL', symbol: 'ALL', tf: 'ALL', compatibility: 'ALL' }};
let detailChart = null;
let detailData = null;

function setFilter(key, value, btn) {{
  FILTERS[key] = value;
  document.querySelectorAll(`.filter-btn[data-group="${{key}}"]`).forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

function switchPageTab(name, btn) {{
  document.querySelectorAll('.page-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.getElementById('page-' + name).classList.add('active');
}}

function applyFilters() {{
  document.querySelectorAll('#results-table tbody tr[data-strategy]').forEach(row => {{
    let visible = true;
    if (FILTERS.batch !== 'ALL' && row.dataset.batch !== FILTERS.batch) visible = false;
    if (FILTERS.symbol !== 'ALL' && row.dataset.symbol !== FILTERS.symbol) visible = false;
    if (FILTERS.tf !== 'ALL' && row.dataset.tf !== FILTERS.tf) visible = false;
    if (FILTERS.compatibility !== 'ALL' && row.dataset.compatibility !== FILTERS.compatibility) visible = false;
    row.style.display = visible ? '' : 'none';
  }});
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function openModal(name) {{
  const s = STRATEGIES[name];
  if (!s) return;
  document.getElementById('modalTitle').textContent = name;
  let metrics = `<p><b>Batch:</b> ${{s.batch}} | <b>Compatibility:</b> ${{s.compatibility}} | <b>Source:</b> ${{s.source_url ? `<a href="${{s.source_url}}" target="_blank">${{s.source_url}}</a>` : 'n/a'}} | <b>Python:</b> ${{s.python_file || 'n/a'}} | <b>Report:</b> ${{s.report_file || 'n/a'}}</p>`;
  metrics += '<table><thead><tr><th>Symbol</th><th>TF</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Win Rate</th><th>Trades</th><th>PF</th><th>Action</th></tr></thead><tbody>';
  s.rows.forEach(r => {{
    metrics += `<tr><td>${{r.symbol}}</td><td>${{r.timeframe}}</td><td>${{r.sharpe_ratio.toFixed(3)}}</td><td>${{r.return_pct.toFixed(2)}}%</td><td>${{r.max_drawdown_pct.toFixed(2)}}%</td><td>${{r.win_rate.toFixed(1)}}%</td><td>${{r.num_trades}}</td><td>${{r.profit_factor.toFixed(2)}}</td><td><button class="detail-btn" onclick="loadDetail('${{name}}','${{r.symbol}}', this)">Orders + Chart</button> <a href="${{r.tv_chart_url}}" target="_blank">TV</a></td></tr>`;
  }});
  metrics += '</tbody></table>';
  document.getElementById('tab-metrics').innerHTML = metrics;
  document.getElementById('tab-detail').innerHTML = '<p class="note">Choose a symbol row in Metrics to load orders and candlestick detail.</p>';
  document.getElementById('tab-report').innerHTML = s.report ? `<pre>${{escHtml(s.report)}}</pre>` : '<p class="note">No report found.</p>';
  document.getElementById('tab-code').innerHTML = s.code ? `<pre>${{escHtml(s.code)}}</pre>` : '<p class="note">No code found.</p>';
  switchTab('metrics', document.querySelector('.tab-btn'));
  document.getElementById('modalOverlay').style.display = 'block';
}}

async function loadDetail(strategy, symbol, btn) {{
  const panel = document.getElementById('tab-detail');
  if (!panel) return;
  if (btn) btn.disabled = true;
  panel.innerHTML = `<p class="note">Loading detail for ${{symbol}}...</p>`;
  switchTab('detail', document.querySelector('.tab-btn[onclick*="detail"]'));
  try {{
    const url = `/api/detail?strategy=${{encodeURIComponent(strategy)}}&symbol=${{encodeURIComponent(symbol)}}`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (data.error) {{
      panel.innerHTML = `<p style="color:#e74c3c">Error: ${{escHtml(data.error)}}</p>`;
      return;
    }}
    detailData = data;
    const m = data.metrics;
    let html = `<div class="detail-panel">`;
    html += `<h3>${{data.symbol}} · ${{data.timeframe}} · ${{data.num_bars}} bars</h3>`;
    html += `<div style="margin:8px 0;font-size:0.85em">Sharpe=<b>${{m.sharpe.toFixed(3)}}</b> | Return=<b>${{m.return_pct > 0 ? '+' : ''}}${{m.return_pct.toFixed(2)}}%</b> | DD=<b>${{m.max_dd_pct.toFixed(2)}}%</b> | WR=<b>${{m.win_rate.toFixed(1)}}%</b> | PF=<b>${{m.profit_factor.toFixed(2)}}</b> | Trades=<b>${{m.num_trades}}</b></div>`;
    html += `<div style="margin:10px 0"><button class="detail-btn" onclick="openCandleChart(detailData)">Open Candlestick Chart</button> <a href="${{data.tv_chart_url}}" target="_blank" style="margin-left:10px">Open on TradingView</a></div>`;
    html += `<h3>Equity Curve</h3><div class="detail-chart"><canvas id="detailEquityChart"></canvas></div>`;
    html += `<h3>Trade History (${{data.trades.length}} trades)</h3>`;
    html += `<div class="trade-list"><table><thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Size</th><th>PnL</th><th>PnL%</th><th>Fees</th><th>Funding</th></tr></thead><tbody>`;
    const maxShow = 200;
    const shown = data.trades.slice(0, maxShow);
    shown.forEach((t, i) => {{
      const dirClass = t.direction === 'LONG' ? 'trade-long' : 'trade-short';
      const pnlClass = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      html += `<tr><td>${{i+1}}</td><td>${{escHtml(t.entry_time.substring(0,16))}}</td><td>${{escHtml(t.exit_time.substring(0,16))}}</td><td class="${{dirClass}}">${{t.direction}}</td><td>${{t.entry_price}}</td><td>${{t.exit_price}}</td><td>${{t.size.toFixed(4)}}</td><td class="${{pnlClass}}">${{t.pnl >= 0 ? '+' : ''}}${{t.pnl.toFixed(2)}}</td><td class="${{pnlClass}}">${{t.pnl_pct >= 0 ? '+' : ''}}${{t.pnl_pct.toFixed(3)}}%</td><td>${{t.fee_cost.toFixed(2)}}</td><td>${{t.funding_cost.toFixed(2)}}</td></tr>`;
    }});
    if (data.trades.length > maxShow) {{
      html += `<tr><td colspan="11" class="note">... and ${{data.trades.length - maxShow}} more trades</td></tr>`;
    }}
    html += `</tbody></table></div></div>`;
    panel.innerHTML = html;

    if (detailChart) detailChart.destroy();
    const ctx = document.getElementById('detailEquityChart').getContext('2d');
    detailChart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: data.equity_labels,
        datasets: [{{
          data: data.equity_curve,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.12)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.1,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
        }}
      }}
    }});
  }} catch (e) {{
    panel.innerHTML = `<p style="color:#e74c3c">Failed to load detail: ${{escHtml(e.message)}}</p>`;
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

function openCandleChart(data) {{
  if (!data || !data.ohlc || data.ohlc.length === 0) return;
  let overlay = document.getElementById('chartOverlay');
  if (!overlay) {{
    overlay = document.createElement('div');
    overlay.id = 'chartOverlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:#0d1117;z-index:2000;display:flex;flex-direction:column;';
    document.body.appendChild(overlay);
  }}
  overlay.style.display = 'flex';
  overlay.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 15px;background:#161b22;border-bottom:1px solid #30363d;">
      <div style="color:#58a6ff;font-size:0.9em;font-weight:bold">
        ${{data.strategy}} · ${{data.symbol}} · ${{data.timeframe}}
        <span style="color:#8b949e;font-weight:normal"> | Sharpe=${{data.metrics.sharpe.toFixed(3)}} | Return=${{data.metrics.return_pct > 0 ? '+' : ''}}${{data.metrics.return_pct.toFixed(2)}}% | DD=${{data.metrics.max_dd_pct.toFixed(2)}}% | Trades=${{data.metrics.num_trades}}</span>
      </div>
      <div id="chartLegend">
        <button onclick="document.getElementById('chartOverlay').style.display='none'" style="background:#e74c3c;border:none;color:#fff;padding:4px 12px;border-radius:4px;cursor:pointer;font-family:monospace">Close (Esc)</button>
      </div>
    </div>
    <div id="candleChartContainer" style="flex:1;position:relative"></div>
    <div id="signalChartContainer" style="height:80px;border-top:1px solid #30363d;position:relative"></div>
  `;
  const container = document.getElementById('candleChartContainer');
  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {{ background: {{ color: '#0d1117' }}, textColor: '#c9d1d9' }},
    grid: {{ vertLines: {{ color: '#1c2128' }}, horzLines: {{ color: '#1c2128' }} }},
    timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#30363d' }},
    rightPriceScale: {{ borderColor: '#30363d' }},
  }});
  const candleSeries = chart.addCandlestickSeries({{
    upColor: '#2ecc71', downColor: '#e74c3c',
    borderUpColor: '#2ecc71', borderDownColor: '#e74c3c',
    wickUpColor: '#2ecc71', wickDownColor: '#e74c3c',
  }});
  candleSeries.setData(data.ohlc);
  if (data.indicators) {{
    Object.entries(data.indicators).forEach(([indName, indInfo]) => {{
      if (indInfo.data && indInfo.data.length > 0) {{
        const series = chart.addLineSeries({{ color: indInfo.color || '#f0883e', lineWidth: 1.5, title: indName }});
        series.setData(indInfo.data);
      }}
    }});
  }}
  const markers = [];
  (data.trade_markers || []).forEach(tm => {{
    markers.push({{ time: tm.entry_time, position: tm.direction === 'LONG' ? 'belowBar' : 'aboveBar', color: tm.direction === 'LONG' ? '#2ecc71' : '#e74c3c', shape: tm.direction === 'LONG' ? 'arrowUp' : 'arrowDown', text: tm.direction[0] }});
    markers.push({{ time: tm.exit_time, position: tm.pnl_pct >= 0 ? 'aboveBar' : 'belowBar', color: tm.pnl_pct >= 0 ? 'rgba(46,204,113,0.8)' : 'rgba(231,76,60,0.8)', shape: 'circle', text: `${{tm.pnl_pct >= 0 ? '+' : ''}}${{tm.pnl_pct.toFixed(2)}}%` }});
  }});
  markers.sort((a,b) => a.time - b.time);
  if (markers.length > 0) candleSeries.setMarkers(markers);

  const sigContainer = document.getElementById('signalChartContainer');
  const sigChart = LightweightCharts.createChart(sigContainer, {{
    width: sigContainer.clientWidth,
    height: sigContainer.clientHeight,
    layout: {{ background: {{ color: '#0d1117' }}, textColor: '#8b949e' }},
    grid: {{ vertLines: {{ color: '#1c2128' }}, horzLines: {{ color: '#1c2128' }} }},
    timeScale: {{ visible: false }},
    rightPriceScale: {{ borderColor: '#30363d' }},
  }});
  const sigSeries = sigChart.addHistogramSeries({{
    priceFormat: {{ type: 'custom', formatter: v => v.toFixed(2) }},
  }});
  sigSeries.setData((data.signals || []).map(s => ({{ time: s.time, value: s.value, color: s.value > 0 ? 'rgba(46,204,113,0.6)' : s.value < 0 ? 'rgba(231,76,60,0.6)' : 'rgba(139,148,158,0.15)' }})));
  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{ if (range) sigChart.timeScale().setVisibleLogicalRange(range); }});
  const resizeObserver = new ResizeObserver(() => {{
    chart.applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
    sigChart.applyOptions({{ width: sigContainer.clientWidth, height: sigContainer.clientHeight }});
  }});
  resizeObserver.observe(container);
  const escHandler = (e) => {{
    if (e.key === 'Escape') {{
      overlay.style.display = 'none';
      document.removeEventListener('keydown', escHandler);
      resizeObserver.disconnect();
      chart.remove();
      sigChart.remove();
    }}
  }};
  document.addEventListener('keydown', escHandler);
}}

function closeModal() {{
  document.getElementById('modalOverlay').style.display = 'none';
}}

function closeModalIfBackground(event) {{
  if (event.target.id === 'modalOverlay') closeModal();
}}

function switchTab(name, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/detail":
            params = parse_qs(parsed.query)
            strategy = params.get("strategy", [""])[0]
            symbol = params.get("symbol", [""])[0]
            payload = run_detail_backtest(strategy, symbol)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        html = render_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView strategy dashboard")
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"TradingView dashboard running at http://127.0.0.1:{args.port}")
    print("Manual refresh only. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
