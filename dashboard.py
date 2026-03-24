#!/usr/bin/env python3
"""
dashboard.py - Live Research Dashboard
=======================================
Web dashboard to monitor the autonomous research loop.

Usage:
    python dashboard.py          # http://localhost:8888
    python dashboard.py --port 9000
"""

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd

from validator import validate_strategy, ValidationResult

STRATEGIES_DIR = Path("strategies")
STRATEGY_FILE = Path("strategy.py")

# --- Page cache: rendered every 30s in background, served instantly ---
_cache_lock = threading.Lock()
_page_cache: dict = {"html": "<html><body>Loading dashboard…</body></html>", "ts": 0.0}

# --- Timeframe cache: persists across renders ---
_tf_cache: dict = {}


def load_results() -> pd.DataFrame:
    """Full load — only used by get_strategy_data() modal. Dashboard render uses targeted queries."""
    from results_db import load_results as _db_load
    return _db_load()


def get_git_log(n: int = 20) -> str:
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def get_current_strategy() -> str:
    try:
        return STRATEGY_FILE.read_text()
    except Exception:
        return "N/A"


def get_strategy_code(strategy_name: str) -> str:
    """Get saved strategy code."""
    path = STRATEGIES_DIR / f"{strategy_name}.py"
    if path.exists():
        return path.read_text()
    return ""


def get_strategy_timeframe(strategy_name: str) -> str:
    """Extract timeframe from strategy code (cached in memory)."""
    import re
    if strategy_name in _tf_cache:
        return _tf_cache[strategy_name]
    code = get_strategy_code(strategy_name)
    tf = "?"
    if code:
        m = re.search(r'timeframe\s*=\s*["\'](\w+)["\']', code)
        if m:
            tf = m.group(1)
    _tf_cache[strategy_name] = tf
    return tf


def run_validation(code: str) -> ValidationResult:
    if not code:
        r = ValidationResult(valid=False)
        r.errors.append("Strategy code not found")
        return r
    return validate_strategy(code)


def build_validation_html(result: ValidationResult) -> str:
    lines = []
    badge = '<span class="badge badge-pass">PASS</span>' if result.valid else '<span class="badge badge-fail">FAIL</span>'
    lines.append(f'<div class="val-header">{badge} Compliance Check</div>')
    if result.errors:
        lines.append('<div class="val-section"><b>Errors</b></div>')
        for e in result.errors:
            lines.append(f'<div class="val-error">✗ {_esc(e)}</div>')
    if result.warnings:
        lines.append('<div class="val-section"><b>Warnings</b></div>')
        for w in result.warnings:
            lines.append(f'<div class="val-warn">⚠ {_esc(w)}</div>')
    if result.info:
        lines.append('<div class="val-section"><b>Info</b></div>')
        for i in result.info:
            lines.append(f'<div class="val-info">ℹ {_esc(i)}</div>')
    return "\n".join(lines)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_strategy_data(df: pd.DataFrame) -> str:
    """Build JSON data for strategy modals."""
    if df.empty or "strategy" not in df.columns:
        return "{}"

    strategies = {}
    for name, group in df.groupby("strategy"):
        code = get_strategy_code(str(name))
        val = run_validation(code)
        rows_by_period = {}
        for _, row in group.iterrows():
            period = row.get("period", "train")
            if period not in rows_by_period:
                rows_by_period[period] = []
            rows_by_period[period].append({
                "symbol": row.get("symbol", ""),
                "sharpe": round(float(row.get("sharpe", 0) or 0), 4),
                "return_pct": round(float(row.get("return_pct", 0) or 0), 2),
                "cagr_pct": round(float(row.get("cagr_pct", 0) or 0), 2),
                "max_dd_pct": round(float(row.get("max_dd_pct", 0) or 0), 2),
                "win_rate": round(float(row.get("win_rate", 0) or 0), 1),
                "profit_factor": round(float(row.get("profit_factor", 0) or 0), 2),
                "trades": int(row.get("trades", 0) or 0),
                "sortino": round(float(row.get("sortino", 0) or 0), 4),
                "calmar": round(float(row.get("calmar", 0) or 0), 4),
                "status": row.get("status", ""),
                "period": period,
            })
        # Compute avg across symbols for train period
        train_rows = rows_by_period.get("train", [])
        avg_sharpe = sum(r["sharpe"] for r in train_rows) / len(train_rows) if train_rows else 0
        avg_dd = sum(r["max_dd_pct"] for r in train_rows) / len(train_rows) if train_rows else 0
        avg_return = sum(r["return_pct"] for r in train_rows) / len(train_rows) if train_rows else 0

        strategies[str(name)] = {
            "name": str(name),
            "code": code,
            "valid": val.valid,
            "validation_html": build_validation_html(val),
            "rows_by_period": rows_by_period,
            "avg_sharpe": round(avg_sharpe, 4),
            "avg_dd": round(avg_dd, 2),
            "avg_return": round(avg_return, 2),
        }
    return json.dumps(strategies, ensure_ascii=False)


def _build_avg_table_rows(df: pd.DataFrame, limit: int = 50, presorted: bool = False) -> str:
    """Build table rows showing average metrics per strategy across all symbols."""
    if df.empty or "sharpe" not in df.columns:
        return '<tr><td colspan="8" class="no-data" style="padding:10px;color:#8b949e">No data</td></tr>'

    if presorted:
        # df is already aggregated+sorted by SQL query_avg_rows()
        agg = df.set_index("strategy")
    else:
        agg = df.groupby("strategy").agg({
            "sharpe": "mean", "return_pct": "mean", "max_dd_pct": "mean",
            "win_rate": "mean", "trades": "mean", "status": "first",
        }).sort_values("sharpe", ascending=False).head(limit)

    rows = ""
    for strategy, row in agg.iterrows():
        sharpe_val = float(row.get("sharpe", 0) or 0)
        dd_val = float(row.get("max_dd_pct", 0) or 0)
        color = "#2ecc71" if sharpe_val > 0 else "#e74c3c"
        dd_color = "#2ecc71" if dd_val > -50 else "#e74c3c"
        status = row.get("status", "")
        badge_cls = f"badge-{status}" if status in ("keep", "discard", "crash") else "badge-discard"
        rows += f"""
        <tr data-strategy="{_esc(str(strategy))}" data-symbol="AVG" data-sharpe="{sharpe_val:.4f}" onclick="openModal('{_esc(str(strategy))}')">
            <td>{_esc(str(strategy))}</td>
            <td style="color:#8b949e">AVG (all)</td>
            <td style="color:{color}">{sharpe_val:.3f}</td>
            <td>{float(row.get('return_pct', 0) or 0):+.1f}%</td>
            <td style="color:{dd_color}">{dd_val:.1f}%</td>
            <td>{float(row.get('win_rate', 0) or 0):.0f}%</td>
            <td>{int(row.get('trades', 0) or 0)}</td>
            <td><span class="badge {badge_cls}">{status}</span></td>
        </tr>"""
    return rows


def render_html() -> str:
    from results_db import (query_stats, query_top_rows, query_avg_rows,
                             query_chart_data, query_distinct_symbols)

    git_log = get_git_log()
    current_strategy = get_current_strategy()
    current_val = run_validation(current_strategy)

    # Validate current strategy
    current_val_html = build_validation_html(current_val)
    current_val_badge = "badge-pass" if current_val.valid else "badge-fail"
    current_val_label = "PASS" if current_val.valid else "FAIL"

    # --- Targeted SQL queries (no full 55k-row scan) ---
    stats = query_stats()
    total       = int(stats.get("total", 0) or 0)
    kept        = int(stats.get("kept", 0) or 0)
    discarded   = int(stats.get("discarded", 0) or 0)
    crashed     = int(stats.get("crashed", 0) or 0)
    best_sharpe = float(stats.get("best_sharpe", 0) or 0)
    train_total = int(stats.get("train_total", 0) or 0)
    train_kept  = int(stats.get("train_kept", 0) or 0)
    train_best  = float(stats.get("train_best", 0) or 0)
    test_total  = int(stats.get("test_total", 0) or 0)
    test_kept   = int(stats.get("test_kept", 0) or 0)
    test_best   = float(stats.get("test_best", 0) or 0)

    # Top rows for tables (already sorted by Sharpe DESC, limit 100)
    train_df     = query_top_rows("train", limit=100)
    train_avg_df = query_avg_rows("train", limit=50)
    test_df      = query_top_rows("test",  limit=100)
    test_avg_df  = query_avg_rows("test",  limit=50)

    train_rows     = _build_table_rows(train_df, presorted=True)
    train_avg_rows = _build_avg_table_rows(train_avg_df, presorted=True)
    test_rows      = _build_table_rows(test_df, presorted=True)
    test_avg_rows  = _build_avg_table_rows(test_avg_df, presorted=True)

    # Symbols and timeframes
    symbols = query_distinct_symbols()
    symbols_json = json.dumps(symbols)
    timeframes = sorted(set(tf for tf in _tf_cache.values() if tf != "?"))

    # Chart data: Sharpe over time for BTCUSDT train
    chart_data, chart_labels, running_best_data = "[]", "[]", "[]"
    if train_total > 0:
        sharpes = query_chart_data("BTCUSDT", "train")
        if sharpes:
            labels = [f"#{i+1}" for i in range(len(sharpes))]
            running_best = []
            best_so_far = -999.0
            for s in sharpes:
                best_so_far = max(best_so_far, s)
                running_best.append(round(best_so_far, 4))
            chart_data = json.dumps([round(s, 4) for s in sharpes])
            chart_labels = json.dumps(labels)
            running_best_data = json.dumps(running_best)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pre-compute test section (avoid nested f""" triple-quote issue in Python 3.11)
    if test_total > 0:
        test_section = (
            '<div class="filter-bar" id="test-filter-bar">\n'
            '  <label>Symbol:</label>\n'
            '  <button class="filter-btn active" onclick="filterTable(\'test\', \'ALL\')">All</button>\n'
            + ''.join(f'  <button class="filter-btn" onclick="filterTable(\'test\', \'{s}\')">{s}</button>\n' for s in symbols)
            + '  <button class="filter-btn" onclick="filterTable(\'test\', \'AVG\')" style="border-color:#f0883e;color:#f0883e">Avg All</button>\n'
            '  <span style="margin:0 8px;color:#30363d">|</span>\n'
            '  <label>TF:</label>\n'
            '  <button class="filter-btn active" onclick="filterTF(\'test\', \'ALL\')">All</button>\n'
            + ''.join(f'  <button class="filter-btn" onclick="filterTF(\'test\', \'{tf}\')">{tf}</button>\n' for tf in timeframes)
            + '  <span style="margin:0 8px;color:#30363d">|</span>\n'
            '  <label>Status:</label>\n'
            '  <button class="filter-btn active" onclick="filterStatus(\'test\', \'ALL\')">All</button>\n'
            '  <button class="filter-btn" onclick="filterStatus(\'test\', \'keep\')" style="border-color:#2ecc71;color:#2ecc71">Keep</button>\n'
            '  <button class="filter-btn" onclick="filterStatus(\'test\', \'discard\')" style="border-color:#e74c3c;color:#e74c3c">Discard</button>\n'
            '  <span class="filter-info" id="test-filter-info"></span>\n'
            '</div>\n'
            '<table id="test-table">\n'
            '  <thead><tr><th>Strategy</th><th>Symbol</th><th>TF</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Win Rate</th><th>Trades</th><th>Status</th></tr></thead>\n'
            f'  <tbody id="test-tbody-rows">{test_rows}</tbody>\n'
            f'  <tbody id="test-tbody-avg" style="display:none">{test_avg_rows}</tbody>\n'
            '</table>'
        )
    else:
        test_section = '<p class="no-data">No test results yet — kept strategies are automatically evaluated on 2025+ data.</p>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="600">
<!-- Manual refresh: press F5 or click button below -->
<title>LLM Trading Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #79c0ff; margin-top: 30px; }}
  h3 {{ color: #79c0ff; margin: 10px 0; font-size: 0.95em; }}
  .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; min-width: 110px; }}
  .stat-card .value {{ font-size: 1.8em; font-weight: bold; color: #58a6ff; }}
  .stat-card .label {{ font-size: 0.78em; color: #8b949e; margin-top: 4px; }}
  .period-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .period-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
  .period-box.train {{ border-color: #1f6feb; }}
  .period-box.test {{ border-color: #238636; }}
  .period-label {{ font-size: 0.75em; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }}
  .period-box.train .period-label {{ color: #58a6ff; }}
  .period-box.test .period-label {{ color: #2ecc71; }}
  .mini-stats {{ display: flex; gap: 15px; margin-bottom: 10px; }}
  .mini-stat {{ text-align: center; }}
  .mini-stat .v {{ font-size: 1.4em; font-weight: bold; }}
  .mini-stat .l {{ font-size: 0.72em; color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
  th {{ background: #21262d; padding: 8px; text-align: left; color: #8b949e; font-size: 0.8em; cursor: pointer; user-select: none; }}
  th:hover {{ color: #58a6ff; }}
  th.sorted-asc::after {{ content: ' ▲'; color: #58a6ff; }}
  th.sorted-desc::after {{ content: ' ▼'; color: #58a6ff; }}
  td {{ padding: 6px 8px; border-top: 1px solid #30363d; font-size: 0.8em; }}
  tr[data-strategy]:hover td {{ background: #1c2128; cursor: pointer; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; height: 280px; }}
  pre {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; overflow: auto; max-height: 400px; font-size: 0.78em; color: #e6edf3; }}
  .git-log {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; max-height: 180px; overflow: auto; font-size: 0.78em; }}
  .timestamp {{ color: #8b949e; font-size: 0.78em; }}
  .badge {{ padding: 2px 7px; border-radius: 4px; font-size: 0.72em; font-weight: bold; }}
  .badge-keep {{ background: #1a4731; color: #2ecc71; }}
  .badge-discard {{ background: #3d1f1f; color: #e74c3c; }}
  .badge-crash {{ background: #3d2f10; color: #f39c12; }}
  .badge-pass {{ background: #1a4731; color: #2ecc71; }}
  .badge-fail {{ background: #3d1f1f; color: #e74c3c; }}
  .compliance-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 15px 0; font-size: 0.82em; }}
  .val-header {{ font-weight: bold; margin-bottom: 8px; }}
  .val-section {{ color: #8b949e; margin: 6px 0 3px; font-size: 0.9em; }}
  .val-error {{ color: #e74c3c; padding: 2px 0; }}
  .val-warn {{ color: #f39c12; padding: 2px 0; }}
  .val-info {{ color: #8b949e; padding: 2px 0; }}
  /* Modal */
  .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.75); z-index: 1000; overflow-y: auto; }}
  .modal {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; max-width: 900px; margin: 40px auto; padding: 25px; position: relative; }}
  .modal-close {{ position: absolute; top: 15px; right: 20px; background: none; border: none; color: #8b949e; font-size: 1.4em; cursor: pointer; line-height: 1; }}
  .modal-close:hover {{ color: #c9d1d9; }}
  .modal h2 {{ margin-top: 0; color: #58a6ff; }}
  .modal-tabs {{ display: flex; gap: 0; margin: 15px 0; border-bottom: 1px solid #30363d; }}
  .tab-btn {{ background: none; border: none; color: #8b949e; padding: 8px 16px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.85em; border-bottom: 2px solid transparent; }}
  .tab-btn.active {{ color: #58a6ff; border-bottom-color: #58a6ff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  .metrics-table th {{ background: #21262d; padding: 6px 10px; text-align: left; color: #8b949e; }}
  .metrics-table td {{ padding: 5px 10px; border-top: 1px solid #30363d; }}
  .no-data {{ color: #8b949e; font-style: italic; padding: 10px 0; }}
  .detail-btn {{ background: #21262d; border: 1px solid #58a6ff; border-radius: 4px; color: #58a6ff; padding: 2px 8px; cursor: pointer; font-family: 'Courier New',monospace; font-size: 0.72em; white-space: nowrap; }}
  .detail-btn:hover {{ background: #1f6feb; color: #fff; }}
  .detail-btn.loading {{ opacity: 0.5; cursor: wait; }}
  .detail-panel {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-top: 10px; }}
  .detail-panel h4 {{ color: #58a6ff; margin: 0 0 10px; }}
  .detail-chart {{ height: 220px; margin: 10px 0; }}
  .trade-list {{ max-height: 350px; overflow-y: auto; font-size: 0.75em; }}
  .trade-list table {{ width: 100%; }}
  .trade-list th {{ position: sticky; top: 0; background: #21262d; z-index: 1; }}
  .trade-long {{ color: #2ecc71; }}
  .trade-short {{ color: #e74c3c; }}
  .pnl-pos {{ color: #2ecc71; }}
  .pnl-neg {{ color: #e74c3c; }}
  /* Symbol filter */
  .filter-bar {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }}
  .filter-bar label {{ color: #8b949e; font-size: 0.78em; }}
  .filter-btn {{ background: #21262d; border: 1px solid #30363d; border-radius: 4px; color: #8b949e; padding: 4px 12px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.78em; }}
  .filter-btn:hover {{ border-color: #58a6ff; color: #c9d1d9; }}
  .filter-btn.active {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}
  .filter-info {{ color: #8b949e; font-size: 0.72em; margin-left: 8px; }}
  /* Ranking mode */
  .rank-bar {{ display: flex; gap: 8px; align-items: center; margin: 8px 0; flex-wrap: wrap; }}
  .rank-btn {{ background: #21262d; border: 1px solid #30363d; border-radius: 4px; color: #8b949e; padding: 4px 12px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.78em; }}
  .rank-btn:hover {{ border-color: #58a6ff; color: #c9d1d9; }}
  .rank-btn.active {{ background: #238636; border-color: #238636; color: #fff; }}
</style>
</head>
<body>
<h1>⚡ LLM Trading Research Dashboard</h1>
<p class="timestamp">Auto-refresh every 10min · Last updated: {now}</p>

<div class="stats">
  <div class="stat-card"><div class="value">{total}</div><div class="label">Total Rows</div></div>
  <div class="stat-card"><div class="value" style="color:#2ecc71">{kept}</div><div class="label">Kept</div></div>
  <div class="stat-card"><div class="value" style="color:#e74c3c">{discarded}</div><div class="label">Discarded</div></div>
  <div class="stat-card"><div class="value" style="color:#f39c12">{crashed}</div><div class="label">Crashed</div></div>
  <div class="stat-card"><div class="value" style="color:#2ecc71">{best_sharpe:.3f}</div><div class="label">Best Sharpe</div></div>
</div>

<div class="period-grid">
  <div class="period-box train">
    <div class="period-label">Train Period (2021–2024)</div>
    <div class="mini-stats">
      <div class="mini-stat"><div class="v">{train_total}</div><div class="l">Results</div></div>
      <div class="mini-stat"><div class="v" style="color:#2ecc71">{train_kept}</div><div class="l">Kept</div></div>
      <div class="mini-stat"><div class="v" style="color:#58a6ff">{train_best:.3f}</div><div class="l">Best Sharpe</div></div>
    </div>
  </div>
  <div class="period-box test">
    <div class="period-label">Test Period (2025+)</div>
    <div class="mini-stats">
      <div class="mini-stat"><div class="v">{test_total}</div><div class="l">Results</div></div>
      <div class="mini-stat"><div class="v" style="color:#2ecc71">{test_kept}</div><div class="l">Kept</div></div>
      <div class="mini-stat"><div class="v" style="color:#58a6ff">{f"{test_best:.3f}" if test_total > 0 else "—"}</div><div class="l">Best Sharpe</div></div>
    </div>
  </div>
</div>

<div class="chart-container">
  <canvas id="sharpeChart"></canvas>
</div>

<h2>Train Results <span style="font-size:0.7em;color:#8b949e">(click row for details)</span></h2>
<div class="filter-bar" id="train-filter-bar">
  <label>Symbol:</label>
  <button class="filter-btn active" onclick="filterTable('train', 'ALL')">All</button>
  {''.join(f'<button class="filter-btn" onclick="filterTable(&#39;train&#39;, &#39;{s}&#39;)">{s}</button>' for s in symbols)}
  <button class="filter-btn" onclick="filterTable('train', 'AVG')" style="border-color:#f0883e;color:#f0883e">Avg All</button>
  <span style="margin:0 8px;color:#30363d">|</span>
  <label>TF:</label>
  <button class="filter-btn active" onclick="filterTF('train', 'ALL')">All</button>
  {''.join(f'<button class="filter-btn" onclick="filterTF(&#39;train&#39;, &#39;{tf}&#39;)">{tf}</button>' for tf in timeframes)}
  <span style="margin:0 8px;color:#30363d">|</span>
  <label>Status:</label>
  <button class="filter-btn active" onclick="filterStatus('train', 'ALL')">All</button>
  <button class="filter-btn" onclick="filterStatus('train', 'keep')" style="border-color:#2ecc71;color:#2ecc71">Keep</button>
  <button class="filter-btn" onclick="filterStatus('train', 'discard')" style="border-color:#e74c3c;color:#e74c3c">Discard</button>
  <span class="filter-info" id="train-filter-info"></span>
</div>
<table id="train-table">
  <thead><tr>
    <th>Strategy</th><th>Symbol</th><th>TF</th><th>Sharpe</th><th>Return</th>
    <th>Max DD</th><th>Win Rate</th><th>Trades</th><th>Status</th>
  </tr></thead>
  <tbody id="train-tbody-rows">{train_rows}</tbody>
  <tbody id="train-tbody-avg" style="display:none">{train_avg_rows}</tbody>
</table>

<h2>Test Results (2025+) <span style="font-size:0.7em;color:#8b949e">(click row for details)</span></h2>
{test_section}

<h2>Current Strategy Compliance</h2>
<div class="compliance-box">
  <div style="margin-bottom:8px"><span class="badge {current_val_badge}">{current_val_label}</span> <b>strategy.py</b></div>
  {current_val_html}
</div>

<h2>Recent Git Commits</h2>
<div class="git-log"><pre>{git_log}</pre></div>

<h2>Current strategy.py</h2>
<pre>{_esc(current_strategy[:3000])}{"..." if len(current_strategy) > 3000 else ""}</pre>

<!-- Strategy Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModalIfBackground(event)">
  <div class="modal" id="modalContent">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <h2 id="modalTitle">Strategy</h2>
    <div class="modal-tabs">
      <button class="tab-btn active" onclick="switchTab('metrics')">Metrics</button>
      <button class="tab-btn" onclick="switchTab('compliance')">Compliance</button>
      <button class="tab-btn" onclick="switchTab('code')">Code</button>
    </div>
    <div id="tab-metrics" class="tab-content active"></div>
    <div id="tab-compliance" class="tab-content"></div>
    <div id="tab-code" class="tab-content"></div>
  </div>
</div>

<script>
const STRATEGIES = {{}};  // lazy-loaded per strategy via /api/strategy

const ctx = document.getElementById('sharpeChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{
        label: 'Sharpe (BTCUSDT train)',
        data: {chart_data},
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.1,
      }},
      {{
        label: 'Running Best',
        data: {running_best_data},
        borderColor: '#2ecc71',
        borderWidth: 2,
        pointRadius: 0,
        borderDash: [5,5],
        tension: 0,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 20 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }}, title: {{ display: true, text: 'Sharpe Ratio', color: '#8b949e' }} }}
    }}
  }}
}});

function openModal(strategyName) {{
  if (STRATEGIES[strategyName]) {{
    _renderModal(strategyName, STRATEGIES[strategyName]);
    return;
  }}
  // Lazy-load strategy data from server
  document.getElementById('modalTitle').textContent = strategyName + ' (loading…)';
  document.getElementById('tab-metrics').innerHTML = '<p style="color:#8b949e;padding:10px">Loading…</p>';
  document.getElementById('tab-compliance').innerHTML = '';
  document.getElementById('tab-code').innerHTML = '';
  document.getElementById('modalOverlay').style.display = 'block';
  switchTab('metrics');
  fetch('/api/strategy?name=' + encodeURIComponent(strategyName))
    .then(r => r.json())
    .then(data => {{
      STRATEGIES[strategyName] = data;
      _renderModal(strategyName, data);
    }})
    .catch(err => {{
      document.getElementById('tab-metrics').innerHTML = '<p style="color:#e74c3c">Error: ' + err + '</p>';
    }});
}}

function _renderModal(strategyName, s) {{
  if (!s) return;

  document.getElementById('modalTitle').textContent = strategyName;

  // Metrics tab — full detail per symbol with averages
  let metricsHtml = '';
  if (s.avg_sharpe !== undefined) {{
    const ddColor = s.avg_dd > -50 ? '#2ecc71' : '#e74c3c';
    const shColor = s.avg_sharpe > 0 ? '#2ecc71' : '#e74c3c';
    metricsHtml += `<div style="margin:10px 0;padding:10px;background:#21262d;border-radius:6px;font-size:0.9em">
      <b>Overall Average:</b> Sharpe=<span style="color:${{shColor}}">${{s.avg_sharpe.toFixed(3)}}</span> | Return=${{s.avg_return > 0 ? '+' : ''}}${{s.avg_return.toFixed(1)}}% | DD=<span style="color:${{ddColor}}">${{s.avg_dd.toFixed(1)}}%</span>
    </div>`;
  }}
  for (const [period, rows] of Object.entries(s.rows_by_period)) {{
    metricsHtml += `<h3>${{period === 'train' ? '📊 Train (2021–2024)' : '🧪 Test (2025+)'}}</h3>`;
    metricsHtml += '<table class="metrics-table"><thead><tr><th>Symbol</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>Return</th><th>CAGR</th><th>Max DD</th><th>Win Rate</th><th>PF</th><th>Trades</th><th>Status</th><th></th></tr></thead><tbody>';
    for (const r of rows) {{
      const color = r.sharpe > 0 ? '#2ecc71' : '#e74c3c';
      const ddColor = r.max_dd_pct > -50 ? '#c9d1d9' : '#e74c3c';
      metricsHtml += `<tr>
        <td>${{r.symbol}}</td>
        <td style="color:${{color}}">${{r.sharpe.toFixed(3)}}</td>
        <td>${{(r.sortino||0).toFixed(3)}}</td>
        <td>${{(r.calmar||0).toFixed(3)}}</td>
        <td>${{r.return_pct > 0 ? '+' : ''}}${{r.return_pct.toFixed(1)}}%</td>
        <td>${{(r.cagr_pct||0) > 0 ? '+' : ''}}${{(r.cagr_pct||0).toFixed(1)}}%</td>
        <td style="color:${{ddColor}}">${{r.max_dd_pct.toFixed(1)}}%</td>
        <td>${{r.win_rate.toFixed(0)}}%</td>
        <td>${{(r.profit_factor||0).toFixed(2)}}</td>
        <td>${{r.trades}}</td>
        <td style="color:${{r.status === 'keep' ? '#2ecc71' : '#e74c3c'}}">${{r.status}}</td>
        <td><button class="detail-btn" onclick="event.stopPropagation();loadDetail('${{strategyName}}','${{r.symbol}}','${{period}}')">View Detail</button></td>
      </tr>`;
    }}
    metricsHtml += '</tbody></table>';
  }}
  if (!metricsHtml) metricsHtml = '<p class="no-data">No metrics available.</p>';
  metricsHtml += '<div id="detail-view" style="margin-top:15px"></div>';
  document.getElementById('tab-metrics').innerHTML = metricsHtml;

  // Compliance tab
  document.getElementById('tab-compliance').innerHTML = `<div class="compliance-box">${{s.validation_html}}</div>`;

  // Code tab
  if (s.code) {{
    document.getElementById('tab-code').innerHTML = `<pre style="max-height:500px">${{escHtml(s.code)}}</pre>`;
  }} else {{
    document.getElementById('tab-code').innerHTML = '<p class="no-data">Strategy code not saved (only kept strategies are archived).</p>';
  }}

  // Reset to metrics tab
  switchTab('metrics');

  document.getElementById('modalOverlay').style.display = 'block';
}}

function closeModal() {{
  document.getElementById('modalOverlay').style.display = 'none';
}}

function closeModalIfBackground(event) {{
  if (event.target === document.getElementById('modalOverlay')) closeModal();
}}

function switchTab(name) {{
  document.querySelectorAll('.tab-btn').forEach((b, i) => {{
    b.classList.toggle('active', ['metrics','compliance','code'][i] === name);
  }});
  document.querySelectorAll('.tab-content').forEach(c => {{
    c.classList.toggle('active', c.id === 'tab-' + name);
  }});
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

// --- Symbol filter ---
function filterTable(tableId, symbol) {{
  const tbodyRows = document.getElementById(tableId + '-tbody-rows');
  const tbodyAvg = document.getElementById(tableId + '-tbody-avg');
  if (!tbodyRows) return;

  // Toggle between rows view and avg view
  if (symbol === 'AVG') {{
    tbodyRows.style.display = 'none';
    if (tbodyAvg) tbodyAvg.style.display = '';
  }} else {{
    tbodyRows.style.display = '';
    if (tbodyAvg) tbodyAvg.style.display = 'none';

    // Filter individual rows
    const rows = tbodyRows.querySelectorAll('tr[data-strategy]');
    let shown = 0;
    rows.forEach(row => {{
      const rowSymbol = row.getAttribute('data-symbol') || '';
      const visible = symbol === 'ALL' || rowSymbol === symbol;
      row.style.display = visible ? '' : 'none';
      if (visible) shown++;
    }});
    const info = document.getElementById(tableId + '-filter-info');
    if (info) info.textContent = symbol === 'ALL' ? '' : `Showing ${{shown}} rows for ${{symbol}}`;
  }}

  // Update button states
  const filterBar = document.getElementById(tableId + '-filter-bar');
  if (filterBar) {{
    filterBar.querySelectorAll('.filter-btn').forEach(btn => {{
      const btnSymbol = btn.textContent.trim();
      const mapping = {{'All Rows': 'ALL', 'Avg All Symbols': 'AVG'}};
      const mapped = mapping[btnSymbol] || btnSymbol;
      btn.classList.toggle('active', mapped === symbol);
    }});
  }}

  const info = document.getElementById(tableId + '-filter-info');
  if (info && symbol === 'AVG') info.textContent = 'Showing average across all symbols per strategy';
}}

// --- Status filter ---
function filterStatus(tableId, status) {{
  const tbodyRows = document.getElementById(tableId + '-tbody-rows');
  if (!tbodyRows) return;
  const rows = tbodyRows.querySelectorAll('tr[data-strategy]');
  rows.forEach(row => {{
    const badge = row.querySelector('.badge');
    const rowStatus = badge ? badge.textContent.trim() : '';
    if (status === 'ALL' || rowStatus === status) {{
      row.style.display = '';
    }} else {{
      row.style.display = 'none';
    }}
  }});
}}

// --- Column sort ---
document.querySelectorAll('th').forEach(th => {{
  th.addEventListener('click', function() {{
    const table = this.closest('table');
    if (!table) return;
    const tbody = table.querySelector('tbody[style*="display: none"]') ? null : table.querySelector('tbody:not([style*="display: none"])');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr[data-strategy]'));
    if (rows.length === 0) return;
    const colIdx = Array.from(this.parentNode.children).indexOf(this);
    const isDesc = this.classList.contains('sorted-asc');
    // Clear all sort indicators in this header row
    this.parentNode.querySelectorAll('th').forEach(h => h.classList.remove('sorted-asc','sorted-desc'));
    this.classList.add(isDesc ? 'sorted-desc' : 'sorted-asc');
    rows.sort((a, b) => {{
      let aVal = a.children[colIdx]?.textContent.replace(/[%$+,]/g,'').trim() || '';
      let bVal = b.children[colIdx]?.textContent.replace(/[%$+,]/g,'').trim() || '';
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) {{
        return isDesc ? aNum - bNum : bNum - aNum;
      }}
      return isDesc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }});
    rows.forEach(row => tbody.appendChild(row));
  }});
}});

// --- Timeframe filter ---
function filterTF(tableId, tf) {{
  const tbodyRows = document.getElementById(tableId + '-tbody-rows');
  if (!tbodyRows) return;
  const rows = tbodyRows.querySelectorAll('tr[data-strategy]');
  let shown = 0;
  rows.forEach(row => {{
    const rowTf = row.getAttribute('data-tf') || '';
    if (tf === 'ALL' || rowTf === tf) {{
      if (row.style.display !== 'none' || tf !== 'ALL') {{ shown++; }}
      // Only hide by TF, don't override symbol filter
      row.dataset.tfHidden = (tf !== 'ALL' && rowTf !== tf) ? '1' : '0';
    }} else {{
      row.dataset.tfHidden = '1';
    }}
    row.style.display = (row.dataset.tfHidden === '1') ? 'none' : '';
  }});
  // Update TF button states
  const filterBar = document.getElementById(tableId + '-filter-bar');
  if (filterBar) {{
    let inTfSection = false;
    filterBar.querySelectorAll('.filter-btn').forEach(btn => {{
      if (btn.previousElementSibling && btn.previousElementSibling.textContent === 'TF:') inTfSection = true;
      if (btn.previousElementSibling && btn.previousElementSibling.textContent === 'Symbol:') inTfSection = false;
      if (inTfSection) {{
        btn.classList.toggle('active', btn.textContent === (tf === 'ALL' ? 'All' : tf));
      }}
    }});
  }}
}}

// --- Detail view: equity chart + trade list ---
let detailChart = null;

async function loadDetail(strategy, symbol, period) {{
  const dv = document.getElementById('detail-view');
  if (!dv) return;
  dv.innerHTML = '<p style="color:#8b949e">Loading backtest for ' + symbol + ' (' + period + ')...</p>';

  try {{
    const url = `/api/detail?strategy=${{encodeURIComponent(strategy)}}&symbol=${{encodeURIComponent(symbol)}}&period=${{encodeURIComponent(period)}}`;
    const resp = await fetch(url);
    const data = await resp.json();

    if (data.error) {{
      dv.innerHTML = `<p style="color:#e74c3c">Error: ${{escHtml(data.error)}}</p>`;
      return;
    }}

    let html = `<div class="detail-panel">`;
    html += `<h4>${{symbol}} · ${{period === 'train' ? 'Train 2021–2024' : 'Test 2025+'}} · ${{data.timeframe}} · ${{data.num_bars}} bars</h4>`;

    // Summary
    const m = data.metrics;
    html += `<div style="margin:8px 0;font-size:0.85em">`;
    html += `Sharpe=<b>${{m.sharpe.toFixed(3)}}</b> | Return=<b>${{m.return_pct > 0 ? '+' : ''}}${{m.return_pct.toFixed(1)}}%</b> | `;
    html += `DD=<b>${{m.max_dd_pct.toFixed(1)}}%</b> | WR=<b>${{m.win_rate.toFixed(0)}}%</b> | `;
    html += `PF=<b>${{m.profit_factor.toFixed(2)}}</b> | Trades=<b>${{m.num_trades}}</b> | `;
    html += `Fees=$<b>${{m.total_fees.toFixed(0)}}</b> | Funding=$<b>${{m.total_funding.toFixed(0)}}</b>`;
    html += `</div>`;

    // Button to open candlestick chart overlay
    html += `<div style="margin:10px 0">`;
    html += `<button class="detail-btn" style="font-size:0.9em;padding:6px 16px" onclick="openCandleChart(detailData)">Open Candlestick Chart (zoom/scroll)</button>`;
    html += `<span style="color:#8b949e;font-size:0.75em;margin-left:10px">Shows OHLC candles, EMA lines, entry/exit markers</span>`;
    html += `</div>`;

    // Equity chart
    html += `<h4>Equity Curve</h4>`;
    html += `<div class="detail-chart"><canvas id="detailEquityChart"></canvas></div>`;

    // Trade list
    html += `<h4 style="margin-top:12px">Trade History (${{data.trades.length}} trades)</h4>`;
    html += `<div class="trade-list"><table class="metrics-table">`;
    html += `<thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Size</th><th>PnL</th><th>PnL%</th><th>Balance</th><th>Fees</th><th>Funding</th></tr></thead><tbody>`;

    const trades = data.trades;
    const maxShow = 200;
    const shown = trades.slice(0, maxShow);
    for (let i = 0; i < shown.length; i++) {{
      const t = shown[i];
      const dirClass = t.direction === 'LONG' ? 'trade-long' : 'trade-short';
      const pnlClass = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const entryShort = t.entry_time.substring(0, 16);
      const exitShort = t.exit_time.substring(0, 16);
      html += `<tr>
        <td>${{i+1}}</td>
        <td>${{entryShort}}</td>
        <td>${{exitShort}}</td>
        <td class="${{dirClass}}">${{t.direction}}</td>
        <td>${{t.entry_price.toLocaleString()}}</td>
        <td>${{t.exit_price.toLocaleString()}}</td>
        <td>${{t.size.toFixed(3)}}</td>
        <td class="${{pnlClass}}">${{t.pnl >= 0 ? '+' : ''}}${{t.pnl.toFixed(2)}}</td>
        <td class="${{pnlClass}}">${{t.pnl_pct >= 0 ? '+' : ''}}${{t.pnl_pct.toFixed(2)}}%</td>
        <td>${{t.balance ? '$' + t.balance.toLocaleString() : ''}}</td>
        <td>${{t.fee_cost.toFixed(2)}}</td>
        <td>${{t.funding_cost.toFixed(2)}}</td>
      </tr>`;
    }}
    if (trades.length > maxShow) {{
      html += `<tr><td colspan="11" style="color:#8b949e;text-align:center">... and ${{trades.length - maxShow}} more trades</td></tr>`;
    }}
    html += `</tbody></table></div>`;
    html += `</div>`;

    dv.innerHTML = html;

    // Store data for chart overlay
    window.detailData = data;

    // Render equity chart
    if (detailChart) detailChart.destroy();
    const ctx = document.getElementById('detailEquityChart').getContext('2d');
    detailChart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: data.equity_labels,
        datasets: [{{
          label: 'Equity ($)',
          data: data.equity_curve,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.1)',
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
          y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#21262d' }} }}
        }}
      }}
    }});

  }} catch(e) {{
    dv.innerHTML = `<p style="color:#e74c3c">Failed to load: ${{e.message}}</p>`;
  }}
}}

// --- Candlestick Chart Overlay (Lightweight Charts) ---
function openCandleChart(data) {{
  if (!data || !data.ohlc || data.ohlc.length === 0) return;

  // Create fullscreen overlay
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
        ${{data.symbol}} · ${{data.timeframe}} · ${{data.period === 'train' ? 'Train 2021–2024' : 'Test 2025+'}}
        <span style="color:#8b949e;font-weight:normal"> | Sharpe=${{data.metrics.sharpe.toFixed(3)}} | Return=${{data.metrics.return_pct > 0 ? '+' : ''}}${{data.metrics.return_pct.toFixed(1)}}% | DD=${{data.metrics.max_dd_pct.toFixed(1)}}% | Trades=${{data.metrics.num_trades}}</span>
      </div>
      <div id="chartLegend">
        <span style="color:#2ecc71;font-size:0.75em;margin-right:10px">▲ Long Entry</span>
        <span style="color:#e74c3c;font-size:0.75em;margin-right:15px">▼ Short Entry</span>
        <button onclick="document.getElementById('chartOverlay').style.display='none'" style="background:#e74c3c;border:none;color:#fff;padding:4px 12px;border-radius:4px;cursor:pointer;font-family:monospace">Close (Esc)</button>
      </div>
    </div>
    <div id="candleChartContainer" style="flex:1;position:relative"></div>
    <div id="signalChartContainer" style="height:80px;border-top:1px solid #30363d;position:relative"></div>
  `;

  // Candlestick chart
  const container = document.getElementById('candleChartContainer');
  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {{ background: {{ color: '#0d1117' }}, textColor: '#c9d1d9' }},
    grid: {{ vertLines: {{ color: '#1c2128' }}, horzLines: {{ color: '#1c2128' }} }},
    crosshair: {{ mode: 0 }},
    timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#30363d' }},
    rightPriceScale: {{ borderColor: '#30363d' }},
  }});

  // OHLC candles
  const candleSeries = chart.addCandlestickSeries({{
    upColor: '#2ecc71', downColor: '#e74c3c',
    borderUpColor: '#2ecc71', borderDownColor: '#e74c3c',
    wickUpColor: '#2ecc71', wickDownColor: '#e74c3c',
  }});
  candleSeries.setData(data.ohlc);

  // Dynamic indicator lines
  if (data.indicators) {{
    for (const [indName, indInfo] of Object.entries(data.indicators)) {{
      if (indInfo.data && indInfo.data.length > 0) {{
        const series = chart.addLineSeries({{
          color: indInfo.color || '#f0883e',
          lineWidth: 1.5,
          lineStyle: 0,
          title: indName,
        }});
        series.setData(indInfo.data);
      }}
    }}
  }}

  // Add indicator names to legend
  if (data.indicators) {{
    const legend = document.getElementById('chartLegend');
    for (const [indName, indInfo] of Object.entries(data.indicators)) {{
      const span = document.createElement('span');
      span.style.cssText = `color:${{indInfo.color}};font-size:0.75em;margin-right:15px`;
      span.textContent = `━ ${{indName}}`;
      legend.insertBefore(span, legend.firstChild);
    }}
  }}

  // Trade markers on candles — limit density to avoid clutter
  const markers = [];
  const allTrades = data.trade_markers || [];
  let lastMarkerTime = 0;
  const minGap = 3600 * 4; // at least 4h between markers on chart
  allTrades.forEach(tm => {{
    if (tm.entry_time - lastMarkerTime >= minGap) {{
      markers.push({{
        time: tm.entry_time,
        position: tm.direction === 'LONG' ? 'belowBar' : 'aboveBar',
        color: tm.direction === 'LONG' ? '#2ecc71' : '#e74c3c',
        shape: tm.direction === 'LONG' ? 'arrowUp' : 'arrowDown',
        text: tm.direction[0] + ' $' + tm.entry_price.toLocaleString(),
      }});
      markers.push({{
        time: tm.exit_time,
        position: tm.pnl_pct >= 0 ? 'aboveBar' : 'belowBar',
        color: tm.pnl_pct >= 0 ? 'rgba(46,204,113,0.8)' : 'rgba(231,76,60,0.8)',
        shape: 'circle',
        text: (tm.pnl_pct >= 0 ? '+' : '') + tm.pnl_pct.toFixed(2) + '%',
      }});
      lastMarkerTime = tm.exit_time;
    }}
  }});
  markers.sort((a, b) => a.time - b.time);
  if (markers.length > 0) candleSeries.setMarkers(markers);

  // Signal pane
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
  const sigData = (data.signals || []).map(s => ({{
    time: s.time,
    value: s.value,
    color: s.value > 0 ? 'rgba(46,204,113,0.6)' : s.value < 0 ? 'rgba(231,76,60,0.6)' : 'rgba(139,148,158,0.15)',
  }}));
  sigSeries.setData(sigData);

  // Sync time scales
  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (range) sigChart.timeScale().setVisibleLogicalRange(range);
  }});

  // Resize handler
  const resizeObserver = new ResizeObserver(() => {{
    chart.applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
    sigChart.applyOptions({{ width: sigContainer.clientWidth, height: sigContainer.clientHeight }});
  }});
  resizeObserver.observe(container);

  // ESC to close
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
</script>
</body>
</html>"""


def _build_table_rows(df: pd.DataFrame, limit: int = 100, presorted: bool = False) -> str:
    if df.empty or "sharpe" not in df.columns:
        return '<tr><td colspan="9" class="no-data" style="padding:10px;color:#8b949e">No data</td></tr>'

    tf_cache = {}
    best = df if presorted else df.sort_values("sharpe", ascending=False).head(limit)
    rows = ""
    for _, row in best.iterrows():
        sharpe_val = float(row.get("sharpe", 0) or 0)
        color = "#2ecc71" if sharpe_val > 0 else "#e74c3c"
        status = row.get("status", "")
        badge_cls = f"badge-{status}" if status in ("keep", "discard", "crash") else "badge-discard"
        strategy = str(row.get("strategy", ""))
        symbol = str(row.get("symbol", ""))
        if strategy not in tf_cache:
            tf_cache[strategy] = get_strategy_timeframe(strategy)
        tf = tf_cache[strategy]
        rows += f"""
        <tr data-strategy="{_esc(strategy)}" data-symbol="{_esc(symbol)}" data-tf="{_esc(tf)}" data-sharpe="{sharpe_val:.4f}" onclick="openModal('{_esc(strategy)}')">
            <td>{_esc(strategy)}</td>
            <td>{_esc(symbol)}</td>
            <td>{_esc(tf)}</td>
            <td style="color:{color}">{sharpe_val:.3f}</td>
            <td>{float(row.get('return_pct', 0) or 0):+.1f}%</td>
            <td>{float(row.get('max_dd_pct', 0) or 0):.1f}%</td>
            <td>{float(row.get('win_rate', 0) or 0):.0f}%</td>
            <td>{int(row.get('trades', 0) or 0)}</td>
            <td><span class="badge {badge_cls}">{status}</span></td>
        </tr>"""
    return rows


def run_detail_backtest(strategy_name: str, symbol: str, period: str) -> dict:
    """Run backtest for a specific strategy+symbol and return detailed results + price chart data."""
    import numpy as np
    from backtest import run_strategy_backtest
    from evaluate import compute_metrics
    from prepare import load_klines, load_config
    import importlib.util

    strategy_path = STRATEGIES_DIR / f"{strategy_name}.py"
    if not strategy_path.exists():
        strategy_path = STRATEGY_FILE
        if not strategy_path.exists():
            return {"error": f"Strategy file not found: {strategy_name}"}

    try:
        result = run_strategy_backtest(
            strategy_path=str(strategy_path),
            symbol=symbol,
            period=period,
        )
        metrics = compute_metrics(result)

        # Equity curve: sample every Nth point for chart (max 500 points)
        eq = result.equity_curve
        n = len(eq)
        step = max(1, n // 500)
        eq_sampled = eq[::step].tolist()
        eq_labels = [f"{i * step}" for i in range(len(eq_sampled))]

        # --- Price data + signals for chart ---
        # Load strategy module to get timeframe and generate signals
        spec = importlib.util.spec_from_file_location("strat", str(strategy_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        timeframe = getattr(mod, "timeframe", "1h")

        config = load_config()
        prices = load_klines(symbol, timeframe)
        # Filter to period
        train_start = pd.Timestamp(config["data"]["train_start"], tz="UTC")
        train_end = pd.Timestamp(config["data"]["train_end"], tz="UTC")
        test_start = pd.Timestamp(config["data"]["test_start"], tz="UTC")
        if period == "train":
            prices = prices[(prices["open_time"] >= train_start) & (prices["open_time"] <= train_end)]
        else:
            prices = prices[prices["open_time"] >= test_start]
        prices = prices.reset_index(drop=True)

        # Generate signals for chart
        try:
            signals = mod.generate_signals(prices)
            if len(signals) != len(prices):
                signals = np.zeros(len(prices))
        except Exception:
            signals = np.zeros(len(prices))

        # OHLC data for candlestick chart (all bars, Lightweight Charts handles zoom)
        price_n = len(prices)
        # Convert timestamps to unix seconds for Lightweight Charts
        open_times = prices["open_time"].values
        # Convert to unix seconds reliably
        unix_times = [int(pd.Timestamp(t).timestamp()) for t in open_times]

        # Sample for performance: max 3000 candles (LWC handles this well)
        price_step = max(1, price_n // 3000)
        ohlc = []
        for i in range(0, price_n, price_step):
            # Aggregate candles within the step
            end = min(i + price_step, price_n)
            ohlc.append({
                "time": int(unix_times[i]),
                "open": round(float(prices["open"].values[i]), 2),
                "high": round(float(prices["high"].values[i:end].max()), 2),
                "low": round(float(prices["low"].values[i:end].min()), 2),
                "close": round(float(prices["close"].values[end - 1]), 2),
            })

        # Compute indicators based on what the strategy code uses
        close_arr = prices["close"].values
        high_arr = prices["high"].values
        low_arr = prices["low"].values
        close_s = pd.Series(close_arr)
        strategy_code = get_strategy_code(strategy_name) or ""

        indicators = {}  # name → list of {time, value}

        # Always compute EMA 21/55 (universal)
        ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
        ema55 = close_s.ewm(span=55, min_periods=55, adjust=False).mean().values

        # Detect which indicators the strategy uses and compute them
        if "hma" in strategy_code.lower() or "hull" in strategy_code.lower():
            # HMA(16) and HMA(48)
            def _wma(arr, w):
                weights = np.arange(1, w + 1, dtype=np.float64)
                weights /= weights.sum()
                out = np.full(len(arr), np.nan)
                for j in range(w - 1, len(arr)):
                    out[j] = np.dot(arr[j - w + 1:j + 1], weights)
                return out
            def _hma(arr, period):
                half = period // 2
                sqrt_n = max(1, int(np.sqrt(period)))
                w_half = _wma(arr, half)
                w_full = _wma(arr, period)
                diff = 2 * w_half - w_full
                return _wma(diff, sqrt_n)
            hma16 = _hma(close_arr, 16)
            hma48 = _hma(close_arr, 48)
            indicators["HMA 16"] = {"data": hma16, "color": "#00bfff"}
            indicators["HMA 48"] = {"data": hma48, "color": "#ff6b9d"}
        elif "supertrend" in strategy_code.lower():
            indicators["EMA 21"] = {"data": ema21, "color": "#f0883e"}
            indicators["EMA 55"] = {"data": ema55, "color": "#a371f7"}
        elif "donchian" in strategy_code.lower():
            # Donchian channels
            period = 20
            don_high = pd.Series(high_arr).rolling(period, min_periods=period).max().values
            don_low = pd.Series(low_arr).rolling(period, min_periods=period).min().values
            indicators["Donchian High"] = {"data": don_high, "color": "#2ecc71"}
            indicators["Donchian Low"] = {"data": don_low, "color": "#e74c3c"}
        if "ema" in strategy_code.lower() or not indicators:
            indicators["EMA 21"] = {"data": ema21, "color": "#f0883e"}
            indicators["EMA 55"] = {"data": ema55, "color": "#a371f7"}
        if "sma" in strategy_code.lower() and "200" in strategy_code:
            sma200 = close_s.rolling(200, min_periods=200).mean().values
            indicators["SMA 200"] = {"data": sma200, "color": "#8b949e"}

        # Build indicator data for chart
        indicator_series = {}
        for ind_name, ind_info in indicators.items():
            series = []
            for i in range(0, price_n, price_step):
                v = float(ind_info["data"][i])
                if not np.isnan(v):
                    series.append({"time": int(unix_times[i]), "value": round(v, 2)})
            indicator_series[ind_name] = {"data": series, "color": ind_info["color"]}

        # Signal data for signal pane
        signal_data = []
        for i in range(0, price_n, price_step):
            sig = float(signals[i])
            signal_data.append({"time": int(unix_times[i]), "value": round(sig, 4)})

        # Trade markers with unix timestamps
        trade_markers = []
        for t in result.trades:
            entry_ts = int(pd.Timestamp(t.entry_time).timestamp())
            exit_ts = int(pd.Timestamp(t.exit_time).timestamp())
            trade_markers.append({
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "entry_time_str": str(t.entry_time)[:16],
                "exit_time_str": str(t.exit_time)[:16],
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(float(t.entry_price), 2),
                "exit_price": round(float(t.exit_price), 2),
                "pnl_pct": round(float(t.pnl_pct) * 100, 3),
            })

        # Full trades list
        trades = []
        running_balance = float(result.equity_curve[0])
        for t in result.trades:
            running_balance += t.pnl
            trades.append({
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size": round(t.size, 4),
                "leverage": round(t.leverage, 1),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct * 100, 3),
                "balance": round(running_balance, 2),
                "fee_cost": round(t.fee_cost, 2),
                "funding_cost": round(t.funding_cost, 2),
            })

        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "period": period,
            "timeframe": result.timeframe,
            "metrics": {
                "sharpe": round(metrics["sharpe_ratio"], 4),
                "return_pct": round(metrics["total_return_pct"], 2),
                "max_dd_pct": round(metrics["max_drawdown_pct"], 2),
                "win_rate": round(metrics["win_rate"], 1),
                "num_trades": metrics["num_trades"],
                "profit_factor": round(metrics["profit_factor"], 2),
                "total_fees": round(metrics["total_fees"], 2),
                "total_funding": round(metrics["total_funding_cost"], 2),
            },
            "equity_curve": eq_sampled,
            "equity_labels": eq_labels,
            "trades": trades,
            "num_bars": n,
            # Candlestick chart data (Lightweight Charts format)
            "ohlc": ohlc,
            "indicators": indicator_series,
            "signals": signal_data,
            "trade_markers": trade_markers,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def get_strategy_data(strategy_name: str) -> dict:
    """Build modal data for a single strategy (used by /api/strategy endpoint)."""
    from results_db import load_results as _db_load
    df = _db_load()
    group = df[df["strategy"] == strategy_name]
    code = get_strategy_code(strategy_name)
    val = run_validation(code)
    rows_by_period: dict = {}
    for _, row in group.iterrows():
        period = row.get("period", "train")
        if period not in rows_by_period:
            rows_by_period[period] = []
        rows_by_period[period].append({
            "symbol": row.get("symbol", ""),
            "sharpe": round(float(row.get("sharpe", 0) or 0), 4),
            "return_pct": round(float(row.get("return_pct", 0) or 0), 2),
            "cagr_pct": round(float(row.get("cagr_pct", 0) or 0), 2),
            "max_dd_pct": round(float(row.get("max_dd_pct", 0) or 0), 2),
            "win_rate": round(float(row.get("win_rate", 0) or 0), 1),
            "profit_factor": round(float(row.get("profit_factor", 0) or 0), 2),
            "trades": int(row.get("trades", 0) or 0),
            "sortino": round(float(row.get("sortino", 0) or 0), 4),
            "calmar": round(float(row.get("calmar", 0) or 0), 4),
            "status": row.get("status", ""),
            "period": period,
        })
    train_rows = rows_by_period.get("train", [])
    avg_sharpe = sum(r["sharpe"] for r in train_rows) / len(train_rows) if train_rows else 0
    avg_dd = sum(r["max_dd_pct"] for r in train_rows) / len(train_rows) if train_rows else 0
    avg_return = sum(r["return_pct"] for r in train_rows) / len(train_rows) if train_rows else 0
    return {
        "name": strategy_name,
        "code": code,
        "valid": val.valid,
        "validation_html": build_validation_html(val),
        "rows_by_period": rows_by_period,
        "avg_sharpe": round(avg_sharpe, 4),
        "avg_dd": round(avg_dd, 2),
        "avg_return": round(avg_return, 2),
    }


def _refresh_cache():
    """Render the dashboard HTML and store in cache."""
    try:
        html = render_html()
        with _cache_lock:
            _page_cache["html"] = html
            _page_cache["ts"] = time.time()
    except Exception as e:
        print(f"[cache] render error: {e}")


def _cache_worker(interval: int = 30):
    """Background thread: refresh cache every `interval` seconds."""
    while True:
        _refresh_cache()
        time.sleep(interval)


from socketserver import ThreadingMixIn
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each HTTP request in its own thread."""
    daemon_threads = True


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self.path)
        if parsed.path == "/api/detail":
            params = parse_qs(parsed.query)
            strategy = params.get("strategy", [""])[0]
            symbol = params.get("symbol", ["BTCUSDT"])[0]
            period = params.get("period", ["train"])[0]

            result = run_detail_backtest(strategy, symbol, period)
            data = json.dumps(result, ensure_ascii=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))

        elif parsed.path == "/api/strategy":
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            result = get_strategy_data(name) if name else {"error": "missing name"}
            data = json.dumps(result, ensure_ascii=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))

        else:
            with _cache_lock:
                html = _page_cache["html"]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress access logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cache-interval", type=int, default=30,
                        help="Seconds between background cache refreshes (default: 30)")
    args = parser.parse_args()

    # Warm up cache before accepting requests
    print("Building initial cache…")
    _refresh_cache()

    # Background refresh thread
    t = threading.Thread(target=_cache_worker, args=(args.cache_interval,), daemon=True)
    t.start()
    print(f"Cache refresh every {args.cache_interval}s (background thread)")

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Auto-refreshes every 10 minutes. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
