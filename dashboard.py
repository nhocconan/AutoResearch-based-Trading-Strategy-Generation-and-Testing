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
                             query_chart_data, query_distinct_symbols,
                             query_recent_experiments)

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
    best_sharpe     = float(stats.get("best_sharpe", 0) or 0)
    train_total     = int(stats.get("train_total", 0) or 0)
    train_kept      = int(stats.get("train_kept", 0) or 0)
    train_best      = float(stats.get("train_best", 0) or 0)
    test_total      = int(stats.get("test_total", 0) or 0)
    test_kept       = int(stats.get("test_kept", 0) or 0)
    test_best       = float(stats.get("test_best", 0) or 0)
    best_kept_test  = float(stats.get("best_kept_test", 0) or 0)

    # Recent experiments for Overview pane
    recent_df = query_recent_experiments(12)
    recent_rows_html = _build_recent_rows(recent_df)

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

    # Chart data: Sharpe over time for BTCUSDT train (downsampled for performance)
    chart_data, chart_labels, running_best_data = "[]", "[]", "[]"
    if train_total > 0:
        sharpes = query_chart_data("BTCUSDT", "train")
        if sharpes:
            # Downsample to max 500 points for chart performance
            n = len(sharpes)
            max_pts = 500
            if n > max_pts:
                step = n / max_pts
                indices = [int(i * step) for i in range(max_pts - 1)] + [n - 1]
                sharpes_ds = [sharpes[i] for i in indices]
            else:
                indices = list(range(n))
                sharpes_ds = sharpes
            labels = [f"#{i+1}" for i in indices]
            running_best = []
            # Compute running best on full data, then downsample
            best_so_far = -999.0
            full_best = []
            for s in sharpes:
                best_so_far = max(best_so_far, s)
                full_best.append(best_so_far)
            running_best = [round(full_best[i], 4) for i in indices]
            chart_data = json.dumps([round(s, 4) for s in sharpes_ds])
            chart_labels = json.dumps(labels)
            running_best_data = json.dumps(running_best)

    # Concept coverage stats (for Concepts tab)
    from results_db import query_concept_stats
    concept_stats = query_concept_stats()
    concept_stats_json = json.dumps(concept_stats)

    from results_db import query_scatter_data
    scatter_data = query_scatter_data(400)
    scatter_data_json = json.dumps(scatter_data)

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
            '  <span id="test-ind-badge" style="display:none;background:#1f4e9e;border:1px solid #58a6ff;border-radius:4px;padding:2px 8px;font-size:0.75em;color:#79c0ff;cursor:pointer" onclick="clearIndicatorFilter(\'test\')" title="Click to clear indicator filter">indicator: <b id="test-ind-name"></b> ✕</span>\n'
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
<html data-theme="dark" lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Trading Research</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {{
      --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
      --border: #30363d; --text: #c9d1d9; --muted: #8b949e;
      --accent: #58a6ff; --accent2: #79c0ff;
      --green: #2ecc71; --red: #e74c3c; --orange: #f0883e; --yellow: #f1c40f;
    }}
    [data-theme="light"] {{
      --bg: #f6f8fa; --surface: #ffffff; --surface2: #eaeef2;
      --border: #d0d7de; --text: #1f2328; --muted: #57606a;
      --accent: #0969da; --accent2: #0550ae;
      --green: #1a7f37; --red: #cf222e; --orange: #bc4c00; --yellow: #9a6700;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Courier New', monospace; background: var(--bg); color: var(--text); margin: 0; padding: 0; }}
    h2 {{ color: var(--accent2); margin-top: 24px; margin-bottom: 8px; }}
    h3 {{ color: var(--accent2); margin: 10px 0; font-size: 0.95em; }}

    /* Header */
    .app-header {{
      position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between;
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 10px 20px;
    }}
    .app-brand {{ color: var(--accent); font-size: 1.1em; font-weight: bold; }}
    .header-actions {{ display: flex; gap: 10px; align-items: center; }}
    .ts {{ color: var(--muted); font-size: 0.75em; }}
    .icon-btn {{
      background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 4px 10px; cursor: pointer; font-size: 0.9em;
      font-family: 'Courier New', monospace;
    }}
    .icon-btn:hover {{ border-color: var(--accent); color: var(--accent); }}

    /* Stats bar */
    .stats-bar {{
      display: flex; gap: 12px; flex-wrap: wrap;
      padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border);
    }}
    .stat-card {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 18px; min-width: 100px;
    }}
    .stat-card .value {{ font-size: 1.6em; font-weight: bold; color: var(--accent); }}
    .stat-card .label {{ font-size: 0.75em; color: var(--muted); margin-top: 2px; }}

    /* Main nav */
    .main-nav {{
      display: flex; gap: 0; padding: 0 20px;
      background: var(--surface); border-bottom: 2px solid var(--border);
    }}
    .nav-tab {{
      background: none; border: none; border-bottom: 3px solid transparent;
      color: var(--muted); padding: 10px 18px; cursor: pointer;
      font-family: 'Courier New', monospace; font-size: 0.85em;
      transition: color 0.15s;
    }}
    .nav-tab:hover {{ color: var(--text); }}
    .nav-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

    /* Global filters in nav */
    .nav-global-filters {{
      display: flex; align-items: center; gap: 6px;
      margin-left: auto; padding: 4px 12px 4px 16px;
      border-left: 1px solid var(--border);
    }}
    .ngf-label {{ font-size: 0.75em; color: var(--muted); white-space: nowrap; }}
    .ngf-input {{ width: 58px !important; padding: 2px 6px !important; font-size: 0.75em !important; }}
    .ngf-clear {{ padding: 2px 8px !important; font-size: 0.75em !important; }}
    .ngf-info {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 16px; height: 16px; border-radius: 50%;
      background: var(--surface2); border: 1px solid var(--border);
      color: var(--muted); font-size: 0.68em; cursor: default;
      position: relative;
    }}
    .ngf-info:hover {{ border-color: var(--accent); color: var(--accent); }}
    .ngf-tooltip {{
      display: none; position: absolute; bottom: calc(100% + 8px); right: 0;
      background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
      padding: 10px 14px; width: 280px; font-size: 0.8em; color: var(--text);
      box-shadow: 0 4px 16px rgba(0,0,0,0.4); z-index: 200;
      white-space: normal; line-height: 1.5;
    }}
    .ngf-info:hover .ngf-tooltip {{ display: block; }}
    .ngf-tooltip b {{ color: var(--accent); }}
    .ngf-tooltip .tt-row {{ margin: 4px 0; }}

    /* Panes */
    .main-pane {{ display: none; padding: 20px; }}
    .main-pane.active {{ display: block; }}

    /* Period grid */
    .period-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
    .period-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px; }}
    .period-box.train {{ border-color: #1f6feb; }}
    .period-box.test {{ border-color: #238636; }}
    .period-label {{ font-size: 0.75em; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }}
    .period-box.train .period-label {{ color: var(--accent); }}
    .period-box.test .period-label {{ color: var(--green); }}
    .mini-stats {{ display: flex; gap: 15px; margin-bottom: 10px; }}
    .mini-stat {{ text-align: center; }}
    .mini-stat .v {{ font-size: 1.4em; font-weight: bold; }}
    .mini-stat .l {{ font-size: 0.72em; color: var(--muted); }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }}
    th {{ background: var(--surface2); padding: 8px; text-align: left; color: var(--muted); font-size: 0.8em; cursor: pointer; user-select: none; }}
    th:hover {{ color: var(--accent); }}
    th.sorted-asc::after {{ content: ' ▲'; color: var(--accent); }}
    th.sorted-desc::after {{ content: ' ▼'; color: var(--accent); }}
    td {{ padding: 6px 8px; border-top: 1px solid var(--border); font-size: 0.8em; }}
    tr[data-strategy]:hover td {{ background: var(--surface2); cursor: pointer; }}

    /* Chart */
    .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin: 20px 0; height: 280px; }}

    /* Pre / code */
    pre {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px; overflow: auto; max-height: 400px; font-size: 0.78em; color: var(--text); }}
    .git-log {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px; max-height: 180px; overflow: auto; font-size: 0.78em; }}

    /* Badges */
    .badge {{ padding: 2px 7px; border-radius: 4px; font-size: 0.72em; font-weight: bold; }}
    .badge-keep {{ background: #1a4731; color: var(--green); }}
    .badge-discard {{ background: #3d1f1f; color: var(--red); }}
    .badge-crash {{ background: #3d2f10; color: var(--orange); }}
    .badge-pass {{ background: #1a4731; color: var(--green); }}
    .badge-fail {{ background: #3d1f1f; color: var(--red); }}

    /* Compliance */
    .compliance-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin: 15px 0; font-size: 0.82em; }}
    .val-header {{ font-weight: bold; margin-bottom: 8px; }}
    .val-section {{ color: var(--muted); margin: 6px 0 3px; font-size: 0.9em; }}
    .val-error {{ color: var(--red); padding: 2px 0; }}
    .val-warn {{ color: var(--orange); padding: 2px 0; }}
    .val-info {{ color: var(--muted); padding: 2px 0; }}

    /* Modal */
    .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.75); z-index: 1000; overflow-y: auto; }}
    .modal {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; max-width: 900px; margin: 40px auto; padding: 25px; position: relative; }}
    .modal-close {{ position: absolute; top: 15px; right: 20px; background: none; border: none; color: var(--muted); font-size: 1.4em; cursor: pointer; line-height: 1; }}
    .modal-close:hover {{ color: var(--text); }}
    .modal h2 {{ margin-top: 0; color: var(--accent); }}
    .modal-tabs {{ display: flex; gap: 0; margin: 15px 0; border-bottom: 1px solid var(--border); }}
    .tab-btn {{ background: none; border: none; color: var(--muted); padding: 8px 16px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.85em; border-bottom: 2px solid transparent; }}
    .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
    .metrics-table th {{ background: var(--surface2); padding: 6px 10px; text-align: left; color: var(--muted); }}
    .metrics-table td {{ padding: 5px 10px; border-top: 1px solid var(--border); }}
    .no-data {{ color: var(--muted); font-style: italic; padding: 10px 0; }}
    .detail-btn {{ background: var(--surface2); border: 1px solid var(--accent); border-radius: 4px; color: var(--accent); padding: 2px 8px; cursor: pointer; font-family: 'Courier New',monospace; font-size: 0.72em; white-space: nowrap; }}
    .detail-btn:hover {{ background: #1f6feb; color: #fff; }}
    .detail-btn.loading {{ opacity: 0.5; cursor: wait; }}
    .detail-panel {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin-top: 10px; }}
    .detail-panel h4 {{ color: var(--accent); margin: 0 0 10px; }}
    .detail-chart {{ height: 220px; margin: 10px 0; }}
    .trade-list {{ max-height: 350px; overflow-y: auto; font-size: 0.75em; }}
    .trade-list table {{ width: 100%; }}
    .trade-list th {{ position: sticky; top: 0; background: var(--surface2); z-index: 1; }}
    .trade-long {{ color: var(--green); }}
    .trade-short {{ color: var(--red); }}
    .pnl-pos {{ color: var(--green); }}
    .pnl-neg {{ color: var(--red); }}

    /* Filter bar */
    .filter-bar {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }}
    .filter-bar label {{ color: var(--muted); font-size: 0.78em; }}
    .filter-btn {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; color: var(--muted); padding: 4px 12px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.78em; }}
    .filter-btn:hover {{ border-color: var(--accent); color: var(--text); }}
    .filter-btn.active {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}
    .filter-info {{ color: var(--muted); font-size: 0.72em; margin-left: 8px; }}
    .filter-input {{
      background: var(--surface2); border: 1px solid var(--border); border-radius: 4px;
      color: var(--text); padding: 3px 7px; width: 70px;
      font-family: 'Courier New', monospace; font-size: 0.78em;
    }}
    .filter-input:focus {{ border-color: var(--accent); outline: none; }}
    .filter-input::placeholder {{ color: var(--muted); }}

    /* Rank bar */
    .rank-bar {{ display: flex; gap: 8px; align-items: center; margin: 8px 0; flex-wrap: wrap; }}
    .rank-btn {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; color: var(--muted); padding: 4px 12px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.78em; }}
    .rank-btn:hover {{ border-color: var(--accent); color: var(--text); }}
    .rank-btn.active {{ background: #238636; border-color: #238636; color: #fff; }}

    /* Concepts tab */
    .concept-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
    .concept-card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 16px; min-width: 130px; font-size: 0.82em;
    }}
    .concept-card.concept-untested {{ opacity: 0.45; }}
    .cc-tf {{ font-size: 1.15em; font-weight: bold; color: var(--accent); margin-bottom: 6px; }}
    .cc-stat {{ margin: 3px 0; }}
    #indicator-table {{ margin-top: 12px; }}
    .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
    .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
    .chart-sub {{ font-size: 0.72em; color: var(--muted); font-weight: normal; }}
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap-table {{ border-collapse: collapse; font-size: 0.72em; white-space: nowrap; }}
    .heatmap-table th {{ background: var(--surface2); padding: 5px 8px; color: var(--muted); text-align: center; }}
    .heatmap-label {{ padding: 4px 8px; color: var(--text); font-weight: bold; background: var(--surface2); }}
    .heatmap-cell {{ padding: 4px 7px; text-align: center; cursor: default; font-size: 0.9em; min-width: 42px; }}
    .heatmap-na {{ color: var(--border); }}
    @media (max-width: 768px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}

    /* Concept results inline panel */
    .concept-results-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    }}
    #concept-results-panel table {{ font-size: 0.78em; }}
    #concept-results-panel td {{ padding: 4px 6px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    #concept-results-panel th {{ font-size: 0.78em; padding: 5px 6px; }}
    @media (max-width: 900px) {{ .concept-results-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>

<header class="app-header">
  <div class="app-brand">⚡ LLM Trading Research</div>
  <div class="header-actions">
    <span class="ts">Updated: {now}</span>
    <button class="icon-btn" onclick="location.reload()" title="Refresh page">⟳</button>
    <button class="icon-btn" id="themeBtn" onclick="toggleTheme()" title="Toggle dark/light mode">🌙</button>
  </div>
</header>

<!-- Always-visible stats bar -->
<div class="stats-bar">
  <div class="stat-card"><div class="value">{total}</div><div class="label">Total Rows</div></div>
  <div class="stat-card"><div class="value" style="color:var(--green)">{kept}</div><div class="label">Kept</div></div>
  <div class="stat-card"><div class="value" style="color:var(--red)">{discarded}</div><div class="label">Discarded</div></div>
  <div class="stat-card"><div class="value" style="color:var(--orange)">{crashed}</div><div class="label">Crashed</div></div>
  <div class="stat-card"><div class="value" style="color:var(--green)">{best_kept_test:.3f}</div><div class="label">Best Kept (Test)</div></div>
</div>

<!-- Main navigation tabs + global Train/Test filters -->
<nav class="main-nav">
  <button class="nav-tab active" id="navbtn-overview" onclick="switchMainTab('overview')">📊 Overview</button>
  <button class="nav-tab" id="navbtn-train" onclick="switchMainTab('train')">🔵 Train Results</button>
  <button class="nav-tab" id="navbtn-test" onclick="switchMainTab('test')">🧪 Test Results</button>
  <button class="nav-tab" id="navbtn-concepts" onclick="switchMainTab('concepts')">💡 Concepts</button>
  <button class="nav-tab" id="navbtn-log" onclick="switchMainTab('log')">📝 Research Log</button>
  <div class="nav-global-filters">
    <span class="ngf-label">Train &amp; Test:</span>
    <label class="ngf-label">WR ≥</label><input type="number" class="filter-input ngf-input" id="global-min-winrate" placeholder="%" min="0" max="100" title="Min Win Rate % (Train &amp; Test)" oninput="applyGlobalFilters()">
    <label class="ngf-label">Ret ≥</label><input type="number" class="filter-input ngf-input" id="global-min-return" placeholder="%" title="Min Return % (Train &amp; Test)" oninput="applyGlobalFilters()">
    <label class="ngf-label">DD ≥</label><input type="number" class="filter-input ngf-input" id="global-min-dd" placeholder="%" max="0" step="1" title="Max Drawdown floor, e.g. -30 hides worse than -30%" oninput="applyGlobalFilters()">
    <label class="ngf-label">Trades ≥</label><input type="number" class="filter-input ngf-input" id="global-min-trades" placeholder="#" min="0" step="1" title="Min number of trades" oninput="applyGlobalFilters()">
    <button class="filter-btn ngf-clear" onclick="clearGlobalFilters()" title="Clear all metric filters">✕</button>
    <span id="global-filter-info" class="ngf-label" style="color:var(--orange);min-width:60px"></span>
    <span class="ngf-info">?<span class="ngf-tooltip">
      <b>Global filters — affect both Train &amp; Test tabs</b><br>
      <div class="tt-row"><b>WR ≥</b> &nbsp;Min Win Rate %. e.g. <b>50</b> → hide rows with win rate below 50%</div>
      <div class="tt-row"><b>Ret ≥</b> &nbsp;Min Return %. e.g. <b>20</b> → hide rows with return below 20%</div>
      <div class="tt-row"><b>DD ≥</b> &nbsp;Max Drawdown floor (negative). e.g. <b>-30</b> → hide rows with drawdown worse than -30%</div>
      <div class="tt-row"><b>Trades ≥</b> &nbsp;Min number of trades. e.g. <b>20</b> → hide rows with fewer than 20 trades</div>
      <div class="tt-row" style="color:var(--muted);margin-top:6px">All filters are AND — only rows passing every condition are shown. ✕ clears all.</div>
    </span></span>
  </div>
</nav>

<!-- Overview pane -->
<div id="pane-overview" class="main-pane active">
  <h2>Period Summary</h2>
  <div class="period-grid">
    <div class="period-box train">
      <div class="period-label">Train Period (2021–2024)</div>
      <div class="mini-stats">
        <div class="mini-stat"><div class="v">{train_total}</div><div class="l">Results</div></div>
        <div class="mini-stat"><div class="v" style="color:var(--green)">{train_kept}</div><div class="l">Kept</div></div>
        <div class="mini-stat"><div class="v" style="color:var(--accent)">{train_best:.3f}</div><div class="l">Best Sharpe</div></div>
      </div>
    </div>
    <div class="period-box test">
      <div class="period-label">Test Period (2025+)</div>
      <div class="mini-stats">
        <div class="mini-stat"><div class="v">{test_total}</div><div class="l">Results</div></div>
        <div class="mini-stat"><div class="v" style="color:var(--green)">{test_kept}</div><div class="l">Kept</div></div>
        <div class="mini-stat"><div class="v" style="color:var(--accent)">{f"{test_best:.3f}" if test_total > 0 else "—"}</div><div class="l">Best Sharpe</div></div>
      </div>
    </div>
  </div>

  <h2>Recent Experiments <span style="font-size:0.7em;color:var(--muted)">(last 12 strategies tried)</span></h2>
  <table id="recent-table" style="width:100%;border-collapse:collapse;font-size:0.85em">
    <thead><tr>
      <th style="text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Strategy</th>
      <th style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Best Sharpe</th>
      <th style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Avg Sharpe</th>
      <th style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Avg WR</th>
      <th style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Avg DD</th>
      <th style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Trades</th>
      <th style="text-align:center;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted)">Status</th>
    </tr></thead>
    <tbody>{recent_rows_html}</tbody>
  </table>

  <h2>Sharpe Progress (BTCUSDT Train)</h2>
  <div class="chart-container">
    <canvas id="sharpeChart"></canvas>
  </div>

  <h2>Current Strategy Compliance</h2>
  <div class="compliance-box">
    <div style="margin-bottom:8px"><span class="badge {current_val_badge}">{current_val_label}</span> <b>strategy.py</b></div>
    {current_val_html}
  </div>
</div>

<!-- Train Results pane -->
<div id="pane-train" class="main-pane">
  <h2>Train Results <span style="font-size:0.7em;color:var(--muted)">(click row for details)</span></h2>
  <div class="filter-bar" id="train-filter-bar">
    <label>Symbol:</label>
    <button class="filter-btn active" onclick="filterTable('train', 'ALL')">All</button>
    {''.join(f'<button class="filter-btn" onclick="filterTable(&#39;train&#39;, &#39;{s}&#39;)">{s}</button>' for s in symbols)}
    <button class="filter-btn" onclick="filterTable('train', 'AVG')" style="border-color:var(--orange);color:var(--orange)">Avg All</button>
    <span style="margin:0 8px;color:var(--border)">|</span>
    <label>TF:</label>
    <button class="filter-btn active" onclick="filterTF('train', 'ALL')">All</button>
    {''.join(f'<button class="filter-btn" onclick="filterTF(&#39;train&#39;, &#39;{tf}&#39;)">{tf}</button>' for tf in timeframes)}
    <span style="margin:0 8px;color:var(--border)">|</span>
    <label>Status:</label>
    <button class="filter-btn active" onclick="filterStatus('train', 'ALL')">All</button>
    <button class="filter-btn" onclick="filterStatus('train', 'keep')" style="border-color:var(--green);color:var(--green)">Keep</button>
    <button class="filter-btn" onclick="filterStatus('train', 'discard')" style="border-color:var(--red);color:var(--red)">Discard</button>
    <span id="train-ind-badge" style="display:none;background:#1f4e9e;border:1px solid #58a6ff;border-radius:4px;padding:2px 8px;font-size:0.75em;color:#79c0ff;cursor:pointer" onclick="clearIndicatorFilter('train')" title="Click to clear indicator filter">indicator: <b id="train-ind-name"></b> ✕</span>
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
</div>

<!-- Test Results pane -->
<div id="pane-test" class="main-pane">
  <h2>Test Results (2025+) <span style="font-size:0.7em;color:var(--muted)">(click row for details)</span></h2>
  {test_section}
</div>

<!-- Concepts pane -->
<div id="pane-concepts" class="main-pane">
  <h2>Strategy Concepts Coverage</h2>
  <p style="color:var(--muted);font-size:0.85em">Analysis of which indicators and timeframes have been explored. Stats based on train-period results.</p>

  <h3>By Timeframe</h3>
  <div id="tf-grid" class="concept-grid"></div>

  <!-- Charts row -->
  <div class="charts-row">
    <div class="chart-box">
      <h3>Indicator &times; Timeframe Heatmap <span class="chart-sub">(best Sharpe, any symbol)</span></h3>
      <div id="heatmap-container" class="heatmap-wrap"></div>
    </div>
    <div class="chart-box">
      <h3>Kept Distribution <span class="chart-sub">(by timeframe)</span></h3>
      <div style="height:220px"><canvas id="keptTfChart"></canvas></div>
    </div>
  </div>
  <div class="chart-box" style="margin-top:16px">
    <h3>Train vs Test Sharpe <span class="chart-sub">(each dot = one strategy &mdash; points near diagonal = good generalization)</span></h3>
    <div style="height:320px"><canvas id="scatterChart"></canvas></div>
  </div>

  <h3>By Indicator / Technique</h3>
  <p style="color:var(--muted);font-size:0.8em">Click any row to filter Train &amp; Test results for that indicator.</p>
  <table id="indicator-table">
    <thead><tr>
      <th>Indicator</th><th>Strategies Tested</th><th>Kept</th>
      <th>Keep Rate</th><th>Best Sharpe (best symbol)</th><th>Coverage</th>
    </tr></thead>
    <tbody id="indicator-tbody"></tbody>
  </table>

  <!-- Inline concept results panel -->
  <div id="concept-results-panel" style="display:none;margin-top:20px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
      <h3 style="margin:0;color:var(--accent2)">Results for: <span id="concept-sel-name" style="color:var(--accent)"></span></h3>
      <div style="display:flex;gap:8px">
        <button class="filter-btn" onclick="switchMainTab('train')" title="View full Train Results tab">Open Train Tab</button>
        <button class="filter-btn" onclick="switchMainTab('test')" title="View full Test Results tab">Open Test Tab</button>
        <button class="filter-btn" onclick="clearConceptPanel()" style="border-color:var(--red);color:var(--red)">Clear ✕</button>
      </div>
    </div>
    <div class="concept-results-grid">
      <div>
        <h3 style="font-size:0.85em;margin:0 0 6px;color:var(--accent)">📊 Train Results (top matches)</h3>
        <table>
          <thead><tr><th>Strategy</th><th>Symbol</th><th>TF</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Status</th></tr></thead>
          <tbody id="concept-train-rows"></tbody>
        </table>
      </div>
      <div>
        <h3 style="font-size:0.85em;margin:0 0 6px;color:var(--green)">🧪 Test Results (top matches)</h3>
        <table>
          <thead><tr><th>Strategy</th><th>Symbol</th><th>TF</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Status</th></tr></thead>
          <tbody id="concept-test-rows"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- Research Log pane -->
<div id="pane-log" class="main-pane">
  <h2>Recent Git Commits</h2>
  <div class="git-log"><pre>{git_log}</pre></div>

  <h2>Current strategy.py</h2>
  <pre>{_esc(current_strategy[:3000])}{"..." if len(current_strategy) > 3000 else ""}</pre>
</div>

<!-- Strategy Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModalIfBackground(event)">
  <div class="modal" id="modalContent">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <h2 id="modalTitle">Strategy</h2>
    <div class="modal-tabs">
      <button class="tab-btn active" onclick="switchModalTab('metrics')">Metrics</button>
      <button class="tab-btn" onclick="switchModalTab('compliance')">Compliance</button>
      <button class="tab-btn" onclick="switchModalTab('code')">Code</button>
    </div>
    <div id="tab-metrics" class="tab-content active"></div>
    <div id="tab-compliance" class="tab-content"></div>
    <div id="tab-code" class="tab-content"></div>
  </div>
</div>

<script>
// --- Theme toggle ---
(function() {{
  const saved = localStorage.getItem('theme');
  if (saved === 'light') {{
    document.documentElement.dataset.theme = 'light';
    document.addEventListener('DOMContentLoaded', () => {{
      const btn = document.getElementById('themeBtn');
      if (btn) btn.textContent = '☀️';
    }});
  }}
}})();

function toggleTheme() {{
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.dataset.theme === 'light') {{
    html.dataset.theme = 'dark';
    localStorage.setItem('theme', 'dark');
    if (btn) btn.textContent = '🌙';
  }} else {{
    html.dataset.theme = 'light';
    localStorage.setItem('theme', 'light');
    if (btn) btn.textContent = '☀️';
  }}
}}

// --- Main tab switching ---
function switchMainTab(name) {{
  document.querySelectorAll('.main-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  const pane = document.getElementById('pane-' + name);
  const btn = document.getElementById('navbtn-' + name);
  if (pane) pane.classList.add('active');
  if (btn) btn.classList.add('active');
  // Re-apply all filters whenever switching to train or test, so global metric
  // filters are always in effect regardless of which tab was active when set.
  if (name === 'train' || name === 'test') applyFilters(name);
}}

// --- Concepts data + rendering ---
const CONCEPT_STATS = {concept_stats_json};
const SCATTER_DATA = {scatter_data_json};

function renderConcepts() {{
  const cs = CONCEPT_STATS;

  // TF grid
  const tfGrid = document.getElementById('tf-grid');
  if (tfGrid && cs.timeframes) {{
    const TF_ORDER = ['15m','30m','1h','4h','6h','12h','1d'];
    let html = '';
    TF_ORDER.forEach(tf => {{
      const d = cs.timeframes[tf];
      if (!d) {{
        html += `<div class="concept-card concept-untested"><div class="cc-tf">${{tf}}</div><div class="cc-stat" style="color:var(--muted)">not tested</div></div>`;
        return;
      }}
      const keepRate = d.total > 0 ? (d.kept / d.total * 100) : 0;
      const best = d.best !== null ? d.best.toFixed(3) : '—';
      const rateColor = keepRate > 15 ? 'var(--green)' : keepRate > 5 ? 'var(--orange)' : 'var(--red)';
      html += `<div class="concept-card">
        <div class="cc-tf">${{tf}}</div>
        <div class="cc-stat"><span style="color:var(--accent)">${{d.total}}</span> tested</div>
        <div class="cc-stat"><span style="color:${{rateColor}}">${{keepRate.toFixed(1)}}%</span> kept</div>
        <div class="cc-stat">Best (1 sym): <span style="color:var(--accent)">${{best}}</span></div>
      </div>`;
    }});
    tfGrid.innerHTML = html;
  }}

  // Indicator table
  const tbody = document.getElementById('indicator-tbody');
  if (tbody && cs.indicators) {{
    const sorted = Object.entries(cs.indicators).sort((a,b) => b[1].total - a[1].total);
    const maxTotal = sorted.length > 0 ? sorted[0][1].total : 1;
    let html = '';
    sorted.forEach(([name, d]) => {{
      const keepRate = d.total > 0 ? (d.kept / d.total * 100) : 0;
      const best = d.best !== null ? d.best.toFixed(3) : '—';
      const rateColor = keepRate > 15 ? 'var(--green)' : keepRate > 5 ? 'var(--orange)' : 'var(--red)';
      const barPct = Math.round(d.total / maxTotal * 100);
      html += `<tr style="cursor:pointer" onclick="filterByIndicator('${{name}}')" title="Click to filter Train & Test results by ${{name}}">
        <td><b style="color:var(--accent)">${{name}}</b> <span style="font-size:0.72em;color:var(--muted)">⇩</span></td>
        <td>${{d.total}}</td>
        <td>${{d.kept}}</td>
        <td style="color:${{rateColor}}">${{keepRate.toFixed(1)}}%</td>
        <td style="color:var(--accent)">${{best}}</td>
        <td><div style="background:var(--accent);height:8px;border-radius:4px;width:${{barPct}}%;opacity:0.6"></div></td>
      </tr>`;
    }});
    tbody.innerHTML = html || '<tr><td colspan="6" style="color:var(--muted)">No data yet</td></tr>';
  }}

  renderHeatmap(cs.heatmap);
  renderKeptChart(cs.timeframes);
  renderScatterChart(SCATTER_DATA);
}}

function sharpeColor(val) {{
  if (val === null || val === undefined) return 'transparent';
  if (val <= 0) {{
    const t = Math.max(0, Math.min(1, (val + 2) / 2));
    return `rgba(231,76,60,${{(0.15 + 0.7*(1-t)).toFixed(2)}})`;
  }}
  const t = Math.min(1, val / 1.8);
  return `rgba(46,204,113,${{(0.15 + 0.7*t).toFixed(2)}})`;
}}

function renderHeatmap(heatmap) {{
  const container = document.getElementById('heatmap-container');
  if (!container || !heatmap) return;
  const TF_ORDER = ['5m','15m','30m','1h','4h','6h','12h','1d','1w'];
  const indicators = Object.keys(heatmap).sort();
  let html = '<table class="heatmap-table"><thead><tr><th>&#9660; Indicator \\ TF &#9654;</th>';
  TF_ORDER.forEach(tf => {{ html += `<th>${{tf}}</th>`; }});
  html += '</tr></thead><tbody>';
  indicators.forEach(ind => {{
    html += `<tr><td class="heatmap-label">${{ind}}</td>`;
    TF_ORDER.forEach(tf => {{
      const val = heatmap[ind] && heatmap[ind][tf] !== undefined ? heatmap[ind][tf] : null;
      if (val === null) {{
        html += `<td class="heatmap-cell heatmap-na" title="${{ind}} x ${{tf}}: not tested">&middot;</td>`;
      }} else {{
        const bg = sharpeColor(val);
        const textColor = val > 0.5 ? 'var(--text)' : val < 0 ? '#fff' : 'var(--text)';
        html += `<td class="heatmap-cell" style="background:${{bg}};color:${{textColor}}" title="${{ind}} x ${{tf}}: ${{val.toFixed(3)}}">${{val.toFixed(2)}}</td>`;
      }}
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  container.innerHTML = html;
}}

let _keptTfChart = null, _scatterChart = null;

function renderKeptChart(timeframes) {{
  const canvas = document.getElementById('keptTfChart');
  if (!canvas || !timeframes) return;
  if (_keptTfChart) _keptTfChart.destroy();
  const TF_ORDER = ['5m','15m','30m','1h','4h','6h','12h','1d','1w'];
  const labels = TF_ORDER.filter(tf => timeframes[tf]);
  const kept = labels.map(tf => timeframes[tf] ? timeframes[tf].kept : 0);
  const discarded = labels.map(tf => timeframes[tf] ? (timeframes[tf].total - timeframes[tf].kept) : 0);
  _keptTfChart = new Chart(canvas.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Kept', data: kept, backgroundColor: 'rgba(46,204,113,0.75)', stack: 'a' }},
        {{ label: 'Discarded', data: discarded, backgroundColor: 'rgba(139,148,158,0.3)', stack: 'a' }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: 'var(--text)', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: 'var(--muted)' }}, grid: {{ color: 'var(--border)' }} }},
        y: {{ stacked: true, ticks: {{ color: 'var(--muted)' }}, grid: {{ color: 'var(--border)' }} }}
      }}
    }}
  }});
}}

function renderScatterChart(scatterData) {{
  const canvas = document.getElementById('scatterChart');
  if (!canvas || !scatterData || scatterData.length === 0) return;
  if (_scatterChart) _scatterChart.destroy();
  const kept = scatterData.filter(d => d.k);
  const disc = scatterData.filter(d => !d.k);
  const allVals = scatterData.map(d => d.tr).concat(scatterData.map(d => d.te));
  const minV = Math.max(-3, Math.min(...allVals) - 0.2);
  const maxV = Math.min(3, Math.max(...allVals) + 0.2);
  _scatterChart = new Chart(canvas.getContext('2d'), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{
          label: 'Kept',
          data: kept.map(d => ({{ x: d.tr, y: d.te, n: d.n }})),
          backgroundColor: 'rgba(46,204,113,0.65)',
          pointRadius: 5, pointHoverRadius: 7,
        }},
        {{
          label: 'Discarded',
          data: disc.map(d => ({{ x: d.tr, y: d.te }})),
          backgroundColor: 'rgba(139,148,158,0.25)',
          pointRadius: 3,
        }},
        {{
          label: 'y = x (perfect generalization)',
          data: [{{x: minV, y: minV}}, {{x: maxV, y: maxV}}],
          type: 'line',
          borderColor: 'rgba(88,166,255,0.4)',
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: 'var(--text)', font: {{ size: 11 }} }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const d = ctx.raw;
              return d.n ? `${{d.n}}: train=${{d.x}}, test=${{d.y}}` : `train=${{d.x}}, test=${{d.y}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Train Sharpe (best symbol)', color: 'var(--muted)' }},
          ticks: {{ color: 'var(--muted)' }}, grid: {{ color: 'var(--border)' }}
        }},
        y: {{
          title: {{ display: true, text: 'Test Sharpe (best symbol)', color: 'var(--muted)' }},
          ticks: {{ color: 'var(--muted)' }}, grid: {{ color: 'var(--border)' }}
        }}
      }}
    }}
  }});
}}

// Render concepts on load
document.addEventListener('DOMContentLoaded', renderConcepts);

// --- Chart.js: Sharpe progress ---
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
  document.getElementById('tab-metrics').innerHTML = '<p style="color:var(--muted);padding:10px">Loading…</p>';
  document.getElementById('tab-compliance').innerHTML = '';
  document.getElementById('tab-code').innerHTML = '';
  document.getElementById('modalOverlay').style.display = 'block';
  switchModalTab('metrics');
  fetch('/api/strategy?name=' + encodeURIComponent(strategyName))
    .then(r => r.json())
    .then(data => {{
      STRATEGIES[strategyName] = data;
      _renderModal(strategyName, data);
    }})
    .catch(err => {{
      document.getElementById('tab-metrics').innerHTML = '<p style="color:var(--red)">Error: ' + err + '</p>';
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
  switchModalTab('metrics');

  document.getElementById('modalOverlay').style.display = 'block';
}}

function closeModal() {{
  document.getElementById('modalOverlay').style.display = 'none';
}}

function closeModalIfBackground(event) {{
  if (event.target === document.getElementById('modalOverlay')) closeModal();
}}

function switchModalTab(name) {{
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

// --- Unified filter state + apply ---
const _tblState = {{
  train: {{ symbol: 'ALL', tf: 'ALL', status: 'ALL', indicator: null }},
  test:  {{ symbol: 'ALL', tf: 'ALL', status: 'ALL', indicator: null }}
}};

// Global metric filters stored in memory — never read from DOM inside applyFilters
// so tab switching always uses the latest values.
const _gf = {{ minWR: null, minRet: null, minDD: null, minTrades: null }};

function applyFilters(tableId) {{
  const s = _tblState[tableId] || {{}};
  const tbodyRows = document.getElementById(tableId + '-tbody-rows');
  if (!tbodyRows) return;

  const rows = tbodyRows.querySelectorAll('tr[data-strategy]');
  let shown = 0;
  rows.forEach(row => {{
    const rowSym  = row.getAttribute('data-symbol')  || '';
    const rowTf   = row.getAttribute('data-tf')      || '';
    const badge   = row.querySelector('.badge');
    const rowSt   = badge ? badge.textContent.trim() : '';
    const rowWR     = parseFloat(row.getAttribute('data-winrate') || '0');
    const rowRet    = parseFloat(row.getAttribute('data-return')  || '0');
    const rowDD     = parseFloat(row.getAttribute('data-dd')      || '0');
    const rowTrades = parseFloat(row.getAttribute('data-trades')  || '0');
    const rowName   = (row.getAttribute('data-strategy') || '').toLowerCase();

    let vis = true;
    if (s.symbol !== 'ALL' && rowSym !== s.symbol)                                  vis = false;
    if (s.tf     !== 'ALL' && rowTf  !== s.tf)                                      vis = false;
    if (s.status !== 'ALL' && rowSt  !== s.status)                                  vis = false;
    if (s.indicator && !rowName.includes(s.indicator))                               vis = false;
    if (_gf.minWR     !== null && !isNaN(_gf.minWR)     && rowWR     < _gf.minWR)  vis = false;
    if (_gf.minRet    !== null && !isNaN(_gf.minRet)    && rowRet    < _gf.minRet) vis = false;
    if (_gf.minDD     !== null && !isNaN(_gf.minDD)     && rowDD     < _gf.minDD)  vis = false;
    if (_gf.minTrades !== null && !isNaN(_gf.minTrades) && rowTrades < _gf.minTrades) vis = false;

    row.style.display = vis ? '' : 'none';
    if (vis) shown++;
  }});

  const info = document.getElementById(tableId + '-filter-info');
  if (info) {{
    const total = rows.length;
    const indLabel = s.indicator ? ` · ${{s.indicator}}` : '';
    info.textContent = shown < total ? `${{shown}}/${{total}}${{indLabel}}` : (s.indicator ? indLabel.trim() : '');
  }}
}}

function applyGlobalFilters() {{
  // Read DOM inputs → store in _gf → then apply to both tables
  const wrEl     = document.getElementById('global-min-winrate');
  const retEl    = document.getElementById('global-min-return');
  const ddEl     = document.getElementById('global-min-dd');
  const trEl     = document.getElementById('global-min-trades');
  _gf.minWR     = wrEl  && wrEl.value  !== '' ? parseFloat(wrEl.value)  : null;
  _gf.minRet    = retEl && retEl.value !== '' ? parseFloat(retEl.value) : null;
  _gf.minDD     = ddEl  && ddEl.value  !== '' ? parseFloat(ddEl.value)  : null;
  _gf.minTrades = trEl  && trEl.value  !== '' ? parseFloat(trEl.value)  : null;

  applyFilters('train');
  applyFilters('test');

  const active = [_gf.minWR, _gf.minRet, _gf.minDD, _gf.minTrades].filter(v => v !== null).length;
  const info = document.getElementById('global-filter-info');
  if (info) info.textContent = active > 0 ? `${{active}} filter${{active>1?'s':''}} active` : '';
}}

function clearGlobalFilters() {{
  ['global-min-winrate','global-min-return','global-min-dd','global-min-trades'].forEach(id => {{
    const el = document.getElementById(id);
    if (el) el.value = '';
  }});
  _gf.minWR = null; _gf.minRet = null; _gf.minDD = null; _gf.minTrades = null;
  applyFilters('train');
  applyFilters('test');
  const info = document.getElementById('global-filter-info');
  if (info) info.textContent = '';
}}

// --- Symbol filter ---
function filterTable(tableId, symbol) {{
  const tbodyRows = document.getElementById(tableId + '-tbody-rows');
  const tbodyAvg  = document.getElementById(tableId + '-tbody-avg');
  if (!tbodyRows) return;

  if (symbol === 'AVG') {{
    tbodyRows.style.display = 'none';
    if (tbodyAvg) tbodyAvg.style.display = '';
    const info = document.getElementById(tableId + '-filter-info');
    if (info) info.textContent = 'Average per strategy across all symbols';
  }} else {{
    // Show rows tbody, hide avg tbody — BEFORE filtering so applyFilters sees visible tbody
    if (tbodyAvg) tbodyAvg.style.display = 'none';
    tbodyRows.style.display = '';
    if (!_tblState[tableId]) _tblState[tableId] = {{ symbol:'ALL', tf:'ALL', status:'ALL', indicator:null }};
    _tblState[tableId].symbol = symbol;
    applyFilters(tableId);  // re-applies ALL active filters (symbol + TF + status + global metrics)
  }}

  // Update symbol button states
  const filterBar = document.getElementById(tableId + '-filter-bar');
  if (filterBar) {{
    let inSymSection = true;
    filterBar.querySelectorAll('.filter-btn').forEach(btn => {{
      const prev = btn.previousElementSibling;
      if (prev && prev.tagName === 'LABEL' && prev.textContent.trim() === 'TF:') inSymSection = false;
      if (!inSymSection) return;
      const t = btn.textContent.trim();
      const mapped = t === 'All' ? 'ALL' : t === 'Avg All' ? 'AVG' : t;
      btn.classList.toggle('active', mapped === symbol);
    }});
  }}
}}

// --- Status filter ---
function filterStatus(tableId, status) {{
  if (!_tblState[tableId]) _tblState[tableId] = {{ symbol:'ALL', tf:'ALL', status:'ALL', indicator:null }};
  _tblState[tableId].status = status;
  applyFilters(tableId);
  // Update status button states (last section of filter bar)
  const filterBar = document.getElementById(tableId + '-filter-bar');
  if (filterBar) {{
    let inStSection = false;
    filterBar.querySelectorAll('.filter-btn').forEach(btn => {{
      const prev = btn.previousElementSibling;
      if (prev && prev.tagName === 'LABEL' && prev.textContent.trim() === 'Status:') inStSection = true;
      if (!inStSection) return;
      const t = btn.textContent.trim();
      btn.classList.toggle('active', t === 'All' ? status === 'ALL' : t.toLowerCase() === status.toLowerCase());
    }});
  }}
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
  if (!_tblState[tableId]) _tblState[tableId] = {{ symbol:'ALL', tf:'ALL', status:'ALL', indicator:null }};
  _tblState[tableId].tf = tf;
  applyFilters(tableId);
  // Update TF button states
  const filterBar = document.getElementById(tableId + '-filter-bar');
  if (filterBar) {{
    let inTfSection = false;
    filterBar.querySelectorAll('.filter-btn').forEach(btn => {{
      const prev = btn.previousElementSibling;
      if (prev && prev.tagName === 'LABEL' && prev.textContent.trim() === 'TF:') inTfSection = true;
      if (prev && prev.tagName === 'LABEL' && prev.textContent.trim() === 'Status:') inTfSection = false;
      if (!inTfSection) return;
      const t = btn.textContent.trim();
      btn.classList.toggle('active', t === 'All' ? tf === 'ALL' : t === tf);
    }});
  }}
}}

// --- Filter by indicator (from Concepts tab) ---
const _IND_SEARCH = {{
  'cRSI': 'crsi', 'HMA': 'hma', 'RSI': 'rsi', 'Donchian': 'donchian',
  'KAMA': 'kama', 'Fisher': 'fisher', 'Bollinger': 'bb', 'Keltner': 'keltner',
  'ADX': 'adx', 'Chop': 'chop', 'ATR': 'atr', 'Volume': 'vol',
  'Funding': 'funding', 'Regime': 'regime', 'Pullback': 'pullback',
  'STC': 'stc', 'Ichimoku': 'ichimoku', 'Pivot': 'pivot',
}};

function filterByIndicator(indName) {{
  const searchTerm = (_IND_SEARCH[indName] || indName).toLowerCase();

  // Apply indicator filter to BOTH train and test background tables
  for (const period of ['train', 'test']) {{
    if (!_tblState[period]) _tblState[period] = {{ symbol:'ALL', tf:'ALL', status:'ALL', indicator:null }};
    _tblState[period].indicator = searchTerm;
    const tbodyRows = document.getElementById(period + '-tbody-rows');
    const tbodyAvg  = document.getElementById(period + '-tbody-avg');
    if (tbodyRows) tbodyRows.style.display = '';
    if (tbodyAvg)  tbodyAvg.style.display  = 'none';
    applyFilters(period);
    const badge = document.getElementById(period + '-ind-badge');
    const label = document.getElementById(period + '-ind-name');
    if (badge) badge.style.display = '';
    if (label) label.textContent = indName;
  }}

  // Show inline results panel in Concepts tab (don't navigate away)
  showConceptResults(indName, searchTerm);
}}

function showConceptResults(indName, searchTerm) {{
  const panel = document.getElementById('concept-results-panel');
  if (!panel) return;

  const nameEl = document.getElementById('concept-sel-name');
  if (nameEl) nameEl.textContent = indName;

  for (const period of ['train', 'test']) {{
    const srcTbody = document.getElementById(period + '-tbody-rows');
    const dstTbody = document.getElementById('concept-' + period + '-rows');
    if (!srcTbody || !dstTbody) continue;

    let html = '';
    let count = 0;
    srcTbody.querySelectorAll('tr[data-strategy]').forEach(row => {{
      if (count >= 30) return;
      const rowName = (row.getAttribute('data-strategy') || '').toLowerCase();
      if (!rowName.includes(searchTerm)) return;
      count++;

      const strat  = row.getAttribute('data-strategy') || '';
      const sym    = row.getAttribute('data-symbol') || '';
      const tf     = row.getAttribute('data-tf') || '';
      const sharpe = parseFloat(row.getAttribute('data-sharpe') || '0');
      const ret    = parseFloat(row.getAttribute('data-return') || '0');
      const dd     = parseFloat(row.getAttribute('data-dd') || '0');
      const badge  = row.querySelector('.badge');
      const status = badge ? badge.textContent.trim() : '';
      const statusCls  = status === 'keep' ? 'badge-keep' : 'badge-discard';
      const sharpeClr  = sharpe > 0 ? 'var(--green)' : 'var(--red)';
      const ddClr      = dd > -50 ? 'var(--text)' : 'var(--red)';
      const stratEsc   = strat.replace(/'/g, "\\'");

      html += `<tr data-strategy="${{strat}}" onclick="openModal('${{stratEsc}}')" style="cursor:pointer">
        <td title="${{strat}}">${{strat}}</td>
        <td>${{sym}}</td>
        <td>${{tf}}</td>
        <td style="color:${{sharpeClr}}">${{sharpe.toFixed(3)}}</td>
        <td>${{ret >= 0 ? '+' : ''}}${{ret.toFixed(1)}}%</td>
        <td style="color:${{ddClr}}">${{dd.toFixed(1)}}%</td>
        <td><span class="badge ${{statusCls}}">${{status}}</span></td>
      </tr>`;
    }});

    if (!html) {{
      html = `<tr><td colspan="7" style="color:var(--muted);padding:8px;font-style:italic">No ${{period}} results loaded for "${{indName}}" — results table shows top 100 only.</td></tr>`;
    }}
    dstTbody.innerHTML = html;
  }}

  panel.style.display = '';
  setTimeout(() => panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }}), 80);
}}

function clearConceptPanel() {{
  const panel = document.getElementById('concept-results-panel');
  if (panel) panel.style.display = 'none';
  clearIndicatorFilter('train');
  clearIndicatorFilter('test');
}}

function clearIndicatorFilter(period) {{
  if (_tblState[period]) _tblState[period].indicator = null;
  const badge = document.getElementById(period + '-ind-badge');
  if (badge) badge.style.display = 'none';
  applyFilters(period);
}}

// --- Detail view: equity chart + trade list ---
let detailChart = null;

async function loadDetail(strategy, symbol, period) {{
  const dv = document.getElementById('detail-view');
  if (!dv) return;
  dv.innerHTML = '<p style="color:var(--muted)">Loading backtest for ' + symbol + ' (' + period + ')...</p>';

  try {{
    const url = `/api/detail?strategy=${{encodeURIComponent(strategy)}}&symbol=${{encodeURIComponent(symbol)}}&period=${{encodeURIComponent(period)}}`;
    const resp = await fetch(url);
    const data = await resp.json();

    if (data.error) {{
      dv.innerHTML = `<p style="color:var(--red)">Error: ${{escHtml(data.error)}}</p>`;
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
    html += `<span style="color:var(--muted);font-size:0.75em;margin-left:10px">Shows OHLC candles, EMA lines, entry/exit markers</span>`;
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
      html += `<tr><td colspan="11" style="color:var(--muted);text-align:center">... and ${{trades.length - maxShow}} more trades</td></tr>`;
    }}
    html += `</tbody></table></div>`;
    html += `</div>`;

    dv.innerHTML = html;

    // Store data for chart overlay
    window.detailData = data;

    // Render equity chart
    if (detailChart) detailChart.destroy();
    const ctx2 = document.getElementById('detailEquityChart').getContext('2d');
    detailChart = new Chart(ctx2, {{
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
    dv.innerHTML = `<p style="color:var(--red)">Failed to load: ${{e.message}}</p>`;
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


def _build_recent_rows(df: pd.DataFrame) -> str:
    """Build rows for the Recent Experiments table in Overview pane."""
    if df.empty:
        return '<tr><td colspan="7" style="padding:10px;color:var(--muted);text-align:center">No experiments yet</td></tr>'
    rows = ""
    for _, row in df.iterrows():
        strategy = str(row.get("strategy", ""))
        best_sh = float(row.get("best_sharpe", 0) or 0)
        avg_sh = float(row.get("avg_sharpe", 0) or 0)
        avg_wr = float(row.get("avg_winrate", 0) or 0)
        avg_dd = float(row.get("avg_dd", 0) or 0)
        trades = int(row.get("total_trades", 0) or 0)
        kept = int(row.get("kept", 0) or 0)
        sh_color = "var(--green)" if best_sh > 0 else "var(--red)"
        avg_color = "var(--green)" if avg_sh > 0 else "var(--red)"
        badge = '<span class="badge badge-keep">keep</span>' if kept else '<span class="badge badge-discard">discard</span>'
        rows += f"""
        <tr onclick="openModal('{_esc(strategy)}')" style="cursor:pointer;border-bottom:1px solid var(--border)">
          <td style="padding:5px 8px;font-size:0.82em;color:var(--text)">{_esc(strategy)}</td>
          <td style="padding:5px 8px;text-align:right;color:{sh_color};font-weight:bold">{best_sh:.3f}</td>
          <td style="padding:5px 8px;text-align:right;color:{avg_color}">{avg_sh:.3f}</td>
          <td style="padding:5px 8px;text-align:right">{avg_wr:.0f}%</td>
          <td style="padding:5px 8px;text-align:right">{avg_dd:.1f}%</td>
          <td style="padding:5px 8px;text-align:right;color:var(--muted)">{trades:,}</td>
          <td style="padding:5px 8px;text-align:center">{badge}</td>
        </tr>"""
    return rows


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
        trades_val = int(row.get('trades', 0) or 0)
        rows += f"""
        <tr data-strategy="{_esc(strategy)}" data-symbol="{_esc(symbol)}" data-tf="{_esc(tf)}" data-sharpe="{sharpe_val:.4f}" data-winrate="{float(row.get('win_rate', 0) or 0):.1f}" data-return="{float(row.get('return_pct', 0) or 0):.2f}" data-dd="{float(row.get('max_dd_pct', 0) or 0):.2f}" data-trades="{trades_val}" onclick="openModal('{_esc(strategy)}')">
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
