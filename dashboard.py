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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd

from validator import validate_strategy, ValidationResult

RESULTS_FILE = Path("results.tsv")
STRATEGIES_DIR = Path("strategies")
STRATEGY_FILE = Path("strategy.py")


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RESULTS_FILE, sep="\t")
        if "period" not in df.columns:
            df["period"] = "train"
        return df
    except Exception:
        return pd.DataFrame()


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


def _build_avg_table_rows(df: pd.DataFrame, limit: int = 50) -> str:
    """Build table rows showing average metrics per strategy across all symbols."""
    if df.empty or "sharpe" not in df.columns:
        return '<tr><td colspan="8" class="no-data" style="padding:10px;color:#8b949e">No data</td></tr>'

    agg = df.groupby("strategy").agg({
        "sharpe": "mean",
        "return_pct": "mean",
        "max_dd_pct": "mean",
        "win_rate": "mean",
        "trades": "mean",
        "status": "first",
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
    df = load_results()
    git_log = get_git_log()
    current_strategy = get_current_strategy()
    current_val = run_validation(current_strategy)

    # Validate current strategy
    current_val_html = build_validation_html(current_val)
    current_val_badge = "badge-pass" if current_val.valid else "badge-fail"
    current_val_label = "PASS" if current_val.valid else "FAIL"

    # Overall stats
    total = len(df)
    kept = len(df[df["status"] == "keep"]) if total > 0 else 0
    discarded = len(df[df["status"] == "discard"]) if total > 0 else 0
    crashed = len(df[df["status"] == "crash"]) if total > 0 else 0
    best_sharpe = df["sharpe"].max() if total > 0 and "sharpe" in df.columns else 0

    # Split by period
    train_df = df[df["period"] == "train"] if total > 0 else pd.DataFrame()
    test_df = df[df["period"] == "test"] if total > 0 else pd.DataFrame()

    train_total = len(train_df)
    test_total = len(test_df)
    train_kept = len(train_df[train_df["status"] == "keep"]) if train_total > 0 else 0
    test_kept = len(test_df[test_df["status"] == "keep"]) if test_total > 0 else 0
    train_best = train_df["sharpe"].max() if train_total > 0 and "sharpe" in train_df.columns else 0
    test_best = test_df["sharpe"].max() if test_total > 0 and "sharpe" in test_df.columns else 0

    # Build results table rows (all rows, JS filters client-side)
    train_rows = _build_table_rows(train_df, limit=100)
    train_avg_rows = _build_avg_table_rows(train_df, limit=50)
    test_rows = _build_table_rows(test_df, limit=100)
    test_avg_rows = _build_avg_table_rows(test_df, limit=50)

    # Get unique symbols
    symbols = sorted(df["symbol"].unique().tolist()) if total > 0 and "symbol" in df.columns else []
    symbols_json = json.dumps(symbols)

    # Chart data: Sharpe over time for BTCUSDT (train only)
    chart_data, chart_labels, running_best_data = "[]", "[]", "[]"
    if train_total > 0 and "sharpe" in train_df.columns:
        btc = train_df[train_df["symbol"] == "BTCUSDT"].reset_index(drop=True)
        if len(btc) > 0:
            sharpes = btc["sharpe"].tolist()
            labels = [f"#{i+1}" for i in range(len(sharpes))]
            running_best = []
            best_so_far = -999
            for s in sharpes:
                best_so_far = max(best_so_far, s)
                running_best.append(round(best_so_far, 4))
            chart_data = json.dumps([round(s, 4) for s in sharpes])
            chart_labels = json.dumps(labels)
            running_best_data = json.dumps(running_best)

    # Strategy modal data
    strategy_json = build_strategy_data(df)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="600">
<title>LLM Trading Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
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
  th {{ background: #21262d; padding: 8px; text-align: left; color: #8b949e; font-size: 0.8em; }}
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
  <label>View:</label>
  <button class="filter-btn active" onclick="filterTable('train', 'ALL')">All Rows</button>
  {''.join(f'<button class="filter-btn" onclick="filterTable(&#39;train&#39;, &#39;{s}&#39;)">{s}</button>' for s in symbols)}
  <button class="filter-btn" onclick="filterTable('train', 'AVG')" style="border-color:#f0883e;color:#f0883e">Avg All Symbols</button>
  <span class="filter-info" id="train-filter-info"></span>
</div>
<table id="train-table">
  <thead><tr>
    <th>Strategy</th><th>Symbol</th><th>Sharpe</th><th>Return</th>
    <th>Max DD</th><th>Win Rate</th><th>Trades</th><th>Status</th>
  </tr></thead>
  <tbody id="train-tbody-rows">{train_rows}</tbody>
  <tbody id="train-tbody-avg" style="display:none">{train_avg_rows}</tbody>
</table>

<h2>Test Results (2025+) <span style="font-size:0.7em;color:#8b949e">(click row for details)</span></h2>
{f"""<div class="filter-bar" id="test-filter-bar">
  <label>View:</label>
  <button class="filter-btn active" onclick="filterTable('test', 'ALL')">All Rows</button>
  {''.join(f'<button class="filter-btn" onclick="filterTable(&#39;test&#39;, &#39;{s}&#39;)">{s}</button>' for s in symbols)}
  <button class="filter-btn" onclick="filterTable('test', 'AVG')" style="border-color:#f0883e;color:#f0883e">Avg All Symbols</button>
  <span class="filter-info" id="test-filter-info"></span>
</div>
<table id="test-table">
  <thead><tr><th>Strategy</th><th>Symbol</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Win Rate</th><th>Trades</th><th>Status</th></tr></thead>
  <tbody id="test-tbody-rows">{test_rows}</tbody>
  <tbody id="test-tbody-avg" style="display:none">{test_avg_rows}</tbody>
</table>""" if test_total > 0 else '<p class="no-data">No test results yet — kept strategies are automatically evaluated on 2025+ data.</p>'}

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
const STRATEGIES = {strategy_json};

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
  const s = STRATEGIES[strategyName];
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

    // Price chart with entries/exits + indicators
    html += `<h4 style="margin-top:8px">Price Chart (${{data.timeframe}}) with Entry/Exit Markers</h4>`;
    html += `<div class="detail-chart" style="height:320px"><canvas id="detailPriceChart"></canvas></div>`;

    // Signal strength chart
    html += `<div style="height:80px;margin:-5px 0 5px"><canvas id="detailSignalChart"></canvas></div>`;

    // Equity chart
    html += `<h4>Equity Curve</h4>`;
    html += `<div class="detail-chart"><canvas id="detailEquityChart"></canvas></div>`;

    // Trade list
    html += `<h4 style="margin-top:12px">Trade History (${{data.trades.length}} trades)</h4>`;
    html += `<div class="trade-list"><table class="metrics-table">`;
    html += `<thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Size</th><th>PnL</th><th>PnL%</th><th>Fees</th><th>Funding</th></tr></thead><tbody>`;

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

    // --- Price chart with indicators + trade markers ---
    if (data.price_close && data.price_close.length > 0) {{
      if (window._priceChart) window._priceChart.destroy();
      if (window._sigChart) window._sigChart.destroy();

      // Build entry/exit annotation points
      const entryPoints = [];
      const exitPoints = [];
      const timeLookup = {{}};
      data.price_times.forEach((t, i) => {{ timeLookup[t.substring(0,16)] = i; }});

      (data.trade_markers || []).forEach(tm => {{
        const ei = timeLookup[tm.entry_time];
        const xi = timeLookup[tm.exit_time];
        if (ei !== undefined) entryPoints.push({{x: ei, y: tm.entry_price, dir: tm.direction}});
        if (xi !== undefined) exitPoints.push({{x: xi, y: tm.exit_price, pnl: tm.pnl_pct}});
      }});

      const pLabels = data.price_times.map(t => t.substring(5,16));

      const pCtx = document.getElementById('detailPriceChart').getContext('2d');
      window._priceChart = new Chart(pCtx, {{
        type: 'line',
        data: {{
          labels: pLabels,
          datasets: [
            {{
              label: 'Close',
              data: data.price_close,
              borderColor: '#c9d1d9',
              borderWidth: 1,
              pointRadius: 0,
              tension: 0.1,
              yAxisID: 'y',
              order: 3,
            }},
            {{
              label: 'EMA 21',
              data: data.ema21,
              borderColor: '#f0883e',
              borderWidth: 1,
              pointRadius: 0,
              borderDash: [3,2],
              tension: 0.1,
              yAxisID: 'y',
              order: 4,
            }},
            {{
              label: 'EMA 55',
              data: data.ema55,
              borderColor: '#a371f7',
              borderWidth: 1,
              pointRadius: 0,
              borderDash: [5,3],
              tension: 0.1,
              yAxisID: 'y',
              order: 4,
            }},
            {{
              label: 'Long Entry',
              data: entryPoints.filter(p=>p.dir==='LONG').map(p=>({{x:p.x,y:p.y}})),
              type: 'scatter',
              backgroundColor: '#2ecc71',
              borderColor: '#2ecc71',
              pointRadius: 4,
              pointStyle: 'triangle',
              yAxisID: 'y',
              order: 1,
            }},
            {{
              label: 'Short Entry',
              data: entryPoints.filter(p=>p.dir==='SHORT').map(p=>({{x:p.x,y:p.y}})),
              type: 'scatter',
              backgroundColor: '#e74c3c',
              borderColor: '#e74c3c',
              pointRadius: 4,
              pointStyle: 'triangleDown',  // Fixed: use 'triangle' and rotation
              rotation: 180,
              yAxisID: 'y',
              order: 1,
            }},
            {{
              label: 'Exit (win)',
              data: exitPoints.filter(p=>p.pnl>=0).map(p=>({{x:p.x,y:p.y}})),
              type: 'scatter',
              backgroundColor: 'rgba(46,204,113,0.5)',
              pointRadius: 3,
              pointStyle: 'crossRot',
              yAxisID: 'y',
              order: 2,
            }},
            {{
              label: 'Exit (loss)',
              data: exitPoints.filter(p=>p.pnl<0).map(p=>({{x:p.x,y:p.y}})),
              type: 'scatter',
              backgroundColor: 'rgba(231,76,60,0.5)',
              pointRadius: 3,
              pointStyle: 'crossRot',
              yAxisID: 'y',
              order: 2,
            }},
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', boxWidth: 12, font: {{size: 10}} }}, position: 'top' }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + (ctx.parsed.y||0).toLocaleString() }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 15, font: {{size: 9}} }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#21262d' }} }}
          }}
        }}
      }});

      // Signal strength mini-chart
      const sCtx = document.getElementById('detailSignalChart').getContext('2d');
      const sigColors = data.signals.map(s => s > 0 ? 'rgba(46,204,113,0.6)' : s < 0 ? 'rgba(231,76,60,0.6)' : 'rgba(139,148,158,0.2)');
      window._sigChart = new Chart(sCtx, {{
        type: 'bar',
        data: {{
          labels: pLabels,
          datasets: [{{
            label: 'Signal',
            data: data.signals,
            backgroundColor: sigColors,
            borderWidth: 0,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ display: false }},
            y: {{ min: -0.5, max: 0.5, ticks: {{ color: '#8b949e', stepSize: 0.25, font: {{size: 9}} }}, grid: {{ color: '#21262d' }} }}
          }}
        }}
      }});
    }}

  }} catch(e) {{
    dv.innerHTML = `<p style="color:#e74c3c">Failed to load: ${{e.message}}</p>`;
  }}
}}
</script>
</body>
</html>"""


def _build_table_rows(df: pd.DataFrame, limit: int = 100) -> str:
    if df.empty or "sharpe" not in df.columns:
        return '<tr><td colspan="8" class="no-data" style="padding:10px;color:#8b949e">No data</td></tr>'

    best = df.sort_values("sharpe", ascending=False).head(limit)
    rows = ""
    for _, row in best.iterrows():
        sharpe_val = float(row.get("sharpe", 0) or 0)
        color = "#2ecc71" if sharpe_val > 0 else "#e74c3c"
        status = row.get("status", "")
        badge_cls = f"badge-{status}" if status in ("keep", "discard", "crash") else "badge-discard"
        strategy = str(row.get("strategy", ""))
        symbol = str(row.get("symbol", ""))
        rows += f"""
        <tr data-strategy="{_esc(strategy)}" data-symbol="{_esc(symbol)}" data-sharpe="{sharpe_val:.4f}" onclick="openModal('{_esc(strategy)}')">
            <td>{_esc(strategy)}</td>
            <td>{_esc(symbol)}</td>
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

        # Sample price data for chart (max 800 points for good resolution)
        price_n = len(prices)
        price_step = max(1, price_n // 800)
        price_times = prices["open_time"].astype(str).values[::price_step].tolist()
        price_close = [round(float(v), 2) for v in prices["close"].values[::price_step]]
        price_high = [round(float(v), 2) for v in prices["high"].values[::price_step]]
        price_low = [round(float(v), 2) for v in prices["low"].values[::price_step]]
        signal_sampled = [round(float(v), 4) for v in signals[::price_step]]

        # Compute common indicators for chart overlay
        close_s = pd.Series(prices["close"].values)
        ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
        ema55 = close_s.ewm(span=55, min_periods=55, adjust=False).mean().values
        ema21_sampled = [round(float(v), 2) if not np.isnan(v) else None for v in ema21[::price_step]]
        ema55_sampled = [round(float(v), 2) if not np.isnan(v) else None for v in ema55[::price_step]]

        # Trade entry/exit markers (map to chart x-axis)
        trade_markers = []
        for t in result.trades[:500]:  # max 500 markers
            trade_markers.append({
                "entry_time": str(t.entry_time)[:16],
                "exit_time": str(t.exit_time)[:16],
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "pnl_pct": round(t.pnl_pct * 100, 3),
            })

        # Full trades list
        trades = []
        for t in result.trades:
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
            # Price chart data
            "price_times": price_times,
            "price_close": price_close,
            "price_high": price_high,
            "price_low": price_low,
            "signals": signal_sampled,
            "ema21": ema21_sampled,
            "ema55": ema55_sampled,
            "trade_markers": trade_markers,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


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
        else:
            html = render_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress access logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Auto-refreshes every 10 minutes. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
