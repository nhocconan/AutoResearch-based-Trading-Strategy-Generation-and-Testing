"""
results_db.py - SQLite backend for experiment results.

Replaces results.tsv. SQLite WAL mode handles concurrent writers
(agent_research + run_systematic) without fcntl locking.

Schema mirrors TSV columns exactly; UNIQUE(strategy, symbol, period)
prevents duplicates natively via INSERT OR IGNORE.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_FILE = Path("results.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    git_commit   TEXT    DEFAULT '',
    strategy     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    sharpe       REAL,
    return_pct   REAL,
    cagr_pct     REAL,
    max_dd_pct   REAL,
    win_rate     REAL,
    profit_factor REAL,
    trades       INTEGER,
    sortino      REAL,
    calmar       REAL,
    status       TEXT    DEFAULT 'discard',
    description  TEXT    DEFAULT '',
    period       TEXT    NOT NULL,
    UNIQUE(strategy, symbol, period)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_status_period ON results(status, period);
"""


def get_conn() -> sqlite3.Connection:
    """Open DB with WAL mode for concurrent access."""
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    """Create table and indexes if they don't exist."""
    with get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)


def append_results(results: list[dict], status: str, description: str, period: str = "train"):
    """Insert experiment results into SQLite. Skips duplicates (strategy, symbol, period).

    Each dict in results should contain the backtest metrics with keys:
    sharpe_ratio, total_return_pct, annual_return_pct, max_drawdown_pct,
    win_rate, profit_factor, num_trades, sortino_ratio, calmar_ratio, symbol, strategy.
    """
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        commit = r.stdout.strip()
    except Exception:
        commit = ""

    rows = []
    for m in results:
        rows.append((
            commit,
            m.get("strategy", "unknown"),
            m["symbol"],
            round(m["sharpe_ratio"], 4),
            round(m["total_return_pct"], 2),
            round(m["annual_return_pct"], 2),
            round(m["max_drawdown_pct"], 2),
            round(m["win_rate"], 1),
            round(m["profit_factor"], 2),
            int(m["num_trades"]),
            round(m["sortino_ratio"], 4),
            round(m["calmar_ratio"], 4),
            status,
            description[:80],
            period,
        ))

    if not rows:
        return

    sql = """
        INSERT OR IGNORE INTO results
            (git_commit, strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct,
             win_rate, profit_factor, trades, sortino, calmar, status, description, period)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)


def load_results() -> pd.DataFrame:
    """Return all results as a DataFrame (same columns as old TSV)."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(
                'SELECT git_commit AS "commit", strategy, symbol, sharpe, return_pct, cagr_pct, '
                'max_dd_pct, win_rate, profit_factor, trades, sortino, calmar, '
                'status, description, period FROM results',
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def upsert_results(rows: list[dict]):
    """Insert or replace rows (used by revalidate.py full rebuild).

    Each dict must have all column keys matching the DB schema.
    """
    if not rows:
        return
    sql = """
        INSERT OR REPLACE INTO results
            (git_commit, strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct,
             win_rate, profit_factor, trades, sortino, calmar, status, description, period)
        VALUES (:commit, :strategy, :symbol, :sharpe, :return_pct, :cagr_pct, :max_dd_pct,
                :win_rate, :profit_factor, :trades, :sortino, :calmar, :status, :description, :period)
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)


def delete_strategy(strategy_name: str):
    """Remove all rows for a given strategy (used by revalidate --strategy)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM results WHERE strategy = ?", (strategy_name,))


def metrics_to_db_dict(
    metrics: dict,
    strategy_name: str,
    symbol: str,
    commit: str = "",
    status: str = "keep",
    description: str = "",
    period: str = "train",
) -> dict:
    """Convert backtest metrics dict to a flat DB row dict."""
    return {
        "commit": commit,
        "strategy": strategy_name,
        "symbol": symbol,
        "sharpe": round(metrics["sharpe_ratio"], 4),
        "return_pct": round(metrics["total_return_pct"], 2),
        "cagr_pct": round(metrics["annual_return_pct"], 2),
        "max_dd_pct": round(metrics["max_drawdown_pct"], 2),
        "win_rate": round(metrics["win_rate"], 1),
        "profit_factor": round(metrics["profit_factor"], 2),
        "trades": int(metrics["num_trades"]),
        "sortino": round(metrics["sortino_ratio"], 4),
        "calmar": round(metrics["calmar_ratio"], 4),
        "status": status,
        "description": description[:80],
        "period": period,
    }


# Auto-init on import
init_db()
