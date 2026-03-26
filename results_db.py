"""
results_db.py - SQLite backend for experiment results.

Replaces results.tsv. SQLite WAL mode handles concurrent writers
(agent_research + run_systematic) without fcntl locking.

Schema mirrors TSV columns exactly; UNIQUE(strategy, symbol, period)
prevents duplicates natively via INSERT OR IGNORE.
"""

import sqlite3
import time
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

_CREATE_INDEXES = [
    # For ORDER BY sharpe DESC per period (tables)
    "CREATE INDEX IF NOT EXISTS idx_period_sharpe    ON results(period, sharpe DESC);",
    # For GROUP BY strategy per period (avg table) — avoids temp B-TREE for GROUP BY
    "CREATE INDEX IF NOT EXISTS idx_period_strategy  ON results(period, strategy);",
    # For chart data (ordered by id per symbol+period)
    "CREATE INDEX IF NOT EXISTS idx_symbol_period_id ON results(symbol, period, id);",
    # For strategy modal lookups
    "CREATE INDEX IF NOT EXISTS idx_strategy         ON results(strategy);",
]

# In-memory stats cache — TTL 60s (cross-process safe) + dirty flag for same-process writes
_stats_cache: dict = {}
_stats_dirty: bool = True
_stats_ts: float = 0.0
_STATS_TTL: float = 60.0

# Concept stats cache — expensive (iterates all strategies), refresh every 5 min
_concept_cache: dict = {}
_concept_ts: float = 0.0
_CONCEPT_TTL: float = 300.0

# Scatter data cache — complex JOIN, refresh every 2 min
_scatter_cache: list = []
_scatter_ts: float = 0.0
_SCATTER_TTL: float = 120.0


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
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)


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

    global _stats_dirty
    _stats_dirty = True


def load_results() -> pd.DataFrame:
    """Return all results as a DataFrame (same columns as old TSV).
    NOTE: Loads full table — use query_*() functions for dashboard to avoid 55k-row scan."""
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


def query_stats() -> dict:
    """Return aggregate stats without loading all rows. Used by dashboard.
    Cache expires after 60s (cross-process safe) or immediately after same-process writes."""
    global _stats_cache, _stats_dirty, _stats_ts
    if _stats_cache and not _stats_dirty and (time.monotonic() - _stats_ts) < _STATS_TTL:
        return _stats_cache
    if not DB_FILE.exists():
        return {}
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(status='keep') as kept,
                SUM(status='discard') as discarded,
                SUM(status='crash') as crashed,
                MAX(sharpe) as best_sharpe,
                SUM(period='train') as train_total,
                SUM(period='train' AND status='keep') as train_kept,
                MAX(CASE WHEN period='train' THEN sharpe END) as train_best,
                SUM(period='test') as test_total,
                SUM(period='test' AND status='keep') as test_kept,
                MAX(CASE WHEN period='test' THEN sharpe END) as test_best,
                MAX(CASE WHEN period='test' AND status='keep' THEN sharpe END) as best_kept_test
            FROM results
        """).fetchone()
    keys = ["total","kept","discarded","crashed","best_sharpe",
            "train_total","train_kept","train_best",
            "test_total","test_kept","test_best","best_kept_test"]
    _stats_cache = dict(zip(keys, row))
    _stats_dirty = False
    _stats_ts = time.monotonic()
    return _stats_cache


def query_top_rows(period: str, limit: int = 100) -> pd.DataFrame:
    """Return top rows by Sharpe for a given period. Used by dashboard tables."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql_query(
            'SELECT strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct, '
            'win_rate, profit_factor, trades, sortino, calmar, status, period '
            f'FROM results WHERE period=? ORDER BY sharpe DESC LIMIT {limit}',
            conn, params=(period,)
        )


def query_avg_rows(period: str, limit: int = 50) -> pd.DataFrame:
    """Return per-strategy average metrics for a given period. Used by dashboard avg table."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql_query(
            """SELECT strategy,
                AVG(sharpe) as sharpe, AVG(return_pct) as return_pct,
                AVG(max_dd_pct) as max_dd_pct, AVG(win_rate) as win_rate,
                AVG(trades) as trades, MAX(status) as status
               FROM results WHERE period=?
               GROUP BY strategy
               ORDER BY AVG(sharpe) DESC LIMIT ?""",
            conn, params=(period, limit)
        )


def query_chart_data(symbol: str = "BTCUSDT", period: str = "train") -> list:
    """Return ordered Sharpe values for progress chart."""
    if not DB_FILE.exists():
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT sharpe FROM results WHERE symbol=? AND period=? ORDER BY id",
            (symbol, period)
        ).fetchall()
    return [r[0] for r in rows]


def query_distinct_symbols() -> list:
    """Return sorted list of distinct symbols."""
    if not DB_FILE.exists():
        return []
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM results ORDER BY symbol").fetchall()
    return [r[0] for r in rows]


def query_best_kept_sharpe() -> float:
    """Return best mean-per-strategy Sharpe among kept strategies. Used by agent loop."""
    if not DB_FILE.exists():
        return 0.0
    with get_conn() as conn:
        row = conn.execute("""
            SELECT MAX(avg_sharpe) FROM (
                SELECT AVG(sharpe) as avg_sharpe FROM results
                WHERE status='keep' GROUP BY strategy
            )
        """).fetchone()
    return float(row[0] or 0.0)


def query_recent_experiments(n: int = 12) -> pd.DataFrame:
    """Return last n distinct strategies (train period) with aggregate metrics, ordered newest first."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql_query(
            """SELECT strategy,
                      MAX(sharpe) as best_sharpe,
                      AVG(sharpe) as avg_sharpe,
                      AVG(win_rate) as avg_winrate,
                      AVG(max_dd_pct) as avg_dd,
                      SUM(trades) as total_trades,
                      MAX(CASE WHEN status='keep' THEN 1 ELSE 0 END) as kept
               FROM results
               WHERE period='train' AND strategy IN (
                   SELECT strategy FROM (
                       SELECT strategy, MAX(id) as last_id
                       FROM results WHERE period='train'
                       GROUP BY strategy
                       ORDER BY last_id DESC
                       LIMIT ?
                   )
               )
               GROUP BY strategy
               ORDER BY MAX(id) DESC""",
            conn, params=(n,)
        )


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


def query_concept_stats() -> dict:
    """Parse strategy names → indicator + timeframe coverage stats. For Concepts tab.
    Cached for 5 minutes — expensive iteration over all strategies."""
    global _concept_cache, _concept_ts
    now = time.monotonic()
    if _concept_cache and (now - _concept_ts) < _CONCEPT_TTL:
        return _concept_cache

    if not DB_FILE.exists():
        return {"indicators": {}, "timeframes": {}}
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT strategy,
                   MAX(sharpe) as best_sharpe,
                   SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) > 0 as any_kept
            FROM results
            GROUP BY strategy
        """).fetchall()

    VALID_TFS = ["1w", "1d", "12h", "6h", "4h", "1h", "30m", "15m", "5m"]

    # (name, test_fn) — ordered specific→general to avoid double-counting
    INDICATORS = [
        ("cRSI",     lambda s: "crsi" in s),
        ("HMA",      lambda s: "hma" in s or "hull" in s),
        ("RSI",      lambda s: ("rsi" in s.split("_") or any(p.startswith("rsi") for p in s.split("_"))) and "crsi" not in s),
        ("Donchian", lambda s: "donchian" in s),
        ("KAMA",     lambda s: "kama" in s),
        ("Fisher",   lambda s: "fisher" in s),
        ("Bollinger",lambda s: any(p in ("bb","bbands","bollinger","squeeze") for p in s.split("_"))),
        ("Keltner",  lambda s: "keltner" in s),
        ("ADX",      lambda s: "adx" in s.split("_")),
        ("Chop",     lambda s: "chop" in s),
        ("ATR",      lambda s: "atr" in s.split("_")),
        ("Volume",   lambda s: "volume" in s or "vol" in s.split("_")),
        ("Funding",  lambda s: "funding" in s),
        ("Regime",   lambda s: "regime" in s),
        ("Pullback", lambda s: "pullback" in s),
        ("STC",      lambda s: "stc" in s.split("_")),
        ("Ichimoku", lambda s: "ichimoku" in s),
        ("Pivot",    lambda s: "pivot" in s or "cpr" in s),
    ]

    ind_stats = {name: {"total": 0, "kept": 0, "best": None} for name, _ in INDICATORS}
    tf_stats  = {tf:   {"total": 0, "kept": 0, "best": None} for tf in VALID_TFS}
    heatmap: dict = {}

    for strategy, best_sharpe, any_kept in rows:
        s = strategy.lower()
        best_sharpe = float(best_sharpe or 0.0)

        # Primary TF: search all parts (handles both mtf_TF_... and gen_..._TF_... naming)
        tf_found = None
        parts = s.split("_")
        for p in parts:
            if p in tf_stats:
                tf_stats[p]["total"] += 1
                if any_kept:
                    tf_stats[p]["kept"] += 1
                if tf_stats[p]["best"] is None or best_sharpe > tf_stats[p]["best"]:
                    tf_stats[p]["best"] = best_sharpe
                tf_found = p
                break

        for ind_name, test_fn in INDICATORS:
            try:
                if test_fn(s):
                    ind_stats[ind_name]["total"] += 1
                    if any_kept:
                        ind_stats[ind_name]["kept"] += 1
                    if ind_stats[ind_name]["best"] is None or best_sharpe > ind_stats[ind_name]["best"]:
                        ind_stats[ind_name]["best"] = best_sharpe
            except Exception:
                pass

        # Heatmap: indicator × TF
        if tf_found:
            for ind_name, test_fn in INDICATORS:
                try:
                    if test_fn(s):
                        if ind_name not in heatmap:
                            heatmap[ind_name] = {}
                        prev = heatmap[ind_name].get(tf_found)
                        if prev is None or best_sharpe > prev:
                            heatmap[ind_name][tf_found] = round(best_sharpe, 3)
                except Exception:
                    pass

    result = {
        "indicators": {k: v for k, v in ind_stats.items() if v["total"] > 0},
        "timeframes":  {k: v for k, v in tf_stats.items()  if v["total"] > 0},
        "heatmap": heatmap,
    }
    _concept_cache = result
    _concept_ts = time.monotonic()
    return result


def query_scatter_data(limit: int = 400) -> list:
    """Per-strategy train vs test best Sharpe. For scatter plot.
    Cached for 2 minutes — complex JOIN query."""
    global _scatter_cache, _scatter_ts
    now = time.monotonic()
    if _scatter_cache and (now - _scatter_ts) < _SCATTER_TTL:
        return _scatter_cache

    if not DB_FILE.exists():
        return []
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT t.strategy, t.best_train, s.best_test, t.any_kept
            FROM (
                SELECT strategy,
                       MAX(sharpe) as best_train,
                       SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) > 0 as any_kept
                FROM results WHERE period='train'
                GROUP BY strategy
            ) t
            INNER JOIN (
                SELECT strategy, MAX(sharpe) as best_test
                FROM results WHERE period='test'
                GROUP BY strategy
            ) s ON t.strategy = s.strategy
            ORDER BY t.best_train DESC
            LIMIT ?
        """, (limit,)).fetchall()
    result = [
        {"n": r[0][:35], "tr": round(float(r[1]), 3), "te": round(float(r[2]), 3), "k": bool(r[3])}
        for r in rows
    ]
    _scatter_cache = result
    _scatter_ts = time.monotonic()
    return result


# Auto-init on import
init_db()
