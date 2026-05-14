"""
results_db.py - SQLite backend for experiment results.

Replaces results.tsv. SQLite WAL mode handles concurrent writers
(agent_research + run_systematic) without fcntl locking.

Schema mirrors TSV columns exactly; UNIQUE(strategy, symbol, period)
prevents duplicates natively via INSERT OR IGNORE.
"""

import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_FILE = Path("results.db")

TIMEFRAME_TOKENS = {
    "1m", "3m", "5m", "15m", "30m", "45m",
    "1h", "2h", "3h", "4h", "6h", "8h", "12h",
    "1d", "2d", "3d", "4d", "1w", "1mo",
}
FAMILY_SKIP_TOKENS = TIMEFRAME_TOKENS | {"mtf", "gen", "v", "strategy"}
SYMBOL_FOCUS_WEIGHTS = {
    "BTCUSDT": 1.35,
    "ETHUSDT": 1.00,
    "SOLUSDT": 0.35,
}

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
    """Insert experiment results into SQLite.

    Rows are upserted by (strategy, symbol, period). This prevents stale train
    rows from older runs surviving when the same strategy name is reused later.

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
        INSERT INTO results
            (git_commit, strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct,
             win_rate, profit_factor, trades, sortino, calmar, status, description, period)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy, symbol, period) DO UPDATE SET
            git_commit=excluded.git_commit,
            sharpe=excluded.sharpe,
            return_pct=excluded.return_pct,
            cagr_pct=excluded.cagr_pct,
            max_dd_pct=excluded.max_dd_pct,
            win_rate=excluded.win_rate,
            profit_factor=excluded.profit_factor,
            trades=excluded.trades,
            sortino=excluded.sortino,
            calmar=excluded.calmar,
            status=excluded.status,
            description=excluded.description
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


def query_top_rows(period: str, limit: int = 100, status: str = "keep") -> pd.DataFrame:
    """Return top active rows by Sharpe for a given period. Used by dashboard tables."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql_query(
            'SELECT strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct, '
            'win_rate, profit_factor, trades, sortino, calmar, status, period '
            f'FROM results WHERE period=? AND status=? ORDER BY sharpe DESC LIMIT {limit}',
            conn, params=(period, status)
        )


def query_avg_rows(period: str, limit: int = 50, status: str = "keep") -> pd.DataFrame:
    """Return per-strategy average metrics for active rows in a given period."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql_query(
            """SELECT strategy,
                AVG(sharpe) as sharpe, AVG(return_pct) as return_pct,
                AVG(max_dd_pct) as max_dd_pct, AVG(win_rate) as win_rate,
                AVG(trades) as trades, MAX(status) as status
               FROM results WHERE period=? AND status=?
               GROUP BY strategy
               ORDER BY AVG(sharpe) DESC LIMIT ?""",
            conn, params=(period, status, limit)
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


def strategy_family_key(strategy_name: str) -> str:
    """Collapse version/timeframe-heavy names into a reusable family key."""
    raw = (strategy_name or "").strip().lower()
    if not raw:
        return "unknown"
    raw = re.sub(r"_v\d+$", "", raw)
    tokens = []
    for token in raw.split("_"):
        if not token or token in FAMILY_SKIP_TOKENS:
            continue
        if re.fullmatch(r"v\d+", token):
            continue
        tokens.append(token)
    return "_".join(tokens[:8]) if tokens else raw


def compute_focus_score(rows: list[dict]) -> float:
    """Reward BTC/ETH test keeps much more than SOL-only wins."""
    if not rows:
        return 0.0

    best_by_symbol: dict[str, float] = {}
    for row in rows:
        symbol = row.get("symbol", "")
        sharpe = float(row.get("sharpe", 0) or 0)
        prev = best_by_symbol.get(symbol)
        if prev is None or sharpe > prev:
            best_by_symbol[symbol] = sharpe

    score = 0.0
    kept_symbols = {
        symbol for symbol, sharpe in best_by_symbol.items()
        if sharpe > 0
    }
    for symbol, sharpe in best_by_symbol.items():
        score += SYMBOL_FOCUS_WEIGHTS.get(symbol, 0.25) * max(0.0, sharpe)

    if "BTCUSDT" in kept_symbols and "ETHUSDT" in kept_symbols:
        score += 0.75
    elif "BTCUSDT" in kept_symbols:
        score += 0.35
    elif "ETHUSDT" in kept_symbols:
        score += 0.20
    elif kept_symbols == {"SOLUSDT"}:
        score -= 0.60

    if len(kept_symbols) >= 2:
        score += 0.20 * (len(kept_symbols) - 1)

    return round(score, 4)


def query_best_focus_score() -> float:
    """Best BTC/ETH-weighted score among historically kept strategies."""
    if not DB_FILE.exists():
        return 0.0
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT strategy, symbol, sharpe
            FROM results
            WHERE period='test' AND status='keep'
            """
        ).fetchall()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(dict(row))
    if not grouped:
        return 0.0
    return max(compute_focus_score(items) for items in grouped.values())


def query_strategy_family_stats(strategy_name: str | None = None, family: str | None = None) -> dict:
    """Summarize existing kept test performance for a strategy family."""
    family_key = family or strategy_family_key(strategy_name or "")
    if not DB_FILE.exists():
        return {
            "family": family_key,
            "distinct_strategies": 0,
            "btc_eth_variants": 0,
            "best_btc_eth_sharpe": 0.0,
            "best_sol_sharpe": 0.0,
            "best_focus_score": 0.0,
        }

    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT strategy, symbol, sharpe
            FROM results
            WHERE period='test' AND status='keep'
            """
        ).fetchall()

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        name = row["strategy"]
        if strategy_family_key(name) != family_key:
            continue
        if strategy_name and name == strategy_name:
            continue
        grouped.setdefault(name, []).append(dict(row))

    best_btc_eth = 0.0
    best_sol = 0.0
    btc_eth_variants = 0
    best_focus = 0.0
    for items in grouped.values():
        symbols = {row["symbol"] for row in items if float(row.get("sharpe", 0) or 0) > 0}
        if {"BTCUSDT", "ETHUSDT"} & symbols:
            btc_eth_variants += 1
        for row in items:
            sharpe = float(row.get("sharpe", 0) or 0)
            if row.get("symbol") in ("BTCUSDT", "ETHUSDT"):
                best_btc_eth = max(best_btc_eth, sharpe)
            elif row.get("symbol") == "SOLUSDT":
                best_sol = max(best_sol, sharpe)
        best_focus = max(best_focus, compute_focus_score(items))

    return {
        "family": family_key,
        "distinct_strategies": len(grouped),
        "btc_eth_variants": btc_eth_variants,
        "best_btc_eth_sharpe": round(best_btc_eth, 4),
        "best_sol_sharpe": round(best_sol, 4),
        "best_focus_score": round(best_focus, 4),
    }


def query_exhausted_families(limit: int = 15, min_variants: int = 4) -> list[dict]:
    """Families with lots of variants, useful for duplicate-avoidance prompts."""
    if not DB_FILE.exists():
        return []
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT strategy, symbol, status, period, sharpe
            FROM results
            """
        ).fetchall()

    families: dict[str, dict] = {}
    for row in rows:
        family = strategy_family_key(row["strategy"])
        info = families.setdefault(
            family,
            {
                "family": family,
                "strategies": set(),
                "kept_strategies": set(),
                "best_btc_eth_sharpe": 0.0,
                "best_sol_sharpe": 0.0,
            },
        )
        info["strategies"].add(row["strategy"])
        if row["period"] == "test" and row["status"] == "keep":
            info["kept_strategies"].add(row["strategy"])
            sharpe = float(row["sharpe"] or 0.0)
            if row["symbol"] in ("BTCUSDT", "ETHUSDT"):
                info["best_btc_eth_sharpe"] = max(info["best_btc_eth_sharpe"], sharpe)
            elif row["symbol"] == "SOLUSDT":
                info["best_sol_sharpe"] = max(info["best_sol_sharpe"], sharpe)

    ranked = []
    for info in families.values():
        total_variants = len(info["strategies"])
        if total_variants < min_variants:
            continue
        ranked.append(
            {
                "family": info["family"],
                "total_variants": total_variants,
                "kept_variants": len(info["kept_strategies"]),
                "best_btc_eth_sharpe": round(info["best_btc_eth_sharpe"], 4),
                "best_sol_sharpe": round(info["best_sol_sharpe"], 4),
            }
        )

    ranked.sort(
        key=lambda item: (
            item["total_variants"],
            item["kept_variants"],
            item["best_sol_sharpe"],
        ),
        reverse=True,
    )
    return ranked[:limit]


def query_recent_experiments(n: int = 12) -> pd.DataFrame:
    """Return last n active distinct strategies (train period), ordered newest first."""
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
                       FROM results WHERE period='train' AND status='keep'
                       GROUP BY strategy
                       ORDER BY last_id DESC
                       LIMIT ?
                   )
               )
               GROUP BY strategy
               HAVING MAX(CASE WHEN status='keep' THEN 1 ELSE 0 END) = 1
               ORDER BY MAX(id) DESC""",
            conn, params=(n,)
        )


def query_recent_strategy_names(limit: int = 25) -> list[str]:
    """Return the most recently written distinct strategy names by DB row id."""
    if not DB_FILE.exists():
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT strategy
            FROM (
                SELECT strategy, MAX(id) AS last_id
                FROM results
                GROUP BY strategy
                ORDER BY last_id DESC
                LIMIT ?
            )
            ORDER BY last_id DESC
            """,
            (limit,),
        ).fetchall()
    return [r[0] for r in rows]


def query_last_experiment_num() -> int:
    """Return the highest recorded exp#NNN value from result descriptions."""
    if not DB_FILE.exists():
        return 0
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT description
            FROM results
            WHERE description LIKE 'exp#%'
            ORDER BY id DESC
            LIMIT 5000
            """
        ).fetchall()

    best = 0
    for (description,) in rows:
        match = re.search(r"\bexp#(\d+)\b", description or "")
        if match:
            best = max(best, int(match.group(1)))
    return best


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


def delete_test_result(strategy_name: str, symbol: str):
    """Remove a stale test row for one strategy/symbol pair.

    Used when a later rerun with the same strategy name fails before producing
    fresh test-period output, which would otherwise leave old kept test rows in
    the dashboard.
    """
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM results WHERE strategy = ? AND symbol = ? AND period = 'test'",
            (strategy_name, symbol),
        )


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
    """Per-active-strategy train vs test best Sharpe. For scatter plot.
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
            WHERE t.any_kept = 1
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
