#!/usr/bin/env python3
"""
migrate_to_sqlite.py - One-time migration from results.tsv → results.db

Run once:
    python migrate_to_sqlite.py

Imports all rows from results.tsv into results.db, deduplicating on
(strategy, symbol, period). The TSV file is kept as backup.
"""

from pathlib import Path
import pandas as pd
from results_db import init_db, DB_FILE, get_conn

TSV = Path("results.tsv")


def migrate():
    if not TSV.exists():
        print("results.tsv not found — nothing to migrate.")
        return

    # Read TSV
    df = pd.read_csv(TSV, sep="\t")
    if "period" not in df.columns:
        df["period"] = "train"

    total = len(df)
    print(f"Migrating {total} rows from results.tsv → results.db ...")

    init_db()

    sql = """
        INSERT OR IGNORE INTO results
            (git_commit, strategy, symbol, sharpe, return_pct, cagr_pct, max_dd_pct,
             win_rate, profit_factor, trades, sortino, calmar, status, description, period)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row.get("commit", "")),
            str(row["strategy"]),
            str(row["symbol"]),
            float(row.get("sharpe", 0) or 0),
            float(row.get("return_pct", 0) or 0),
            float(row.get("cagr_pct", 0) or 0),
            float(row.get("max_dd_pct", 0) or 0),
            float(row.get("win_rate", 0) or 0),
            float(row.get("profit_factor", 0) or 0),
            int(row.get("trades", 0) or 0),
            float(row.get("sortino", 0) or 0),
            float(row.get("calmar", 0) or 0),
            str(row.get("status", "discard")),
            str(row.get("description", "")),
            str(row.get("period", "train")),
        ))

    with get_conn() as conn:
        conn.executemany(sql, rows)
        count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    print(f"Done. results.db now has {count} rows (inserted {min(total, count)}, "
          f"skipped {total - count} duplicates).")
    print(f"results.tsv kept as backup at {TSV.resolve()}")


if __name__ == "__main__":
    migrate()
