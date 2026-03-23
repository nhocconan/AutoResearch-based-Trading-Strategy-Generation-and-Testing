#!/usr/bin/env python3
"""
revalidate.py - Rerun all saved strategies and rebuild results.db
==================================================================
Reruns backtest on ALL strategies in strategies/ dir with current
backtest engine and fee model. Replaces results.db rows with fresh data.

Usage:
    python revalidate.py              # Revalidate all
    python revalidate.py --strategy mtf_hma_rsi_zscore_v1  # Single strategy
"""

import argparse
from pathlib import Path

from backtest import run_strategy_backtest
from evaluate import compute_metrics, metrics_to_tsv_row
from results_db import upsert_results, delete_strategy, metrics_to_db_dict

STRATEGIES_DIR = Path("strategies")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get_git_commit() -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return "revalidate"


def revalidate_strategy(strategy_path: str, symbols: list, commit: str) -> list[dict]:
    """Run backtest on all symbols for both train and test. Returns list of DB row dicts."""
    rows = []
    strategy_name = Path(strategy_path).stem

    # Train first — early stop if first symbol fails
    train_rows = []
    train_pass = True
    for symbol in symbols:
        try:
            result = run_strategy_backtest(strategy_path=strategy_path, symbol=symbol, period="train")
            m = compute_metrics(result)
            sharpe = m["sharpe_ratio"]
            trades = m["num_trades"]
            dd = m["max_drawdown_pct"]
            status = "keep" if sharpe > 0 and trades >= 5 and dd > -50 else "discard"
            train_rows.append(metrics_to_db_dict(m, strategy_name, symbol, commit, status, f"revalidated {strategy_name}", "train"))
            print(f"  {symbol:8s} train Sharpe={sharpe:7.3f} Return={m['total_return_pct']:+10.1f}% DD={dd:7.1f}% Trades={trades:5d} [{status}]")
            if sharpe <= 0 or trades < 5:
                print(f"  → EARLY STOP train: {symbol} failed (Sharpe={sharpe:.3f} trades={trades})")
                train_pass = False
                break
        except Exception as e:
            print(f"  {symbol:8s} train ERROR: {e}")
            train_pass = False
            break

    rows.extend(train_rows)

    # Only run test if ALL train symbols passed
    if not train_pass:
        print(f"  → SKIP test (train failed)")
        return rows

    for symbol in symbols:
        try:
            result = run_strategy_backtest(strategy_path=strategy_path, symbol=symbol, period="test")
            m = compute_metrics(result)
            sharpe = m["sharpe_ratio"]
            trades = m["num_trades"]
            dd = m["max_drawdown_pct"]
            status = "keep" if sharpe > 0 and trades >= 3 and dd > -50 else "discard"
            rows.append(metrics_to_db_dict(m, strategy_name, symbol, commit, status, f"revalidated {strategy_name}", "test"))
            print(f"  {symbol:8s} test  Sharpe={sharpe:7.3f} Return={m['total_return_pct']:+10.1f}% DD={dd:7.1f}% Trades={trades:5d} [{status}]")
            if sharpe <= 0 or trades < 3:
                print(f"  → EARLY STOP test: {symbol} failed")
                break
        except Exception as e:
            print(f"  {symbol:8s} test  ERROR: {e}")
            break

    return rows


def main():
    parser = argparse.ArgumentParser(description="Revalidate all saved strategies")
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy to revalidate")
    args = parser.parse_args()

    commit = get_git_commit()

    if args.strategy:
        paths = [STRATEGIES_DIR / f"{args.strategy}.py"]
    else:
        paths = sorted(STRATEGIES_DIR.glob("*.py"))

    if not paths:
        print("No strategies found in strategies/")
        return

    print(f"Revalidating {len(paths)} strategies on {SYMBOLS}")
    print(f"Fee model: 0.04% taker + 0.01% slippage per side (both entry & exit)")
    print(f"{'=' * 70}")

    all_rows = []
    for path in paths:
        if not path.exists():
            print(f"\nSkipping {path} (not found)")
            continue
        print(f"\n{path.stem}:")
        if args.strategy:
            # Single strategy: remove old rows then insert fresh ones
            delete_strategy(path.stem)
        rows = revalidate_strategy(str(path), SYMBOLS, commit)
        all_rows.extend(rows)

    # Write to SQLite (upsert replaces existing rows for full rebuild)
    upsert_results(all_rows)
    print(f"\n{'=' * 70}")
    print(f"Upserted {len(all_rows)} rows into results.db")

    # Summary
    kept = sum(1 for r in all_rows if r.get("status") == "keep")
    print(f"Kept: {kept} / {len(all_rows)} rows")


if __name__ == "__main__":
    main()
