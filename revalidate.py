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
from evaluate import compute_metrics
from research_rules import test_symbol_pass, train_symbol_pass
from results_db import upsert_results, delete_strategy, metrics_to_db_dict
from validator import run_prefix_lookahead_check, validate_file

STRATEGIES_DIR = Path("strategies")
DOCS_DIR = Path("docs/strategies")
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

    train_passed_symbols = []
    for symbol in symbols:
        try:
            result = run_strategy_backtest(strategy_path=strategy_path, symbol=symbol, period="train")
            m = compute_metrics(result)
            sharpe = m["sharpe_ratio"]
            trades = m["num_trades"]
            dd = m["max_drawdown_pct"]
            status = "keep" if train_symbol_pass(m) else "discard"
            rows.append(metrics_to_db_dict(m, strategy_name, symbol, commit, status, f"revalidated {strategy_name}", "train"))
            print(f"  {symbol:8s} train Sharpe={sharpe:7.3f} Return={m['total_return_pct']:+10.1f}% DD={dd:7.1f}% Trades={trades:5d} [{status}]")
            if status == "keep":
                train_passed_symbols.append(symbol)
            else:
                print(f"  {symbol:8s} test  SKIP: train failed")
        except Exception as e:
            print(f"  {symbol:8s} train ERROR: {e}")

    for symbol in train_passed_symbols:
        try:
            result = run_strategy_backtest(strategy_path=strategy_path, symbol=symbol, period="test")
            m = compute_metrics(result)
            sharpe = m["sharpe_ratio"]
            trades = m["num_trades"]
            dd = m["max_drawdown_pct"]
            status = "keep" if test_symbol_pass(m) else "discard"
            rows.append(metrics_to_db_dict(m, strategy_name, symbol, commit, status, f"revalidated {strategy_name}", "test"))
            print(f"  {symbol:8s} test  Sharpe={sharpe:7.3f} Return={m['total_return_pct']:+10.1f}% DD={dd:7.1f}% Trades={trades:5d} [{status}]")
        except Exception as e:
            print(f"  {symbol:8s} test  ERROR: {e}")

    return rows


def purge_invalid_strategy(strategy_name: str) -> None:
    delete_strategy(strategy_name)
    doc_path = DOCS_DIR / f"{strategy_name}.md"
    if doc_path.exists():
        doc_path.unlink()


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
    print("Fee model: 0.04% taker + 0.01% slippage per side (both entry & exit)")
    print(f"{'=' * 70}")

    all_rows = []
    refreshed_strategies = set()
    for path in paths:
        if not path.exists():
            print(f"\nSkipping {path} (not found)")
            continue
        print(f"\n{path.stem}:")
        validation = validate_file(str(path))
        if not validation.valid:
            purge_invalid_strategy(path.stem)
            print("  INVALID: purged stored rows/docs and skipped backtest")
            for error in validation.errors[:5]:
                print(f"    - {error}")
            continue
        prefix_ok, prefix_msg = run_prefix_lookahead_check(str(path))
        if not prefix_ok:
            purge_invalid_strategy(path.stem)
            print("  INVALID: prefix look-ahead audit failed; purged stored rows/docs and skipped backtest")
            print(f"    - {prefix_msg}")
            continue

        # Remove old rows before inserting fresh ones so invalid/stale rows cannot survive.
        delete_strategy(path.stem)
        rows = revalidate_strategy(str(path), SYMBOLS, commit)
        all_rows.extend(rows)
        if rows:
            refreshed_strategies.add(path.stem)

    # Write to SQLite (upsert replaces existing rows for full rebuild)
    upsert_results(all_rows)
    if refreshed_strategies:
        from verification_remediation import refresh_strategy_doc
        for strategy_name in sorted(refreshed_strategies):
            refresh_strategy_doc(strategy_name)
    print(f"\n{'=' * 70}")
    print(f"Upserted {len(all_rows)} rows into results.db")

    # Summary
    kept = sum(1 for r in all_rows if r.get("status") == "keep")
    print(f"Kept: {kept} / {len(all_rows)} rows")


if __name__ == "__main__":
    main()
