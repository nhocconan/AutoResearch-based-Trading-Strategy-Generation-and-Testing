#!/usr/bin/env python3
"""
revalidate.py - Rerun all saved strategies and rebuild results.tsv
==================================================================
Reruns backtest on ALL strategies in strategies/ dir with current
backtest engine and fee model. Replaces results.tsv with fresh data.

Usage:
    python revalidate.py              # Revalidate all
    python revalidate.py --strategy mtf_hma_rsi_zscore_v1  # Single strategy
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from backtest import run_strategy_backtest
from evaluate import TSV_HEADER, compute_metrics, metrics_to_tsv_row
from prepare import load_config

STRATEGIES_DIR = Path("strategies")
RESULTS_FILE = Path("results.tsv")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get_git_commit() -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return "revalidate"


def revalidate_strategy(strategy_path: str, symbols: list, commit: str) -> list[str]:
    """Run backtest on all symbols for both train and test. Returns TSV rows."""
    rows = []
    strategy_name = Path(strategy_path).stem

    for period in ["train", "test"]:
        for symbol in symbols:
            try:
                result = run_strategy_backtest(
                    strategy_path=strategy_path,
                    symbol=symbol,
                    period=period,
                )
                m = compute_metrics(result)
                status = "keep" if m["sharpe_ratio"] > 0 and m["max_drawdown_pct"] > -50 else "discard"
                desc = f"revalidated {strategy_name}"
                row = metrics_to_tsv_row(
                    metrics=m,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    commit=commit,
                    status=status,
                    description=desc,
                    period=period,
                )
                rows.append(row)
                sharpe = m["sharpe_ratio"]
                dd = m["max_drawdown_pct"]
                ret = m["total_return_pct"]
                trades = m["num_trades"]
                print(f"  {symbol:8s} {period:5s} Sharpe={sharpe:7.3f} Return={ret:+10.1f}% DD={dd:7.1f}% Trades={trades:5d} [{status}]")
            except Exception as e:
                print(f"  {symbol:8s} {period:5s} ERROR: {e}")
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

    # Backup old results
    if RESULTS_FILE.exists() and not args.strategy:
        backup = RESULTS_FILE.with_suffix(".tsv.pre_revalidate")
        shutil.copy(RESULTS_FILE, backup)
        print(f"Backed up old results to {backup}")

    all_rows = []
    for path in paths:
        if not path.exists():
            print(f"\nSkipping {path} (not found)")
            continue
        print(f"\n{path.stem}:")
        rows = revalidate_strategy(str(path), SYMBOLS, commit)
        all_rows.extend(rows)

    # Write results
    if not args.strategy:
        # Full rewrite
        with open(RESULTS_FILE, "w") as f:
            f.write(TSV_HEADER + "\n")
            for row in all_rows:
                f.write(row + "\n")
        print(f"\n{'=' * 70}")
        print(f"Wrote {len(all_rows)} rows to {RESULTS_FILE}")
    else:
        # Append
        with open(RESULTS_FILE, "a") as f:
            for row in all_rows:
                f.write(row + "\n")
        print(f"\nAppended {len(all_rows)} rows to {RESULTS_FILE}")

    # Summary
    kept = sum(1 for r in all_rows if "\tkeep\t" in r)
    print(f"Kept: {kept} / {len(all_rows)} rows")


if __name__ == "__main__":
    main()
