#!/usr/bin/env python3
"""
run_research.py - Autonomous Research Loop Runner
==================================================
Runs backtest experiments on strategy.py and tracks results.
Can be used standalone or driven by an LLM agent.

Usage:
    python run_research.py                    # Run current strategy on all symbols
    python run_research.py --symbol BTCUSDT   # Single symbol
    python run_research.py --test             # Run on test period
    python run_research.py --full             # Train + test evaluation
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from backtest import run_strategy_backtest, print_result_summary
from evaluate import compute_metrics, print_metrics, metrics_to_tsv_row, TSV_HEADER
from prepare import load_config


def get_git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def ensure_results_file(results_path: Path):
    """Create results.tsv with header if it doesn't exist."""
    if not results_path.exists():
        with open(results_path, "w") as f:
            f.write(TSV_HEADER + "\n")


def run_experiment(
    symbols: list[str],
    period: str = "train",
    strategy_path: str = "strategy.py",
    config: dict = None,
    description: str = "",
) -> dict:
    """
    Run a single experiment across specified symbols.

    Returns dict with:
        - results: {symbol: BacktestResult}
        - metrics: {symbol: dict}
        - avg_sharpe: float
        - status: "keep" | "discard" | "crash"
    """
    if config is None:
        config = load_config()

    results = {}
    all_metrics = {}
    total_time = 0

    for symbol in symbols:
        try:
            result = run_strategy_backtest(
                strategy_path=strategy_path,
                symbol=symbol,
                period=period,
                config=config,
            )
            metrics = compute_metrics(result)
            results[symbol] = result
            all_metrics[symbol] = metrics
            total_time += result.backtest_duration_s

            print_metrics(metrics, f"{symbol} {period.upper()}")

        except Exception as e:
            print(f"\nERROR running {symbol}: {e}")
            return {
                "results": results,
                "metrics": all_metrics,
                "avg_sharpe": 0.0,
                "status": "crash",
                "error": str(e),
            }

    # Compute average metrics across symbols
    if all_metrics:
        avg_sharpe = np.mean([m["sharpe_ratio"] for m in all_metrics.values()])
        avg_return = np.mean([m["total_return_pct"] for m in all_metrics.values()])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in all_metrics.values()])
    else:
        avg_sharpe = avg_return = avg_dd = 0.0

    print(f"\n{'=' * 55}")
    print(f"  AGGREGATE ({period.upper()})")
    print(f"{'=' * 55}")
    print(f"  Avg Sharpe:     {avg_sharpe:.3f}")
    print(f"  Avg Return:     {avg_return:+.2f}%")
    print(f"  Avg Max DD:     {avg_dd:.2f}%")
    print(f"  Total Time:     {total_time:.1f}s")
    print(f"{'=' * 55}")

    return {
        "results": results,
        "metrics": all_metrics,
        "avg_sharpe": avg_sharpe,
        "status": "keep",
    }


def log_results(
    experiment: dict,
    results_path: Path,
    description: str = "",
):
    """Append experiment results to results.tsv."""
    ensure_results_file(results_path)
    commit = get_git_commit()

    with open(results_path, "a") as f:
        for symbol, metrics in experiment["metrics"].items():
            result = experiment["results"][symbol]
            row = metrics_to_tsv_row(
                metrics=metrics,
                strategy_name=result.strategy_name,
                symbol=symbol,
                commit=commit,
                status=experiment["status"],
                description=description,
            )
            f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run trading strategy research experiment")
    parser.add_argument("--strategy", default="strategy.py", help="Strategy file path")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all)")
    parser.add_argument("--test", action="store_true", help="Run on test period")
    parser.add_argument("--full", action="store_true", help="Run both train and test")
    parser.add_argument("--description", "-d", default="", help="Experiment description")
    args = parser.parse_args()

    config = load_config()
    symbols = [args.symbol] if args.symbol else config["data"]["symbols"]
    results_path = Path(config["research"]["results_file"])

    periods = []
    if args.full:
        periods = ["train", "test"]
    elif args.test:
        periods = ["test"]
    else:
        periods = ["train"]

    for period in periods:
        print(f"\n{'#' * 60}")
        print(f"# Running experiment: {period.upper()} period")
        print(f"# Strategy: {args.strategy}")
        print(f"# Symbols: {', '.join(symbols)}")
        print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}")

        experiment = run_experiment(
            symbols=symbols,
            period=period,
            strategy_path=args.strategy,
            config=config,
            description=args.description,
        )

        # Log results
        desc = args.description or f"{period} run"
        log_results(experiment, results_path, description=desc)

        print(f"\nResults logged to {results_path}")


if __name__ == "__main__":
    main()
