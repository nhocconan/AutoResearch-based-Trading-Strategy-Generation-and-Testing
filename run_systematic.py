#!/usr/bin/env python3
"""
run_systematic.py - Systematic combinatorial strategy search
=============================================================
Tests ALL indicator combinations instead of relying on LLM creativity.
Much more efficient: 960 combinations tested systematically.
"""
import random
import sys
import time
from pathlib import Path

from strategy_generator import (
    TREND_INDICATORS, ENTRY_FILTERS, REGIME_FILTERS,
    generate_strategy, get_all_combos
)
from backtest import run_strategy_backtest
from evaluate import compute_metrics, metrics_to_tsv_row, TSV_HEADER

RESULTS_FILE = Path("results.tsv")
STRATEGIES_DIR = Path("strategies")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_strategy(code, symbols=SYMBOLS):
    """Test a strategy on all symbols, train+test, per-symbol."""
    tmp = Path("/tmp/_sys_strategy.py")
    tmp.write_text(code)

    results = {"train": {}, "test": {}}
    kept_symbols = []

    for sym in symbols:
        # Train
        try:
            r = run_strategy_backtest(str(tmp), sym, "train")
            m = compute_metrics(r)
            results["train"][sym] = m
            if m["sharpe_ratio"] > 0 and m["num_trades"] >= 5:
                # Test
                try:
                    rt = run_strategy_backtest(str(tmp), sym, "test")
                    mt = compute_metrics(rt)
                    results["test"][sym] = mt
                    if mt["sharpe_ratio"] > 0 and mt["num_trades"] >= 3:
                        kept_symbols.append(sym)
                except:
                    pass
        except:
            pass

    return results, kept_symbols


def main():
    combos = get_all_combos()
    random.shuffle(combos)  # randomize order for diversity

    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(TSV_HEADER + "\n")

    STRATEGIES_DIR.mkdir(exist_ok=True)
    tested = 0
    kept = 0

    print(f"Systematic search: {len(combos)} combinations")
    print(f"{'='*60}")

    for trend_name, entry_name, regime_name, tf in combos:
        trend_info = TREND_INDICATORS[trend_name]
        entry_info = ENTRY_FILTERS[entry_name]
        regime_info = REGIME_FILTERS[regime_name]

        # Pick default params (first value of each)
        trend_params = {k: v[0] for k, v in trend_info["params"].items()}
        entry_params = {k: v[0] for k, v in entry_info["params"].items()}
        regime_params = {k: v[0] for k, v in regime_info["params"].items()}

        name = f"gen_{trend_name}_{entry_name}_{regime_name}_{tf}_v1"
        size = 0.25

        try:
            code = generate_strategy(
                trend_name, entry_name, regime_name, tf, size,
                trend_params, entry_params, regime_params
            )
        except Exception as e:
            continue

        tested += 1
        t0 = time.time()
        results, kept_symbols = test_strategy(code)
        dt = time.time() - t0

        # Log results
        for period in ["train", "test"]:
            for sym, m in results.get(period, {}).items():
                status = "keep" if sym in kept_symbols else "discard"
                row = metrics_to_tsv_row(m, name, sym, "systematic", status, name, period)
                with open(RESULTS_FILE, "a") as f:
                    f.write(row + "\n")

        if kept_symbols:
            kept += 1
            # Save strategy
            (STRATEGIES_DIR / f"{name}.py").write_text(code)
            print(f"✓ #{tested:>4d} {name}: KEPT for {kept_symbols} ({dt:.1f}s)")
        else:
            train_sharpes = {s: f"{m['sharpe_ratio']:+.3f}" for s, m in results.get("train", {}).items()}
            print(f"  #{tested:>4d} {trend_name}+{entry_name}+{regime_name} {tf}: {train_sharpes} ({dt:.1f}s)")

        if tested % 50 == 0:
            print(f"\n  ── Progress: {tested}/{len(combos)} tested, {kept} kept ──\n")

    print(f"\n{'='*60}")
    print(f"DONE: {tested} tested, {kept} kept")


if __name__ == "__main__":
    main()
