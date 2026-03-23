#!/usr/bin/env python3
"""
run_systematic.py - Systematic combinatorial strategy search
Tests ALL indicator combinations. Results ALWAYS match saved files.
"""
import random
import time
from pathlib import Path

from strategy_generator import (
    TREND_INDICATORS, ENTRY_FILTERS, REGIME_FILTERS,
    generate_strategy, get_all_combos
)
from backtest import run_strategy_backtest
from evaluate import compute_metrics
from agent_research import append_results

STRATEGIES_DIR = Path("strategies")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def main():
    combos = get_all_combos()
    random.shuffle(combos)
    STRATEGIES_DIR.mkdir(exist_ok=True)

    tested = 0
    kept = 0

    print(f"Systematic search: {len(combos)} combinations")

    for trend_name, entry_name, regime_name, tf in combos:
        trend_info = TREND_INDICATORS[trend_name]
        entry_info = ENTRY_FILTERS[entry_name]
        regime_info = REGIME_FILTERS[regime_name]
        trend_params = {k: v[0] for k, v in trend_info["params"].items()}
        entry_params = {k: v[0] for k, v in entry_info["params"].items()}
        regime_params = {k: v[0] for k, v in regime_info["params"].items()}

        name = f"gen_{trend_name}_{entry_name}_{regime_name}_{tf}_v1"
        
        # Skip if already saved and tested
        saved_path = STRATEGIES_DIR / f"{name}.py"
        if saved_path.exists():
            continue

        try:
            code = generate_strategy(
                trend_name, entry_name, regime_name, tf, 0.25,
                trend_params, entry_params, regime_params
            )
        except Exception:
            continue

        # SAVE FIRST, then test from saved file — guarantees consistency
        saved_path.write_text(code)
        
        tested += 1
        t0 = time.time()
        any_kept = False

        for sym in SYMBOLS:
            try:
                # Train from SAVED file
                r = run_strategy_backtest(str(saved_path), sym, "train")
                m = compute_metrics(r)
                sharpe = m["sharpe_ratio"]
                trades = m["num_trades"]
                
                m["strategy"] = name
                m["symbol"] = sym
                train_pass = sharpe > 0 and trades >= 5 and m["max_drawdown_pct"] > -50
                append_results([m], "keep" if train_pass else "discard", name, "train")

                if not train_pass:
                    continue

                # Test from SAME saved file
                rt = run_strategy_backtest(str(saved_path), sym, "test")
                mt = compute_metrics(rt)
                mt["strategy"] = name
                mt["symbol"] = sym
                test_pass = mt["sharpe_ratio"] > 0 and mt["num_trades"] >= 3
                append_results([mt], "keep" if test_pass else "discard", name, "test")

                if test_pass:
                    any_kept = True

            except Exception:
                continue

        dt = time.time() - t0
        if any_kept:
            kept += 1
            print(f"✓ #{tested:>4d} {name} ({dt:.1f}s)")
        else:
            # Remove strategy file if nothing kept
            if not any_kept and saved_path.exists():
                saved_path.unlink()
            if tested % 20 == 0:
                print(f"  #{tested:>4d} {trend_name}+{entry_name}+{regime_name} {tf} ({dt:.1f}s)")

        if tested % 100 == 0:
            print(f"\n  ── Progress: {tested}/{len(combos)}, {kept} kept ──\n")

    print(f"\nDONE: {tested} tested, {kept} kept")


if __name__ == "__main__":
    main()
