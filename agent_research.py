#!/usr/bin/env python3
"""
agent_research.py - LLM-Driven Autonomous Research Loop
=========================================================
The LLM agent reads program.md, edits strategy.py, runs backtests,
evaluates results, and iterates forever - just like autoresearch.

Usage:
    python agent_research.py                  # Run indefinitely
    python agent_research.py --max 20         # Max 20 experiments
    python agent_research.py --provider gemini
    python agent_research.py --symbol BTCUSDT # Single symbol (faster)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from llm_client import LLMClient
from evaluate import TSV_HEADER, compute_metrics, metrics_to_tsv_row
from backtest import run_strategy_backtest
from prepare import load_config


STRATEGY_FILE = Path("strategy.py")
RESULTS_FILE = Path("results.tsv")
STRATEGIES_DIR = Path("strategies")
DOCS_DIR = Path("docs/strategies")


def read_file(path: str) -> str:
    return Path(path).read_text()


def write_strategy(code: str):
    STRATEGY_FILE.write_text(code)


BACKTEST_TIMEOUT_S = 90  # Hard kill after 90s per symbol


def _run_single_backtest(queue, strategy_path, symbol, period):
    """Worker function for subprocess backtest."""
    try:
        result = run_strategy_backtest(
            strategy_path=strategy_path,
            symbol=symbol,
            period=period,
        )
        m = compute_metrics(result)
        m["symbol"] = symbol
        m["strategy"] = result.strategy_name
        queue.put(("ok", m))
    except Exception as e:
        queue.put(("error", str(e)))


class EarlyDiscardError(Exception):
    """Raised when first symbol already fails — skip remaining symbols."""
    def __init__(self, msg, partial_results=None):
        super().__init__(msg)
        self.partial_results = partial_results or []


def run_backtest_all(symbols: list[str], strategy_path: str, period: str = "train",
                     early_discard: bool = True) -> list[dict]:
    """Run backtest on all symbols. Early exit if first symbol Sharpe < 0 or 0 trades."""
    from multiprocessing import Process, Queue

    results = []
    for symbol in symbols:
        q = Queue()
        p = Process(target=_run_single_backtest, args=(q, strategy_path, symbol, period))
        p.start()
        p.join(timeout=BACKTEST_TIMEOUT_S)

        if p.is_alive():
            p.kill()
            p.join(timeout=5)
            raise TimeoutError(f"{symbol} backtest killed after {BACKTEST_TIMEOUT_S}s")

        if q.empty():
            raise RuntimeError(f"{symbol} backtest returned no result")

        status, data = q.get_nowait()
        if status == "error":
            raise RuntimeError(data)
        results.append(data)

        # Early discard: if this symbol has Sharpe < 0 or 0 trades, skip the rest
        if early_discard:
            sharpe = data.get("sharpe_ratio", 0)
            trades = data.get("num_trades", 0)
            dd = data.get("max_drawdown_pct", 0)
            if sharpe < 0 or trades < 5 or dd < -50:
                raise EarlyDiscardError(
                    f"{symbol} Sharpe={sharpe:.3f} trades={trades} DD={dd:.1f}% — skip remaining",
                    partial_results=results
                )

    return results


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def git_commit(message: str):
    subprocess.run(["git", "add", "strategy.py"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)


def git_revert_strategy():
    """Revert strategy.py to last commit."""
    subprocess.run(["git", "checkout", "HEAD", "--", "strategy.py"], check=True)


def append_results(results: list[dict], status: str, description: str, period: str = "train"):
    """Append experiment results to results.tsv."""
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(TSV_HEADER + "\n")

    commit = get_git_commit()
    with open(RESULTS_FILE, "a") as f:
        for m in results:
            row = metrics_to_tsv_row(
                metrics=m,
                strategy_name=m.get("strategy", "unknown"),
                symbol=m["symbol"],
                commit=commit,
                status=status,
                description=description[:80],
                period=period,
            )
            f.write(row + "\n")


def save_strategy(strategy_name: str):
    """Copy current strategy.py to strategies/ dir."""
    STRATEGIES_DIR.mkdir(exist_ok=True)
    dest = STRATEGIES_DIR / f"{strategy_name}.py"
    shutil.copy(STRATEGY_FILE, dest)
    print(f"  Saved strategy to {dest}")


def build_system_prompt() -> str:
    # Load strategy rules file for LLM to follow
    rules_content = ""
    rules_path = Path("STRATEGY_RULES.md")
    if rules_path.exists():
        rules_content = rules_path.read_text()

    return f"""You are an expert quantitative trading researcher implementing strategies for
BTC/ETH/SOL USDT-M perpetual futures on Binance.

=== STRATEGY CODE RULES (READ CAREFULLY — violations = auto-reject) ===
{rules_content}
=== END RULES ===

CRITICAL RULES:
1. NO LOOK-AHEAD: At index i, ONLY use prices.iloc[:i+1]. NEVER .shift(-n).
2. Return signals as np.ndarray, same length as prices, values in [-1.0, 1.0]
3. Output ONLY valid Python code for strategy.py — no markdown, no explanation
4. Start with #!/usr/bin/env python3

HARD LIMITS (auto-reject if violated):
- EACH symbol must have Sharpe > 0 AND trades >= 5 on train
- EACH symbol must have Sharpe > 0 AND trades >= 3 on test
- Max drawdown > -50% per symbol
- 0 trades = ALWAYS discard (Sharpe=0.000 is NOT positive)

##################################################################
# CRITICAL MARKET ANALYSIS — LEARNED FROM 200+ FAILED EXPERIMENTS
##################################################################

BTC 2021-2024 (train): +219% but includes 2022 crash (-77%).
BTC 2025+ (test): -25% (bear/range market).
ETH follows similar pattern. SOL is an outlier.

WHAT FAILED (don't repeat):
- Simple EMA crossover (any period): ALWAYS negative Sharpe on BTC/ETH
- Trend following with short: 2022 whipsaw at bottom destroys gains
- Pure long-only: works on train but fails test (2025 is bearish)
- Strategies only working on SOL: SOL is biased (100x rally)

WHAT MIGHT WORK:
1. REGIME-ADAPTIVE: detect bull/bear/range, different logic per regime
2. VOLATILITY TARGETING: reduce size in high vol (skip 2022 crash)
3. ASYMMETRIC: bigger longs in clear bull, tiny or no shorts in unclear
4. MEAN REVERSION in range-bound periods (2025 test)
5. VERY FEW TRADES on daily/12h (minimize 0.10% cost impact)
6. Combine trend + mean reversion with regime switch
7. Use Bollinger BandWidth percentile as regime detector

RISK MANAGEMENT (MANDATORY):
- Every position MUST have stoploss: signal → 0 when price moves > 2-3*ATR
- Fewer trades = less fee drag. Target 20-50 trades/year, not 200+.

COST MODEL (engine-enforced):
- 0.04% taker fee + 0.01% slippage per side = 0.10% round trip
- Funding rate every 8h on open positions
- Signal at bar t → fill at bar t+1 open
- EVERY signal change triggers a trade and costs fees!

##################################################################
# CRITICAL: POSITION SIZING IS THE #1 FACTOR CONTROLLING DRAWDOWN
##################################################################

The signal value IS the position size. signal=1.0 means 100% long.
BTC dropped 77% in 2022 (69K→16K). With signal=1.0, that's -77% equity.
With signal=0.35, it's only -27% equity.

RULES FOR SIGNAL VALUES:
- Signal = position size. signal=0.35 means 35% of capital.
- MAX signal magnitude: 0.40 — NEVER use 1.0 (that's 100% and will blow up)
- Typical range: 0.20 to 0.35
- Use DISCRETE levels (0.0, ±0.20, ±0.35) to avoid churning costs
- Each signal change costs 0.10% fees on the CHANGE amount
- STOPLOSS: set signal=0 when price moves 2*ATR(14) against position
- TAKE PROFIT: reduce signal to half at 2R profit, trail stop

CURRENT BEST: mtf_hma_rsi_zscore_v1 → Sharpe=5.4, DD=-7.5%, Return=+2871%
This uses 4h HMA trend + 1h RSI pullback entries + Z-score filter.
You must BEAT this. Try combining different signal types.

strategy.py MUST contain:
- name: str
- timeframe: str (one of: "5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d")
- leverage: float (USE 1.0)
- generate_signals(prices: pd.DataFrame) -> np.ndarray

STRATEGY KNOWLEDGE:
- TREND: Supertrend(ATR=10,mult=3), HMA(16/48), KAMA(ER=10), Donchian(20), DEMA(8/21)
- MEAN REVERSION: Bollinger squeeze, RSI(14) extremes + SMA(200) filter, Z-score(20)
- MOMENTUM: MACD(12,26,9) histogram, ROC(10)+RSI(14)
- MULTI-TF: 4h trend + 1h entries (proven to 2x Sharpe)
- REGIME: Bollinger BW percentile detection
- RISK: ATR trailing stop = signal→0 when price < highest - 3*ATR

MULTI-TIMEFRAME — USE mtf_data.py HELPER (mandatory for all MTF strategies):
NEVER resample data yourself! NEVER use pd.date_range()!
Use the mtf_data module which loads ACTUAL Binance HTF candles:
```
from mtf_data import get_htf_data, align_htf_to_ltf
df_4h = get_htf_data(prices, '4h')  # loads actual Binance 4h parquet — call ONCE before the loop!
ema_4h = pd.Series(df_4h['close'].values).ewm(span=21).mean().values
ema_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1) for completed bars only
# NEVER call get_htf_data() inside a for-loop! It loads a Parquet file each call = 45K file reads!
```
This ensures: correct 4h boundaries, no look-ahead (shift by 1 HTF bar),
works on SOLUSDT with data gaps. 46 strategies failed audit without this.
SOLUSDT has 2 data gaps of ~3 days. Synthetic date_range misaligns after gaps."""


def build_experiment_prompt(
    program: str,
    current_strategy: str,
    history: list[dict],
    experiment_num: int,
) -> str:
    # Summarize history with failure reasons
    history_str = ""
    if history:
        history_str = "\n\nEXPERIMENT HISTORY (recent — learn from failures!):\n"
        for h in history[-12:]:
            history_str += (
                f"  #{h['num']:03d} [{h['status']:>7s}] {h['name']} | "
                f"Sharpe={h['avg_sharpe']:.3f} | Return={h['avg_return']:+.1f}% | "
                f"{h['description']}\n"
            )

    best = max(history, key=lambda x: x["avg_sharpe"]) if history else None
    best_str = ""
    if best and best["avg_sharpe"] > 0:
        best_str = f"\nCURRENT BEST: {best['name']} | Sharpe={best['avg_sharpe']:.3f} | Return={best['avg_return']:+.1f}%"
        best_str += "\nYou MUST beat this Sharpe to keep your strategy."

    # Determine which phase and what to try next
    phase_hint = ""
    n = experiment_num
    # Rotate through ALL timeframes equally — 6 TF groups, cycle every 36 experiments
    tf_groups = [
        ("15m", "1h/4h"),
        ("30m", "4h/1d"),
        ("1h", "4h/12h/1d"),
        ("4h", "1d/1w"),
        ("12h", "1d/1w"),
        ("1d", "single TF"),
    ]
    group_idx = (n - 1) % len(tf_groups)
    primary_tf, htf_options = tf_groups[group_idx]

    phase_hint = f"""
THIS EXPERIMENT: Primary timeframe = {primary_tf}, HTF options = {htf_options}
You MUST use timeframe = "{primary_tf}" for this experiment.

Timeframe rotation: each experiment uses a DIFFERENT primary TF.
Available data: 15m, 30m, 1h, 4h, 6h, 12h, 1d (all real Binance data).
For MTF: use mtf_data.get_htf_data(prices, '{htf_options.split("/")[0]}') ONCE before loop.

Strategy ideas for {primary_tf}:
- Supertrend + RSI pullback + ADX filter
- HMA/KAMA crossover + Bollinger BW regime
- MACD histogram + volume confirmation + ATR stop
- Donchian breakout + trend filter from HTF ({htf_options})
- EMA crossover + Z-score mean reversion filter
- Ensemble: 2-3 indicators vote, majority wins

Position sizing: 0.20-0.30, discrete levels, stoploss at 2*ATR.
REMEMBER: call get_htf_data() ONCE before loop, use aligned arrays inside."""
    # Gather failed approaches to avoid repeating
    failed_approaches = set()
    if history:
        for h in history:
            if h["status"] in ("discard", "crash"):
                failed_approaches.add(h["name"])
    avoid_str = ""
    if failed_approaches:
        avoid_str = f"\n\nALREADY TRIED AND FAILED ({len(failed_approaches)} strategies) — try something DIFFERENT:\n"
        avoid_str += ", ".join(sorted(failed_approaches)[-15:])

    return f"""EXPERIMENT #{experiment_num:03d}
{phase_hint}

RULES:
- Train: 2021-2024 | Primary TF: 5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d | HTF ref: up to 1w
- Signal bar t → fill bar t+1 | Costs: 0.10% round trip + funding
- REJECT if: DD < -50% | trades < 10 | Sharpe ≤ 0

#############################################################
# POSITION SIZING IS CRITICAL — THIS IS THE #1 LESSON LEARNED
#############################################################
- Signal value = position size. signal=1.0 means 100% of capital.
- BTC crashed 77% in 2022. signal=1.0 → you lose 77%. signal=0.35 → you lose 27%.
- MAX signal magnitude: 0.40 (absolute max). Normal range: 0.20-0.35.
- Use DISCRETE levels (0.0, ±0.20, ±0.35) — every signal change costs 0.10% fees!
- The current baseline (EMA crossover, size=0.35) has Sharpe=0.33, DD=-40%.
- USE leverage=1.0 (no leverage).

CURRENT strategy.py:
```python
{current_strategy}
```
{history_str}{best_str}{avoid_str}

INSTRUCTIONS:
1. State your hypothesis in a comment at the top (which strategy, timeframe, why)
2. Implement using REAL indicator formulas from quantitative trading literature
3. Use conservative leverage (1.0-2.0x) and keep drawdown under control
4. MULTI-TIMEFRAME: MUST use mtf_data helper (from mtf_data import get_htf_data, align_htf_to_ltf)
   df_4h = get_htf_data(prices, '4h'); vals = your_indicator(df_4h); aligned = align_htf_to_ltf(prices, df_4h, vals)
   NEVER resample yourself! NEVER pd.date_range()!
5. CRITICAL: Use proper min_periods on all rolling calculations

OUTPUT: Complete strategy.py code only. Start with #!/usr/bin/env python3"""


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Strip markdown fences if present
    response = response.strip()
    if "```python" in response:
        response = response.split("```python", 1)[1]
        response = response.split("```", 1)[0]
    elif "```" in response:
        response = response.split("```", 1)[1]
        response = response.split("```", 1)[0]
    # Strip shebang if doubled
    lines = response.strip().splitlines()
    if lines and lines[0].startswith("#!"):
        pass  # keep it
    return response.strip()


def validate_strategy(code: str) -> tuple[bool, str]:
    """Basic validation before running backtest."""
    required = ["name", "timeframe", "leverage", "def generate_signals"]
    for req in required:
        if req not in code:
            return False, f"Missing required: {req}"

    # Reject 1m timeframe
    if re.search(r'timeframe\s*=\s*["\']1m["\']', code):
        return False, "1m timeframe not allowed (too noisy/risky)"

    # CRITICAL: Reject synthetic date_range for MTF resampling
    # This causes alignment bugs and subtle look-ahead on gappy data (SOLUSDT)
    if re.search(r"pd\.date_range\s*\(\s*start\s*=\s*['\"]2021", code):
        return False, "FORBIDDEN: pd.date_range('2021-...') creates fake timestamps. Use prices['open_time'] as index for resampling."
    if re.search(r"date_range\s*\(\s*start\s*=\s*['\"]202", code):
        return False, "FORBIDDEN: synthetic date_range for resampling. Use actual open_time column."

    # Check for obvious look-ahead patterns
    bad_patterns = [
        (r"\.shift\s*\(\s*-[1-9]", "negative shift"),
        (r"prices\.iloc\[i\+", "future indexing"),
        (r"prices\[i\+", "future indexing"),
    ]
    for pattern, desc in bad_patterns:
        if re.search(pattern, code):
            return False, f"Look-ahead: {desc}"

    # CRITICAL: Block manual positional MTF resampling (i // N pattern)
    # This creates look-ahead because it reads unclosed HTF bars
    if re.search(r'//\s*bars_per_', code) or re.search(r'i\s*//\s*\d+\s*\]', code):
        if 'get_htf_data' not in code and 'mtf_data' not in code:
            return False, "Manual MTF resampling (i//N) detected. Use mtf_data.get_htf_data() instead."

    # Block .resample() without mtf_data
    if re.search(r'\.resample\s*\(', code):
        if 'get_htf_data' not in code and 'mtf_data' not in code:
            return False, "Manual .resample() detected. Use mtf_data.get_htf_data() instead."

    # PERFORMANCE + CORRECTNESS: Block get_htf_data inside loops (AST-based)
    try:
        import ast as _ast
        tree = _ast.parse(code)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.For, _ast.While)):
                for child in _ast.walk(node):
                    if isinstance(child, _ast.Call):
                        func = child.func
                        fname = ""
                        if isinstance(func, _ast.Name):
                            fname = func.id
                        elif isinstance(func, _ast.Attribute):
                            fname = func.attr
                        if fname in ('get_htf_data', 'load_klines', 'read_parquet'):
                            return False, f"{fname}() inside loop! Call ONCE before loop, use aligned arrays inside."
    except SyntaxError:
        pass

    # Also catch i//N MTF pattern via regex (simpler, catches idx_4h = i // 16)
    for line in code.split('\n'):
        s = line.strip()
        if re.search(r'i\s*//\s*\d+\s*[,\]]', s) or re.search(r'idx.*=.*i\s*//\s*\d+', s):
            if 'period' not in s and 'half' not in s and 'sqrt' not in s:
                return False, f"Manual MTF index (i//N) detected: {s[:60]}. Use align_htf_to_ltf() instead."

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="LLM Autonomous Research Loop")
    parser.add_argument("--max", type=int, default=999999, help="Max experiments (default: unlimited)")
    parser.add_argument("--provider", default=None, help="LLM provider (openai/anthropic/gemini)")
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config()
    symbols = args.symbols or config["data"]["symbols"]

    # Init LLM client
    print(f"[{datetime.now():%H:%M:%S}] Initializing LLM client...")
    client = LLMClient(provider=args.provider)
    print(f"  Provider: {client.provider} | Model: {client._get_model()}")
    print(f"  Symbols: {symbols}")
    print()

    # Ensure results file exists
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(TSV_HEADER + "\n")

    # Load program
    program = read_file("program.md")
    system_prompt = build_system_prompt()

    history = []
    best_strategy_code = STRATEGY_FILE.read_text()

    # Initialize best_sharpe from existing results (don't start from -999)
    best_sharpe = 0.0  # Minimum bar: must beat Sharpe=0 (better than doing nothing)
    if RESULTS_FILE.exists():
        try:
            import pandas as _pd
            df = _pd.read_csv(RESULTS_FILE, sep="\t")
            kept = df[df["status"] == "keep"]
            if len(kept) > 0 and "sharpe" in kept.columns:
                prev_best = kept.groupby("strategy")["sharpe"].mean().max()
                if prev_best > best_sharpe:
                    best_sharpe = float(prev_best)
                    print(f"  Resuming from previous best Sharpe: {best_sharpe:.3f}")
        except Exception:
            pass

    print("=" * 60)
    print("AUTONOMOUS RESEARCH LOOP STARTED")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    for experiment_num in range(1, args.max + 1):
        print(f"\n{'─' * 60}")
        print(f"[{datetime.now():%H:%M:%S}] EXPERIMENT #{experiment_num:03d}")
        print(f"{'─' * 60}")

        # --- Step 1: Generate new strategy ---
        print("  [1/4] Generating strategy with LLM...")
        current_strategy = STRATEGY_FILE.read_text()

        prompt = build_experiment_prompt(
            program=program,
            current_strategy=current_strategy,
            history=history,
            experiment_num=experiment_num,
        )

        try:
            response = client.chat(prompt, system=system_prompt, temperature=0.8)
            new_code = extract_code(response)
        except Exception as e:
            print(f"  LLM error: {e}")
            time.sleep(5)
            continue

        # --- Step 2: Validate ---
        valid, reason = validate_strategy(new_code)
        if not valid:
            print(f"  [SKIP] Invalid strategy: {reason}")
            history.append({
                "num": experiment_num, "name": "invalid", "status": "crash",
                "avg_sharpe": -999, "avg_return": 0, "description": reason,
            })
            continue

        # Extract strategy name for display
        name_match = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', new_code, re.MULTILINE)
        strategy_name = name_match.group(1) if name_match else f"strategy_{experiment_num:03d}"
        print(f"  Strategy: {strategy_name}")

        # Write and commit
        write_strategy(new_code)
        try:
            git_commit(f"exp#{experiment_num:03d}: {strategy_name}")
        except Exception:
            pass  # git commit failure is non-fatal

        # --- Step 3: Backtest ---
        print("  [2/4] Running backtest on train data...")
        try:
            bt_results = run_backtest_all(symbols, str(STRATEGY_FILE), period="train", early_discard=True)
        except EarlyDiscardError as e:
            print(f"  [EARLY DISCARD] {e}")
            git_revert_strategy()
            STRATEGY_FILE.write_text(best_strategy_code)
            description = f"exp#{experiment_num:03d} {strategy_name} early:{str(e)[:40]}"
            # Log partial results to results.tsv so dashboard shows activity
            append_results(e.partial_results, "discard", description, period="train")
            history.append({
                "num": experiment_num, "name": strategy_name, "status": "discard",
                "avg_sharpe": -999, "avg_return": 0, "description": f"early: {str(e)[:50]}",
            })
            continue
        except Exception as e:
            print(f"  [CRASH] Backtest error: {e}")
            git_revert_strategy()
            history.append({
                "num": experiment_num, "name": strategy_name, "status": "crash",
                "avg_sharpe": -999, "avg_return": 0, "description": str(e)[:60],
            })
            append_results([], "crash", str(e)[:60])
            continue

        # --- Step 3b: PREFIX LOOK-AHEAD TEST ---
        # Run signals on partial data and verify they match full data signals
        print("  [2b/4] Running look-ahead prefix test...")
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("_strat_test", str(STRATEGY_FILE))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)

            from prepare import load_klines, load_config as _lc
            _cfg = _lc()
            _test_sym = symbols[0]  # test on first symbol
            _prices = load_klines(_test_sym, _mod.timeframe)
            import pandas as _pd
            _train_end = _pd.Timestamp(_cfg["data"]["train_end"], tz="UTC")
            _prices = _prices[_prices["open_time"] <= _train_end].reset_index(drop=True)

            _signals_full = _mod.generate_signals(_prices)
            # Test at 3 checkpoints
            _la_ok = True
            import numpy as _np
            for _cp in [1000, 2000, len(_prices) // 2]:
                if _cp >= len(_prices):
                    continue
                _signals_partial = _mod.generate_signals(_prices.iloc[:_cp].reset_index(drop=True))
                _diff = abs(float(_signals_partial[-1]) - float(_signals_full[_cp - 1]))
                if _diff > 0.01:
                    _la_ok = False
                    print(f"  [LOOKAHEAD FAIL] Signal diff={_diff:.4f} at checkpoint {_cp}")
                    break

            if not _la_ok:
                print(f"  [SKIP] Strategy FAILED prefix look-ahead test")
                git_revert_strategy()
                STRATEGY_FILE.write_text(best_strategy_code)
                description = f"exp#{experiment_num:03d} {strategy_name} [LOOKAHEAD_PREFIX_FAIL]"
                append_results(bt_results, "discard", description, period="train")
                history.append({
                    "num": experiment_num, "name": strategy_name, "status": "discard",
                    "avg_sharpe": -999, "avg_return": 0,
                    "description": "LOOKAHEAD prefix test failed",
                })
                continue
            else:
                print("  [OK] Prefix look-ahead test passed")
        except Exception as _e:
            print(f"  [WARN] Prefix test error (non-fatal): {_e}")

        # --- Step 4: Evaluate ---
        avg_sharpe = sum(r["sharpe_ratio"] for r in bt_results) / len(bt_results)
        avg_return = sum(r["total_return_pct"] for r in bt_results) / len(bt_results)
        avg_dd = sum(r["max_drawdown_pct"] for r in bt_results) / len(bt_results)
        avg_trades = sum(r["num_trades"] for r in bt_results) / len(bt_results)

        print(f"  [3/4] Results: Sharpe={avg_sharpe:.3f} | Return={avg_return:+.1f}% | DD={avg_dd:.1f}% | Trades={avg_trades:.0f}")

        # --- Step 5: Keep or discard ---
        # Quality gates
        MIN_TRADES = 10
        MAX_DD_THRESHOLD = -50.0  # reject strategies with > 50% drawdown

        reject_reason = None
        if avg_trades < MIN_TRADES:
            reject_reason = f"too few trades ({avg_trades:.0f}, min={MIN_TRADES})"
        elif avg_dd < MAX_DD_THRESHOLD:
            reject_reason = f"drawdown too deep ({avg_dd:.1f}%, max={MAX_DD_THRESHOLD}%)"

        if reject_reason:
            print(f"  [4/4] ✗ REJECT ({reject_reason})")
            git_revert_strategy()
            STRATEGY_FILE.write_text(best_strategy_code)
            description = f"exp#{experiment_num:03d} {strategy_name}"
            append_results(bt_results, "discard", description, period="train")
            history.append({
                "num": experiment_num, "name": strategy_name, "status": "discard",
                "avg_sharpe": avg_sharpe, "avg_return": avg_return,
                "description": reject_reason,
            })
            continue

        # Compute return/DD ratio (Calmar-like)
        return_dd_ratio = abs(avg_return / avg_dd) if avg_dd < -0.1 else avg_return
        improved_sharpe = avg_sharpe > best_sharpe
        # Keep ANY strategy with positive Sharpe and reasonable return/DD
        # EVERY symbol must be profitable WITH actual trades
        all_symbols_good = all(
            r["sharpe_ratio"] > 0 and r["num_trades"] >= 5
            for r in bt_results
        )
        is_good = avg_sharpe > 0.1 and avg_trades >= 10 and all_symbols_good
        status = "keep" if is_good else "discard"

        # Save ALL strategies that pass quality gates
        if is_good:
            save_strategy(strategy_name)

        if is_good:
            if improved_sharpe:
                print(f"  [4/4] ✓ KEEP+BEST (Sharpe +{avg_sharpe - best_sharpe:.3f} vs best {best_sharpe:.3f})")
                best_sharpe = avg_sharpe
                best_strategy_code = new_code
            else:
                print(f"  [4/4] ✓ KEEP (Sharpe={avg_sharpe:.3f}, Return/DD={return_dd_ratio:.1f})")

            # Run test — early discard if first symbol Sharpe < 0
            test_results = []
            try:
                test_results = run_backtest_all(symbols, str(STRATEGY_FILE), period="test", early_discard=True)
                test_sharpe = sum(r["sharpe_ratio"] for r in test_results) / len(test_results)
                test_return = sum(r["total_return_pct"] for r in test_results) / len(test_results)
                print(f"       Test: Sharpe={test_sharpe:.3f} | Return={test_return:+.1f}%")

                test_trades = sum(r["num_trades"] for r in test_results)
                test_all_good = all(r["sharpe_ratio"] > 0 and r["num_trades"] >= 3 for r in test_results)
                if test_sharpe < 0 or not test_all_good or test_trades < 10:
                    reason = "Sharpe<0" if test_sharpe < 0 else "0-trade symbols" if not test_all_good else "too few trades"
                    print(f"       DEMOTED: Test {reason}")
                    status = "discard"
                    git_revert_strategy()
                    STRATEGY_FILE.write_text(best_strategy_code)
            except EarlyDiscardError as e:
                print(f"       Test EARLY DISCARD: {e}")
                test_results = e.partial_results
                status = "discard"
                git_revert_strategy()
                STRATEGY_FILE.write_text(best_strategy_code)
            except Exception as e:
                print(f"       Test failed: {e}")
                status = "discard"
                git_revert_strategy()
                STRATEGY_FILE.write_text(best_strategy_code)

            # Save doc
            doc_path = DOCS_DIR / f"{strategy_name}.md"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            test_table = ""
            if test_results:
                test_table = "\n## Test Results (2025+)\n| Symbol | Sharpe | Return | Max DD | Trades |\n|--------|--------|--------|--------|--------|\n"
                test_table += "".join(
                    f"| {r['symbol']} | {r['sharpe_ratio']:.3f} | {r['total_return_pct']:+.1f}% | {r['max_drawdown_pct']:.1f}% | {r['num_trades']} |\n"
                    for r in test_results
                )
            doc_path.write_text(f"""# Strategy: {strategy_name}

## Status
ACTIVE - Sharpe={avg_sharpe:.3f} | Return={avg_return:+.1f}% | DD={avg_dd:.1f}%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
""" + "".join(
    f"| {r['symbol']} | {r['sharpe_ratio']:.3f} | {r['total_return_pct']:+.1f}% | {r['max_drawdown_pct']:.1f}% | {r['num_trades']} |\n"
    for r in bt_results
) + test_table + f"""
## Code
```python
{new_code}
```

## Last Updated
{datetime.now().strftime('%Y-%m-%d %H:%M')}
""")
        else:
            print(f"  [4/4] ✗ DISCARD (Sharpe {avg_sharpe:.3f} ≤ best {best_sharpe:.3f})")
            git_revert_strategy()
            STRATEGY_FILE.write_text(best_strategy_code)
            test_results = []

        # Log results
        description = f"exp#{experiment_num:03d} {strategy_name}"
        append_results(bt_results, status, description, period="train")
        if test_results:
            append_results(test_results, status, description, period="test")

        history.append({
            "num": experiment_num,
            "name": strategy_name,
            "status": status,
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
            "description": description,
        })

        # Progress summary every 10 experiments
        if experiment_num % 10 == 0:
            kept = sum(1 for h in history if h["status"] == "keep")
            print(f"\n  ── Summary: {experiment_num} experiments, {kept} kept, best Sharpe={best_sharpe:.3f} ──")

    print("\nResearch loop complete.")


if __name__ == "__main__":
    main()
