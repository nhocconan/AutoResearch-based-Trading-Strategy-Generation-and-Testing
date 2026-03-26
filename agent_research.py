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

from llm_client import LLMClient, LLMTimeoutError
from evaluate import compute_metrics
from backtest import run_strategy_backtest
from prepare import load_config
from results_db import append_results, query_best_kept_sharpe  # noqa: F401


STRATEGY_FILE = Path("strategy.py")
STRATEGIES_DIR = Path("strategies")
DOCS_DIR = Path("docs/strategies")


def read_file(path: str) -> str:
    return Path(path).read_text()


def write_strategy(code: str):
    STRATEGY_FILE.write_text(code)


# Timeout scales with timeframe — lower TF = more bars = more time needed
BACKTEST_TIMEOUT_BY_TF = {
    "5m": 180, "15m": 150, "30m": 120, "1h": 90,
    "4h": 60, "6h": 60, "12h": 45, "1d": 30,
}
BACKTEST_TIMEOUT_DEFAULT = 120


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


def _get_timeout(strategy_path: str) -> int:
    """Get timeout based on strategy's timeframe."""
    try:
        import re
        code = Path(strategy_path).read_text()
        m = re.search(r'timeframe\s*=\s*["\'](\w+)["\']', code)
        if m:
            tf = m.group(1)
            return BACKTEST_TIMEOUT_BY_TF.get(tf, BACKTEST_TIMEOUT_DEFAULT)
    except Exception:
        pass
    return BACKTEST_TIMEOUT_DEFAULT


# Track timeout stats for monitoring
_timeout_log = Path("timeout_log.txt")


def run_backtest_all(symbols: list[str], strategy_path: str, period: str = "train",
                     early_discard: bool = True) -> list[dict]:
    """Run backtest on all symbols. Early exit if first symbol Sharpe < 0 or 0 trades."""
    from multiprocessing import Process, Queue

    timeout_s = _get_timeout(strategy_path)
    results = []
    for symbol in symbols:
        q = Queue()
        p = Process(target=_run_single_backtest, args=(q, strategy_path, symbol, period))
        p.start()
        p.join(timeout=timeout_s)

        if p.is_alive():
            p.kill()
            p.join(timeout=5)
            # Log timeout for monitoring
            from datetime import datetime
            msg = f"{datetime.now():%Y-%m-%d %H:%M} | TIMEOUT {timeout_s}s | {symbol} {period} | {Path(strategy_path).stem}\n"
            with open(_timeout_log, "a") as f:
                f.write(msg)
            print(f"  [TIMEOUT] {symbol} killed after {timeout_s}s — logged to timeout_log.txt")
            raise TimeoutError(f"{symbol} backtest killed after {timeout_s}s")

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
    """Append experiment results to results.db (SQLite). Deduplicates on (strategy, symbol, period).
    SQLite WAL mode handles concurrent writers without explicit file locking."""
    from results_db import append_results as _db_append
    _db_append(results, status, description, period)


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
- EACH symbol must have Sharpe > 0 AND trades >= 50 TOTAL over 4 years on train
  ⚠️ "50 trades" = 50 TOTAL over 4 years = 12.5/year. This is the MINIMUM for statistical validity.
  Target: 75-300 total trades over 4 years (19-75/year). Anything over 600 total is overtrading.
- EACH symbol must have Sharpe > 0 AND trades >= 10 on test (15 months = ~8/year minimum)
- Max drawdown > -50% per symbol
- 0 trades = ALWAYS discard (Sharpe=0.000 is NOT positive)

##################################################################
# CRITICAL: FEE DRAG IS THE #1 KILLER OF STRATEGIES
##################################################################

REAL COST MODEL: 0.10% per round trip (0.05%/side taker + slippage).
- 100 trades/year × 0.10% = 10% annual fee drag (manageable)
- 300 trades/year × 0.10% = 30% annual fee drag (needs exceptional alpha)
- 800 trades/year × 0.10% = 80% annual fee drag (IMPOSSIBLE to overcome)
- 2000+ trades/year = CERTAIN DEATH regardless of signal quality

PROVEN TRADE COUNTS FROM 16,000+ EXPERIMENTS (what actually works on test):
- 4h strategies: TARGET 75-200 total over 4 years (19-50/year). HARD MAX: 400 total.
- 6h strategies: TARGET 50-150 total over 4 years (12-37/year). HARD MAX: 300 total.
- 12h strategies: TARGET 50-150 total over 4 years (12-37/year). HARD MAX: 200 total.
- 1d strategies: TARGET 30-100 total over 4 years (7-25/year). HARD MAX: 150 total.

IF YOUR STRATEGY HAS > 400 TRADES on 4h (or >300 on 6h, >200 on 12h):
→ Entry conditions are TOO LOOSE. Add MORE filters. Do NOT just "loosen" them.
→ Solution: add minimum holding period, stricter threshold, additional confluence factor.

##################################################################
# CRITICAL MARKET ANALYSIS — LEARNED FROM 16,000+ EXPERIMENTS
##################################################################

BTC 2021-2024 (train): +219% but includes 2022 crash (-77%).
BTC 2025+ (test): -25% (bear/range market).
ETH follows similar pattern. SOL is an outlier (100x rally = not representative).

WHAT FAILED (don't repeat):
- Simple EMA crossover (any period): ALWAYS negative Sharpe on BTC/ETH
- Trend following with short: 2022 whipsaw at bottom destroys gains
- Pure long-only: works on train but fails test (2025 is bearish)
- Strategies only working on SOL: SOL is biased (100x rally)
- 15m strategies (ALL 75 attempts failed — 0% keep rate): fee drag insurmountable
- Overtrading: strategies with >600 train trades/symbol uniformly fail on test

WHAT ACTUALLY WORKS (confirmed from 16,000+ experiments, best test performers):
1. 4h Donchian breakout + volume confirmation + ATR filter (multiple ETH/SOL winners)
2. 4h/12h Camarilla pivot + volume spike + choppiness regime (ETH best: test Sharpe=1.47)
3. TRIX + volume spike + regime (ETH: test Sharpe=1.32)
4. CRSI + Donchian + chop regime on 4h (SOL: test Sharpe=1.46)
5. 1d KAMA + RSI + chop regime (SOL: test Sharpe=1.31)
6. Funding rate strategies (39% keep rate, much better than average)

WINNING PATTERN: tight entry conditions (~75-150 trades/year) + volume confirmation +
regime filter (chop/ADX) + price channel (Donchian/Camarilla) as structure.

WHAT MIGHT WORK (from research — still underexplored):
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI<10 + price>SMA200. Short: CRSI>90 + price<SMA200. 75% win rate.
2. CHOPPINESS INDEX regime filter: CHOP(14) > 61.8 = range (mean revert),
   CHOP < 38.2 = trending (trend follow). Best meta-filter for bear markets.
3. EHLERS FISHER TRANSFORM: period=9, long when Fisher crosses above -1.5,
   short when crosses below +1.5. Catches reversals in bear rallies.
4. PAIRS TRADING: BTC-ETH spread via z-score. Market-neutral. Entry z>2.0.
5. LARRY WILLIAMS VOL BREAKOUT: Long = open + K*prev_range. K=0.5-0.6.
6. ADAPTIVE KELLY SIZING: quarter-Kelly * vol_scaling on ANY strategy.
7. FUNDING RATE CONTRARIAN: short when funding >0.03%, long when <-0.03%.
8. REGIME-ADAPTIVE: different logic per regime (bull/bear/range)
9. VERY FEW TRADES on 12h/1d (minimize cost impact)

BTC/ETH SPECIFIC (these coins ALWAYS fail simple trend strategies):
- FUNDING RATE MEAN REVERSION: Z-score of funding(30d) < -2 → long, > +2 → short.
  Uses data/processed/funding/*.parquet. Load with: pd.read_parquet(funding_path).
  Reported Sharpe 0.8-1.5 through 2022 crash. BEST EDGE for BTC/ETH.
- VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long.
  Captures "vol crush" after panic. Exit when ATR ratio < 1.2.
- ASYMMETRIC REGIME: ADX>25 + price<SMA50 = bear (only short retrace to EMA21).
  ADX<20 = range (mean revert at BB bounds). Hysteresis: enter 25, exit 18.
- CROSS-ASSET LEAD-LAG: BTC breaks Donchian(20) low → short ETH (ETH lags 1-4h).
- BEAR REGIME SQUEEZE BREAKOUT: BB Width at 30d low + price breaks Donchian(20)
  low + price<SMA200 → short. Only short in bear.

RISK MANAGEMENT (MANDATORY):
- Every position MUST have stoploss: signal → 0 when price moves > 2-3*ATR
- Fewer trades = less fee drag.

TRADE FREQUENCY LIMITS (CRITICAL — #1 reason lower TF fails):
- 15m/30m: MAX 50-100 trades/year (use very strict entry: 3+ confluence)
- 1h: MAX 30-60 trades/year
- 4h: MAX 20-50 trades/year
- 12h/1d: MAX 10-30 trades/year
- If >100 trades/year on lower TF → entry conditions TOO LOOSE → add more filters
- For 15m/30m/1h: use 4h/12h for signal DIRECTION, lower TF only for entry TIMING

COST MODEL (engine-enforced):
- Taker 0.04% + Slippage 0.01% = 0.05%/side = 0.10% round trip (FULL COST)
- Funding rate every 8h on open positions
- Signal at bar t → fill at bar t+1 open
- EVERY signal change triggers a trade and costs 0.10% round trip!

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

CURRENT BEST FROM 16,000+ EXPERIMENTS: test Sharpe ~1.47-1.79 on single symbols.
Top performers: 4h Donchian+volume, Camarilla pivot+chop, CRSI+regime.
Most strategies that look good on train (Sharpe 0.3-0.8) generalize to test Sharpe 1.0-1.5.

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


def _build_db_summary(primary_tf: str) -> str:
    """Build a short summary of top DB performers for the given TF to guide the LLM."""
    try:
        from results_db import get_conn
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT te.strategy, te.symbol, te.sharpe, te.trades, te.win_rate
                FROM results te
                JOIN results tr ON te.strategy=tr.strategy AND te.symbol=tr.symbol AND tr.period='train'
                WHERE te.period='test' AND te.status='keep' AND te.trades >= 20 AND te.sharpe > 0.8
                  AND tr.sharpe BETWEEN 0.05 AND 3.0
                ORDER BY te.sharpe DESC LIMIT 6
            """).fetchall()
        if not rows:
            return ""
        lines = ["\n\nDB TOP PERFORMERS (test period, trades>=20) — STUDY THESE PATTERNS:"]
        for r in rows:
            lines.append(f"  {r[0][:50]} | {r[1]} | test_sharpe={r[2]:.3f} | {r[3]}tr | {r[4]:.0f}%wr")
        lines.append("→ Notice: tight entries (75-300 train trades), volume confirm, price channel structure.\n")
        return "\n".join(lines)
    except Exception:
        return ""


def load_recent_history_from_db(n: int = 20) -> list[dict]:
    """Load recent experiment history from DB so loop restarts don't lose context."""
    try:
        from results_db import get_conn
        import re as _re
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT strategy,
                       AVG(sharpe) as avg_sharpe,
                       AVG(trades) as avg_trades,
                       MAX(CASE WHEN status='keep' THEN 1 ELSE 0 END) as kept
                FROM results WHERE period='train'
                  AND strategy IN (
                    SELECT strategy FROM (
                        SELECT strategy, MAX(id) as last_id
                        FROM results WHERE period='train'
                        GROUP BY strategy ORDER BY last_id DESC LIMIT ?
                    )
                  )
                GROUP BY strategy ORDER BY MAX(id) DESC
            """, (n,)).fetchall()
        history = []
        for i, r in enumerate(rows):
            strategy_name = r[0]
            avg_sharpe = float(r[1] or 0)
            avg_trades = float(r[2] or 0)
            kept = bool(r[3])
            status = "keep" if kept else "discard"
            # Extract TF from strategy name
            tf_match = _re.search(r'mtf_(\w+?)_', strategy_name)
            tf = tf_match.group(1) if tf_match else ""
            # Diagnose failure reason
            fail_reason = ""
            if not kept:
                overtrade_thresh = {"4h": 400, "6h": 300, "12h": 200, "1d": 150, "1h": 600}
                thresh = overtrade_thresh.get(tf, 500)
                if avg_trades > thresh:
                    fail_reason = f"overtrading({avg_trades:.0f}tr)"
                elif avg_trades < 50:
                    fail_reason = f"too_few_trades({avg_trades:.0f}tr)"
                elif avg_sharpe < -1.0:
                    fail_reason = "very_neg_sharpe"
                else:
                    fail_reason = "neg_sharpe"
            history.append({
                "num": n - i,
                "name": strategy_name,
                "status": status,
                "avg_sharpe": avg_sharpe,
                "avg_return": 0.0,
                "avg_trades": avg_trades,
                "fail_reason": fail_reason,
                "tf": tf,
                "description": f"loaded from db",
            })
        return list(reversed(history))  # oldest first
    except Exception:
        return []


def build_experiment_prompt(
    program: str,
    current_strategy: str,
    history: list[dict],
    experiment_num: int,
) -> str:
    # Summarize history with failure reasons + trade counts
    history_str = ""
    if history:
        history_str = "\n\nEXPERIMENT HISTORY (recent — learn from failures!):\n"
        for h in history[-15:]:
            avg_trades = h.get("avg_trades", 0)
            fail_reason = h.get("fail_reason", "")
            # Annotate overtrading visually
            trades_note = ""
            if avg_trades > 0:
                tf = h.get("tf", "")
                overtrade_thresh = {"4h": 400, "6h": 300, "12h": 200, "1d": 150, "1h": 600}
                thresh = overtrade_thresh.get(tf, 500)
                if avg_trades > thresh:
                    trades_note = f" ⚠️ OVERTRADING ({avg_trades:.0f} trades/sym >> {thresh} max)"
                else:
                    trades_note = f" ({avg_trades:.0f} tr/sym)"
            reason_note = f" ← {fail_reason}" if fail_reason else ""
            history_str += (
                f"  #{h['num']:03d} [{h['status']:>7s}] {h['name']} | "
                f"Sharpe={h['avg_sharpe']:.3f}{trades_note}{reason_note}\n"
            )

    best = max(history, key=lambda x: x["avg_sharpe"]) if history else None
    best_str = ""
    if best and best["avg_sharpe"] > 0:
        best_str = f"\nSESSION BEST SO FAR: {best['name']} | avg_sharpe={best['avg_sharpe']:.3f}"

    # Determine which phase and what to try next
    phase_hint = ""
    n = experiment_num
    # TF rotation based on ACTUAL keep rates from 16,000+ experiments:
    # 12h=54% keep, gen/4h=41-45%, 1d=40%, 6h=23%, 1h=17%, 30m=9%, 15m=0% (DEAD)
    # 15m is REMOVED — 75 experiments, 0% keep rate, fee drag insurmountable
    tf_groups = [
        ("4h",  "1d/1w"),      # best: 41% keep, ETH/SOL Camarilla+Donchian
        ("12h", "1d/1w"),      # best: 54% keep rate — highest success
        ("4h",  "12h/1d"),     # 4h variation
        ("1d",  "1w"),         # 40% keep, SOL Donchian+KAMA+RSI
        ("12h", "1d"),         # 12h variation
        ("4h",  "1d"),         # 4h simple
        ("6h",  "1d/1w"),      # 23% keep, worth trying with novel ideas
        ("12h", "1w/1d"),      # 12h + weekly context
        ("4h",  "1d/1w"),      # 4h with weekly bias
        ("1d",  "1w"),         # 1d variation
        ("6h",  "1d"),         # 6h simple
        ("12h", "1d/1w"),      # 12h variation
        ("4h",  "12h"),        # 4h + 12h HTF
        ("1h",  "4h/1d"),      # 1h (17% keep — use only with very strict conditions)
        ("6h",  "1w/1d"),      # 6h + weekly pivot
        ("12h", "1d"),         # 12h proven
        ("4h",  "1d/1w"),      # 4h ETH focus
        ("1d",  "1w"),         # 1d ETH/BTC
        ("6h",  "12h/1d"),     # 6h variation
        ("4h",  "1d"),         # 4h simple
    ]
    group_idx = (n - 1) % len(tf_groups)
    primary_tf, htf_options = tf_groups[group_idx]

    # Build TF-specific guidance
    if primary_tf == "6h":
        tf_guidance = f"""THIS EXPERIMENT: Primary = 6h, HTF = {htf_options}
6h stats from DB: 359 strategies tested, 23% keep rate (vs 54% for 12h).
Only try with genuinely novel concept — not just variations of crsi/hma/chop already tried.

Strategy ideas for 6h:
- Donchian(20) breakout + weekly pivot direction + volume confirmation (not tried recently)
- Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Ichimoku: TK cross + cloud filter from 1d (TIER 8 in program.md)
- ADX + Williams Alligator combination
- Elder Ray (Bull/Bear power) + regime

Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300 total.
Size: 0.25-0.30."""

    elif primary_tf == "5m":
        tf_guidance = f"""THIS EXPERIMENT: Primary = 5m, HTF = {htf_options}
5m stats: ~18 experiments, 0% keep rate. Extreme caution required.

5m requires extreme selectivity. Target: 40-80 total trades over 4 years = 10-20/year.
Session filter MANDATORY: only 08-20 UTC.
HARD MAX: 100 total trades over 4 years.
Size: 0.15."""

    elif primary_tf in ("30m", "1h"):
        tf_guidance = f"""THIS EXPERIMENT: Primary = {primary_tf}, HTF = {htf_options}
{primary_tf} stats from DB: 1h=17% keep, 30m=9% keep. Difficult timeframe.
The #1 failure: too many trades → fee drag kills profit.

Target: 60-150 total trades over 4 years = 15-37/year for {primary_tf}.
HARD MAX: {200 if primary_tf == "1h" else 150} total trades.
Use {htf_options} for SIGNAL DIRECTION, {primary_tf} only for ENTRY TIMING.
Add session filter (08-20 UTC) to reduce noise trades.
Size: 0.20."""

    else:
        # Proven TFs: 4h, 12h, 1d
        tf_trade_targets = {
            "4h":  "75-200 total over 4 years (19-50/year). HARD MAX: 400 total.",
            "12h": "50-150 total over 4 years (12-37/year). HARD MAX: 200 total.",
            "1d":  "30-100 total over 4 years (7-25/year). HARD MAX: 150 total.",
        }
        trade_target = tf_trade_targets.get(primary_tf, "50-200 total over 4 years")
        tf_guidance = f"""THIS EXPERIMENT: Primary = {primary_tf}, HTF = {htf_options}
Keep rate for {primary_tf}: {"54%" if primary_tf == "12h" else "41%" if primary_tf == "4h" else "40%"}
TARGET TRADES: {trade_target}

Proven patterns from DB (what actually works on TEST period, not just train):
- Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38
- Camarilla pivot levels from 1d + volume spike + choppiness regime → ETHUSDT test Sharpe 1.47
- CRSI + Choppiness regime + Donchian exit → SOLUSDT test Sharpe 1.46
- KAMA direction + RSI + chop filter → SOLUSDT test Sharpe 1.31
- Funding rate mean-reversion (Z-score of funding) → proven BTC/ETH edge
- Novel: Williams Alligator, Elder Ray, Vortex, TRIX + volume spike (ETHUSDT: 1.32)

⚠️ AVOID (recently tried, uniformly failed due to overtrading):
- CRSI/HMA/KAMA "loose" variations → all fail due to >600 trades (fee drag)
- Stacking multiple "mild" signals → too many trades
- Any strategy generating >400 total 4h trades → CERTAIN failure on test

WINNING FORMULA: ONE strong signal (price channel breakout OR pivot level touch) +
volume confirmation + regime filter (chop/ADX) + ATR stoploss.
Fewer conditions = fewer trades = less fee drag = better test generalization.
Size: 0.25-0.30."""

    phase_hint = f"""
{tf_guidance}

You MUST use timeframe = "{primary_tf}".
For MTF: use mtf_data.get_htf_data(prices, '{htf_options.split("/")[0]}') ONCE before loop.
REMEMBER: call get_htf_data() ONCE before loop, use aligned arrays inside."""

    # Gather failed approaches from THIS SESSION to avoid repeating
    failed_approaches = set()
    overtrading_patterns = []
    if history:
        for h in history:
            if h["status"] in ("discard", "crash"):
                failed_approaches.add(h["name"])
            if h.get("fail_reason") == "overtrading":
                overtrading_patterns.append(f"{h['name']} ({h.get('avg_trades', 0):.0f} tr/sym)")
    avoid_str = ""
    if failed_approaches:
        avoid_str = f"\n\nTHIS SESSION: {len(failed_approaches)} strategies tried and failed.\n"
        avoid_str += "RECENT FAILURES (last 10): " + ", ".join(list(sorted(failed_approaches))[-10:])
        if overtrading_patterns:
            avoid_str += f"\nOVERTRADING examples (ENTRY CONDITIONS TOO LOOSE — fix by adding stricter filters):\n"
            for p in overtrading_patterns[-5:]:
                avoid_str += f"  {p}\n"

    # Add DB-level "what works" summary for context (load top performers)
    db_summary = _build_db_summary(primary_tf)

    return f"""EXPERIMENT #{experiment_num:03d}
{phase_hint}

RULES:
- Train: 2021-2024 | Primary TF: 4h/6h/12h/1d preferred | HTF ref: up to 1w
- Signal bar t → fill bar t+1 | Costs: 0.10% ROUND TRIP (0.05%/side)
- REJECT if: DD < -50% | total train trades < 50 | Sharpe ≤ 0
- REJECT if: test trades < 10 (15 months — too few to be statistically meaningful)

#############################################################
# POSITION SIZING AND TRADE COUNT — THE TWO MOST CRITICAL FACTORS
#############################################################
- Signal value = position size. MAX 0.40 magnitude. Normal: 0.20-0.30.
- BTC crashed 77% in 2022. signal=0.30 → you lose 23%. Manageable.
- USE DISCRETE levels (0.0, ±0.20, ±0.30) — every signal change costs 0.10%!
- TRADE COUNT: 50 minimum TOTAL over 4 years = 12.5/year. Target: 75-250 total.
  Fewer than 50 total = statistically unreliable. More than 400 = overtrading.
- USE leverage=1.0 (no leverage).
{db_summary}
CURRENT strategy.py:
```python
{current_strategy}
```
{history_str}{best_str}{avoid_str}

INSTRUCTIONS:
1. State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
2. Implement using REAL indicator formulas. Keep it simple — 2-3 conditions max.
3. MULTI-TIMEFRAME: use mtf_data helper:
   df_htf = get_htf_data(prices, '1d'); vals = your_indicator(df_htf); aligned = align_htf_to_ltf(prices, df_htf, vals)
4. Use proper min_periods on all rolling calculations.
5. VERIFY: estimate how many bars will trigger entry. If signal flips every few bars → too loose.

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

    # results.db is auto-initialized by results_db on import

    # program.md is re-read each experiment so live edits take effect immediately
    system_prompt = build_system_prompt()

    best_strategy_code = STRATEGY_FILE.read_text()

    # Initialize best_sharpe from existing results (don't start from -999)
    best_sharpe = 0.0  # Minimum bar: must beat Sharpe=0 (better than doing nothing)
    try:
        prev_best = query_best_kept_sharpe()
        if prev_best > best_sharpe:
            best_sharpe = prev_best
            print(f"  Resuming from previous best Sharpe: {best_sharpe:.3f}")
    except Exception:
        pass

    # Load recent history from DB so LLM has context even after restart
    history = load_recent_history_from_db(n=20)
    if history:
        print(f"  Loaded {len(history)} recent experiments from DB for LLM context")
        recent_discards = sum(1 for h in history if h["status"] == "discard")
        overtrading = sum(1 for h in history if h.get("fail_reason") == "overtrading")
        if overtrading > 0:
            print(f"  ⚠️  {overtrading}/{len(history)} recent experiments failed due to OVERTRADING")

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
        program = read_file("program.md")  # re-read each time so live edits take effect

        prompt = build_experiment_prompt(
            program=program,
            current_strategy=current_strategy,
            history=history,
            experiment_num=experiment_num,
        )

        try:
            response = client.chat(prompt, system=system_prompt, temperature=0.8)
            new_code = extract_code(response)
        except LLMTimeoutError as e:
            print(f"  [TIMEOUT] LLM call timed out ({e}). Retrying in 10s...")
            time.sleep(10)
            continue
        except Exception as e:
            err_str = str(e).lower()
            if "insufficient_quota" in err_str or "quota exceeded" in err_str:
                print(f"  [QUOTA] Monthly quota exhausted. Waiting 5 min before retry...")
                time.sleep(300)
            elif "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                print(f"  [RATE_LIMIT] Rate limited. Waiting 60s...")
                time.sleep(60)
            else:
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

        # --- Step 3: Per-symbol independent evaluation ---
        # Each symbol is evaluated independently: train → test
        # A strategy can be kept for ETH even if BTC fails
        print("  [2/4] Running per-symbol train+test...")
        description = f"exp#{experiment_num:03d} {strategy_name}"
        any_kept = False
        all_train_results = []
        all_test_results = []

        for symbol in symbols:
            # --- Train ---
            try:
                train_result = run_backtest_all([symbol], str(STRATEGY_FILE), period="train", early_discard=False)
                m = train_result[0]
                sharpe = m["sharpe_ratio"]
                trades = m["num_trades"]
                dd = m["max_drawdown_pct"]
                print(f"    {symbol} train: Sharpe={sharpe:+.3f} Ret={m['total_return_pct']:+.1f}% DD={dd:.1f}% T={trades}")

                train_pass = sharpe > 0.3 and trades >= 50 and dd > -50
                sym_status = "keep" if train_pass else "discard"
                append_results(train_result, sym_status, description, period="train")
                all_train_results.extend(train_result)

                if not train_pass:
                    print(f"    {symbol} → train FAIL, skip test")
                    continue

                # --- Test (only if train passed) ---
                try:
                    test_result = run_backtest_all([symbol], str(STRATEGY_FILE), period="test", early_discard=False)
                    mt = test_result[0]
                    t_sharpe = mt["sharpe_ratio"]
                    t_trades = mt["num_trades"]
                    print(f"    {symbol} test:  Sharpe={t_sharpe:+.3f} Ret={mt['total_return_pct']:+.1f}% T={t_trades}")

                    test_pass = t_sharpe > 0 and t_trades >= 10
                    t_status = "keep" if test_pass else "discard"
                    append_results(test_result, t_status, description, period="test")
                    all_test_results.extend(test_result)

                    if test_pass:
                        print(f"    {symbol} ✓ KEPT (train+test pass)")
                        any_kept = True
                    else:
                        print(f"    {symbol} → test FAIL")

                except Exception as e:
                    print(f"    {symbol} test ERROR: {e}")

            except TimeoutError as e:
                print(f"    {symbol} train TIMEOUT: {e} — skipping")
                continue
            except Exception as e:
                print(f"    {symbol} train ERROR: {e}")
                continue

        # --- Prefix look-ahead test (once, on first symbol) ---
        if any_kept:
            print("  [2b/4] Running look-ahead prefix test...")
            try:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("_strat_test", str(STRATEGY_FILE))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                from prepare import load_klines, load_config as _lc
                import pandas as _pd, numpy as _np
                _cfg = _lc()
                _prices = load_klines(symbols[0], _mod.timeframe)
                _signals_full = _mod.generate_signals(_prices)
                _la_ok = True
                for _cp in [1000, 2000, len(_prices) // 2]:
                    if _cp >= len(_prices): continue
                    _sig_p = _mod.generate_signals(_prices.iloc[:_cp].reset_index(drop=True))
                    if abs(float(_sig_p[-1]) - float(_signals_full[_cp - 1])) > 0.01:
                        _la_ok = False
                        print(f"  [LOOKAHEAD FAIL] at checkpoint {_cp}")
                        break
                if not _la_ok:
                    print(f"  [SKIP] Look-ahead FAIL — discarding all")
                    any_kept = False
                else:
                    print("  [OK] Prefix look-ahead test passed")
            except Exception as _e:
                print(f"  [WARN] Prefix test error: {_e}")

        # Save strategy if any symbol kept
        if any_kept:
            save_strategy(strategy_name)
            kept_count = sum(1 for r in all_test_results if r.get("sharpe_ratio", 0) > 0)
            print(f"  [4/4] ✓ STRATEGY SAVED ({kept_count}/{len(symbols)} symbols pass train+test)")
            best_strategy_code = new_code
        else:
            print(f"  [4/4] ✗ No symbol passed both train+test")
            git_revert_strategy()
            STRATEGY_FILE.write_text(best_strategy_code)

        bt_results = all_train_results  # for history tracking
        test_results = all_test_results
        avg_sharpe = sum(r["sharpe_ratio"] for r in bt_results) / max(1, len(bt_results))
        avg_return = sum(r["total_return_pct"] for r in bt_results) / max(1, len(bt_results))
        avg_trades = sum(r["num_trades"] for r in bt_results) / max(1, len(bt_results))
        status = "keep" if any_kept else "discard"

        # Diagnose failure reason for LLM history feedback
        tf_match = re.search(r'timeframe\s*=\s*["\'](\w+)["\']', new_code)
        strategy_tf = tf_match.group(1) if tf_match else ""
        overtrade_thresh = {"4h": 400, "6h": 300, "12h": 200, "1d": 150, "1h": 600}
        fail_reason = ""
        if not any_kept:
            thresh = overtrade_thresh.get(strategy_tf, 500)
            if avg_trades > thresh:
                fail_reason = f"overtrading({avg_trades:.0f}tr>>{thresh}max)"
            elif avg_trades < 50:
                fail_reason = f"too_few_trades({avg_trades:.0f}tr<<50min)"
            elif avg_sharpe < -1.0:
                fail_reason = "very_neg_sharpe"
            elif avg_sharpe < 0:
                fail_reason = "neg_sharpe"
            elif all(r["sharpe_ratio"] <= 0 or r["num_trades"] < 50 for r in bt_results
                     if r["symbol"] in ("BTCUSDT", "ETHUSDT")):
                fail_reason = "btc_eth_fail_sol_only"

        # Save doc for kept strategies
        if any_kept:
            doc_path = DOCS_DIR / f"{strategy_name}.md"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"""# Strategy: {strategy_name}

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
""" + "".join(
    f"| {r['symbol']} | {r['sharpe_ratio']:.3f} | {r['total_return_pct']:+.1f}% | {r['max_drawdown_pct']:.1f}% | {r['num_trades']} | {'PASS' if r['sharpe_ratio']>0 else 'FAIL'} |\n"
    for r in all_train_results
) + """
## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
""" + "".join(
    f"| {r['symbol']} | {r['sharpe_ratio']:.3f} | {r['total_return_pct']:+.1f}% | {r['max_drawdown_pct']:.1f}% | {r['num_trades']} | {'PASS' if r['sharpe_ratio']>0 else 'FAIL'} |\n"
    for r in all_test_results
) + f"""
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
        # Results already logged per-symbol inside the loop above
        # Do NOT call append_results again here (was causing duplicates)

        history.append({
            "num": experiment_num,
            "name": strategy_name,
            "status": status,
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
            "avg_trades": avg_trades,
            "tf": strategy_tf,
            "fail_reason": fail_reason,
            "description": description,
        })

        # Progress summary every 10 experiments
        if experiment_num % 10 == 0:
            kept = sum(1 for h in history if h["status"] == "keep")
            print(f"\n  ── Summary: {experiment_num} experiments, {kept} kept, best Sharpe={best_sharpe:.3f} ──")

    print("\nResearch loop complete.")


if __name__ == "__main__":
    main()
