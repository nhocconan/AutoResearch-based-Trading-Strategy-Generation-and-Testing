#!/usr/bin/env python3
"""
auto_concept_research.py - Automated Concept Discovery (runs every 12h)
========================================================================
Searches for new trading strategy concepts not already in program.md,
analyzes recent experiment results to avoid exhausted combinations,
and appends promising new concepts to the knowledge base.

Run:
    python auto_concept_research.py [--dry-run]
"""

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Load environment
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from llm_client import LLMClient

LOG_FILE = Path("auto_concept_research.log")
PROGRAM_MD = Path("program.md")
DB_FILE = Path("results.db")
MARKER = "## RECENTLY DISCOVERED CONCEPTS"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def resolve_analysis_model() -> str:
    return (
        os.environ.get("OLLAMA_ANALYSIS_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "glm-5"
    )


def get_indicator_stats() -> dict:
    """Get indicator keep rates and counts from results.db."""
    if not DB_FILE.exists():
        return {}
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT description, status, COUNT(*) as cnt
        FROM results WHERE period='train'
        GROUP BY description, status
    """).fetchall()
    conn.close()

    # Simple: use strategy names to infer indicators
    conn = sqlite3.connect(DB_FILE)
    ind_rows = conn.execute("""
        SELECT strategy, MAX(sharpe) as best_sharpe, status
        FROM results WHERE period='test' AND status='keep'
        GROUP BY strategy ORDER BY best_sharpe DESC LIMIT 5
    """).fetchall()
    conn.close()
    return {"top_test": [r[0] for r in ind_rows]}


def get_exhausted_combinations() -> list[str]:
    """Get most-tested strategy patterns to avoid repeating."""
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT substr(strategy, 1, 40) as prefix, COUNT(*) as cnt,
               SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) as kept
        FROM results WHERE period='train'
        GROUP BY prefix
        HAVING cnt > 50
        ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    conn.close()
    return [f"{r[0]} ({r[1]} tested, {r[2]} kept)" for r in rows]


def already_in_program(text: str, program_content: str) -> bool:
    """Check if a concept keyword is already heavily covered in program.md."""
    key_words = re.findall(r'\*\*([A-Z][A-Za-z ]+)\*\*', text)
    for kw in key_words:
        if kw.lower() in program_content.lower() and len(kw) > 5:
            return True
    return False


def build_prompt(program_content: str, exhausted: list[str], stats: dict) -> str:
    # Extract what's already covered
    existing_tiers = re.findall(r'### TIER \d+.*?(?=### TIER|\Z)', program_content, re.DOTALL)
    existing_names = re.findall(r'\*\*\d+\. ([^\*]+)\*\*', program_content)
    top_test = stats.get("top_test", [])

    existing_summary = ", ".join(existing_names[:30]) if existing_names else "many standard indicators"

    exhausted_str = "\n".join(f"  - {e}" for e in exhausted[:10]) if exhausted else "  (none yet)"

    top_test_str = "\n".join(f"  - {s}" for s in top_test[:5]) if top_test else "  (none yet)"

    return f"""You are a quantitative trading researcher. Your job is to discover NEW trading strategy concepts for crypto futures (BTC/ETH/SOL USDT-M perpetuals on Binance).

CONTEXT:
- We run backtests on 2021-2024 train data, then validate on 2025+ test period (BTC -25%, bearish/range)
- Already covered strategies: {existing_summary}
- Cost: 0.10% round trip (taker 0.04%/side + slippage 0.01%/side)
- Best test results so far: {top_test_str}

MOST EXHAUSTED COMBINATIONS (do NOT suggest these):
{exhausted_str}

TASK: Generate 4-6 SPECIFIC, NOVEL trading strategy concepts NOT already listed above. These must be:
1. Implementable in Python with numpy/pandas (vectorized, no external data other than OHLCV + funding rate)
2. Potentially profitable in BEAR/RANGE markets (2025 context: BTC -25%, ranging)
3. Novel — not simple variations of EMA/SMA crossover or basic RSI mean reversion
4. Include SPECIFIC parameters and formulas (not vague descriptions)
5. Prioritize: market microstructure, order flow proxies, cross-timeframe divergence, volatility regime, seasonality patterns

FORMAT: For each strategy, use this exact format (markdown):
**N. Strategy Name** *(best timeframe(s))*
- Core idea: [one sentence]
- Formula: [specific formulas/parameters]
- Entry: [specific entry condition]
- Exit: [specific exit condition]
- Why it might work in 2025 bear market: [one sentence]
- Priority: [HIGH/MEDIUM] — [reason]

Focus on concepts like: Vortex Indicator, Elder Ray, Williams Alligator, Fractal Adaptive MA (FRAMA), MESA Adaptive MA, Hurst Exponent regime, Ehlers Roofing Filter, Cyber Cycle, Stochastic RSI divergence, Order Book Imbalance proxy (taker_buy_volume), OBV divergence from price with Heikin-Ashi confirmation, Accumulation/Distribution line, Market Facilitation Index, Detrended Price Oscillator, Price Oscillator, Commodity Channel Index extremes, TRIX indicator, Ultimate Oscillator, Aroon Oscillator with ADX, Parabolic SAR optimized for crypto, Schaff Trend Cycle (STC), CPR + VWAP combo, Gann levels with HMA, or entirely different approaches not listed here.

Output ONLY the formatted strategy descriptions, no preamble."""


def append_concepts(concepts: str, dry_run: bool = False) -> bool:
    """Append new concepts to program.md under the RECENTLY DISCOVERED section."""
    if not PROGRAM_MD.exists():
        log("ERROR: program.md not found")
        return False

    content = PROGRAM_MD.read_text()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    new_block = f"\n### Batch discovered {ts}\n\n{concepts.strip()}\n"

    if MARKER in content:
        # Append after existing marker
        insert_pos = content.index(MARKER) + len(MARKER)
        new_content = content[:insert_pos] + new_block + content[insert_pos:]
    else:
        # Add new section before NEVER STOP or at end
        if "## NEVER STOP" in content:
            insert_pos = content.index("## NEVER STOP")
            new_content = content[:insert_pos] + f"{MARKER}\n{new_block}\n" + content[insert_pos:]
        else:
            new_content = content + f"\n{MARKER}\n{new_block}\n"

    if dry_run:
        print("\n--- DRY RUN: would append to program.md ---")
        print(new_block[:1000])
        print("---")
        return True

    PROGRAM_MD.write_text(new_content)
    log(f"Appended {len(concepts)} chars of new concepts to program.md")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't write to program.md")
    args = parser.parse_args()

    log("=== Auto Concept Research START ===")

    # Read current program.md
    if not PROGRAM_MD.exists():
        log("ERROR: program.md not found, aborting")
        sys.exit(1)

    program_content = PROGRAM_MD.read_text()

    # Get context from results DB
    stats = get_indicator_stats()
    exhausted = get_exhausted_combinations()

    log(f"Got {len(exhausted)} exhausted combinations from DB")

    # Build prompt and call LLM
    prompt = build_prompt(program_content, exhausted, stats)

    try:
        model = resolve_analysis_model()
        client = LLMClient(provider="ollama", model_override=model)
        log(f"Using provider: {client.provider}, model: {client._get_model()}")

        response = client.chat(
            message=prompt,
            system="You are an expert quantitative trading researcher. Be specific, concrete, and novel.",
            max_tokens=4096,
        )

        if not response or len(response) < 200:
            log(f"WARNING: LLM returned suspiciously short response ({len(response)} chars)")
            sys.exit(1)

        log(f"LLM returned {len(response)} chars")

        # Append to program.md
        success = append_concepts(response, dry_run=args.dry_run)
        if success:
            log("=== Auto Concept Research DONE ===")
        else:
            log("ERROR: Failed to append concepts")
            sys.exit(1)

    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
