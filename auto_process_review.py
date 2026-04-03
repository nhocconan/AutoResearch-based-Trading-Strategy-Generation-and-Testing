#!/usr/bin/env python3
"""
auto_process_review.py - Automated process review for the research pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load environment
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for raw_line in _env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.partition("=")[::2]
        os.environ.setdefault(key.strip(), value.strip())

from llm_client import LLMClient

DB_FILE = Path("results.db")
PROGRAM_MD = Path("program.md")
REPORT_PATH = Path("docs/auto_research_review.md")
LOG_FILE = Path("logs/auto_process_review.log")

OVERTRADE_THRESH = {"4h": 400, "6h": 300, "12h": 200, "1d": 150, "1h": 600}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def resolve_analysis_model() -> str:
    return (
        os.environ.get("OLLAMA_ANALYSIS_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "glm-5"
    )


def _strategy_tf(name: str) -> str:
    match = re.search(r"mtf_(\w+?)_", name or "")
    return match.group(1) if match else ""


def collect_recent_train(limit: int = 240) -> list[sqlite3.Row]:
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT strategy, symbol, sharpe, trades, max_dd_pct, status
            FROM results
            WHERE period='train'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def collect_top_test(limit: int = 12) -> list[sqlite3.Row]:
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT strategy, symbol, sharpe, trades, return_pct, max_dd_pct
            FROM results
            WHERE period='test' AND status='keep'
            ORDER BY sharpe DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def summarize_recent_failures(rows: list[sqlite3.Row]) -> dict[str, int]:
    buckets = {
        "overtrading": 0,
        "too_few_trades": 0,
        "negative_sharpe": 0,
        "deep_drawdown": 0,
        "other": 0,
    }
    for row in rows:
        strategy = row["strategy"] or ""
        tf = _strategy_tf(strategy)
        trades = float(row["trades"] or 0)
        sharpe = float(row["sharpe"] or 0)
        max_dd = float(row["max_dd_pct"] or 0)
        thresh = OVERTRADE_THRESH.get(tf, 500)
        if trades > thresh:
            buckets["overtrading"] += 1
        elif trades < 50:
            buckets["too_few_trades"] += 1
        elif sharpe <= 0:
            buckets["negative_sharpe"] += 1
        elif max_dd <= -50:
            buckets["deep_drawdown"] += 1
        else:
            buckets["other"] += 1
    return buckets


def summarize_keep_rates(rows: list[sqlite3.Row]) -> list[str]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        tf = _strategy_tf(row["strategy"] or "") or "unknown"
        state = grouped.setdefault(tf, {"total": 0, "kept": 0})
        state["total"] += 1
        if row["status"] == "keep":
            state["kept"] += 1
    lines = []
    for tf, stats in sorted(grouped.items()):
        total = stats["total"]
        kept = stats["kept"]
        keep_rate = (100.0 * kept / total) if total else 0.0
        lines.append(f"- {tf}: kept {kept}/{total} ({keep_rate:.1f}%)")
    return lines


def build_prompt(train_rows: list[sqlite3.Row], top_test_rows: list[sqlite3.Row]) -> str:
    failure_buckets = summarize_recent_failures(train_rows)
    keep_rates = summarize_keep_rates(train_rows)
    top_test_lines = [
        (
            f"- {row['strategy']} | {row['symbol']} | "
            f"Sharpe={float(row['sharpe'] or 0):.3f} | "
            f"Trades={int(row['trades'] or 0)} | "
            f"Ret={float(row['return_pct'] or 0):+.1f}% | "
            f"DD={float(row['max_dd_pct'] or 0):.1f}%"
        )
        for row in top_test_rows[:10]
    ]
    program_excerpt = ""
    if PROGRAM_MD.exists():
        text = PROGRAM_MD.read_text(encoding="utf-8")
        program_excerpt = text[:5000]

    return f"""You are reviewing an autonomous crypto strategy research system.

GOAL:
Produce a sharp operational review of how to improve the research loop itself, not new trading indicators.

CURRENT SETUP:
- Main generator: agent_research.py
- Knowledge base: program.md
- Concept discovery: auto_concept_research.py
- Validation: static checks + backtests + prefix look-ahead test
- Current setup already uses Ollama Cloud

RECENT TRAIN FAILURE BUCKETS:
- overtrading: {failure_buckets['overtrading']}
- too_few_trades: {failure_buckets['too_few_trades']}
- negative_sharpe: {failure_buckets['negative_sharpe']}
- deep_drawdown: {failure_buckets['deep_drawdown']}
- other: {failure_buckets['other']}

RECENT TRAIN KEEP RATES BY TF:
{chr(10).join(keep_rates) or "- no recent data"}

TOP TEST WINNERS:
{chr(10).join(top_test_lines) or "- none"}

PROGRAM EXCERPT:
{program_excerpt}

TASK:
Write a concise markdown memo with these sections:
1. Executive Summary
2. Structural Weaknesses
3. Highest-Impact Changes
4. Model Split Recommendation
5. Next 7 Concrete Actions

Requirements:
- Focus on sample efficiency, invalid-code reduction, overtrading control, better failure feedback, and model-role separation.
- Be implementation-specific and reference the existing scripts by filename when relevant.
- Recommend one Ollama Cloud model for generation and one for analysis/review, with short justification.
- Only choose models from this exact candidate set:
  `nemotron-3-super`, `glm-5`, `gpt-oss:120b`, `qwen3-next:80b`, `deepseek-v3.2`, `kimi-k2-thinking`
- Keep it under 900 words.
- Output markdown only.
"""


def write_report(content: str, dry_run: bool = False) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = f"# Auto Research Review\n\nGenerated: {ts}\n\n{content.strip()}\n"
    if dry_run:
        print(body)
        return
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an LLM review of the auto-research process")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing docs/auto_research_review.md")
    args = parser.parse_args()

    log("=== Auto Process Review START ===")

    train_rows = collect_recent_train()
    top_test_rows = collect_top_test()
    log(f"Loaded {len(train_rows)} recent train rows and {len(top_test_rows)} top test rows")

    model = resolve_analysis_model()
    prompt = build_prompt(train_rows, top_test_rows)

    try:
        client = LLMClient(provider="ollama", model_override=model)
        log(f"Using provider: {client.provider}, model: {client._get_model()}")
        response = client.chat(
            prompt,
            system="You are a rigorous research-ops reviewer for systematic trading experiments.",
            temperature=0.2,
            max_tokens=2500,
            timeout=300,
        )
        if not response or len(response.strip()) < 300:
            log(f"ERROR: suspiciously short response ({len(response or '')} chars)")
            return 1
        write_report(response, dry_run=args.dry_run)
        log("=== Auto Process Review DONE ===")
        return 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
