#!/usr/bin/env python3
"""
Periodic internet discovery for fresh strategy ideas.

Pulls recent public articles/papers via RSS feeds, combines them with current
results.db evidence, and writes a compact digest for the research loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote_plus

import requests
from research_rules import MAX_ALLOWED_DRAWDOWN_PCT, TRAIN_MIN_TRADES

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "results.db"
DOC_PATH = ROOT / "docs" / "latest_strategy_discovery.md"
JSON_PATH = ROOT / "logs" / "latest_strategy_discovery.json"
LOG_PATH = ROOT / "logs" / "internet_strategy_discovery.log"

GOOGLE_NEWS_QUERIES = [
    "crypto futures trading strategy volume breakout trend filter",
    "crypto funding rate trading strategy perp futures",
    "tradingview strategy crypto breakout volume trend",
    "algorithmic trading crypto regime filter momentum strategy",
]

ARXIV_QUERIES = [
    "all:crypto trading strategy",
    "all:algorithmic trading momentum regime",
]

KEYWORDS = [
    "volume", "breakout", "trend", "momentum", "mean reversion", "regime",
    "funding", "donchian", "camarilla", "ichimoku", "adx", "rsi", "ema",
    "kama", "hma", "volatility", "ensemble",
]

TRADING_TERMS = {
    "trading", "strategy", "breakout", "momentum", "mean reversion",
    "funding", "futures", "perpetual", "perp", "regime", "volatility",
    "donchian", "camarilla", "ichimoku", "adx", "rsi", "ema", "volume",
    "vwap", "pivot", "oscillator", "trend", "pullback", "channel",
}

MARKET_TERMS = {
    "crypto", "bitcoin", "ethereum", "solana", "btc", "eth", "sol",
    "altcoin", "binance", "futures", "perp", "perpetual", "market",
}

IRRELEVANT_TERMS = {
    "medical", "hallucination", "forest", "networking", "board game",
    "oversight", "token grounding", "lvlm", "device", "serious game",
}


def log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fetch_xml(url: str, timeout: int = 30) -> ET.Element:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "trading-llm-auto-research/1.0"})
    response.raise_for_status()
    return ET.fromstring(response.text)


def text_or_empty(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def fetch_google_news(limit_per_query: int = 5) -> list[dict]:
    items: list[dict] = []
    for query in GOOGLE_NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            root = fetch_xml(url)
        except Exception as exc:
            log(f"google news fetch failed for query='{query}': {exc}")
            continue
        channel = root.find("channel")
        if channel is None:
            continue
        for item in channel.findall("item")[:limit_per_query]:
            title = text_or_empty(item, "title")
            link = text_or_empty(item, "link")
            pub = text_or_empty(item, "pubDate")
            source = text_or_empty(item, "source")
            if not title or not link:
                continue
            items.append({
                "source_type": "google_news",
                "query": query,
                "title": title,
                "url": link,
                "published": pub,
                "source": source or "Google News",
            })
    return items


def fetch_arxiv(limit_per_query: int = 5) -> list[dict]:
    items: list[dict] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for query in ARXIV_QUERIES:
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query={quote_plus(query)}&start=0&max_results={limit_per_query}&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            root = fetch_xml(url)
        except Exception as exc:
            log(f"arxiv fetch failed for query='{query}': {exc}")
            continue
        for entry in root.findall("atom:entry", ns):
            title = text_or_empty(entry, "{http://www.w3.org/2005/Atom}title")
            summary = text_or_empty(entry, "{http://www.w3.org/2005/Atom}summary")
            url_link = ""
            for link in entry.findall("atom:link", ns):
                href = link.attrib.get("href", "").strip()
                if href and link.attrib.get("rel", "alternate") == "alternate":
                    url_link = href
                    break
            published = text_or_empty(entry, "{http://www.w3.org/2005/Atom}published")
            if not title or not url_link:
                continue
            items.append({
                "source_type": "arxiv",
                "query": query,
                "title": " ".join(title.split()),
                "url": url_link,
                "published": published,
                "source": "arXiv",
                "summary": " ".join(summary.split()),
            })
    return items


def relevance_score(item: dict) -> int:
    base = " ".join(
        str(item.get(key, "")).lower()
        for key in ("title", "source", "summary")
    )
    query = str(item.get("query", "")).lower()
    score = 0
    if any(term in base for term in TRADING_TERMS):
        score += 2
    if any(term in base for term in MARKET_TERMS):
        score += 2
    elif any(term in query for term in MARKET_TERMS):
        score += 1
    if any(term in base for term in KEYWORDS):
        score += 1
    if item.get("source_type") == "arxiv" and "trading" in base:
        score += 1
    if any(term in base for term in IRRELEVANT_TERMS):
        score -= 5
    return score


def is_relevant_item(item: dict) -> bool:
    score = relevance_score(item)
    if item.get("source_type") == "arxiv":
        base = " ".join(
            str(item.get(key, "")).lower()
            for key in ("title", "summary")
        )
        direct_market_match = any(
            term in base
            for term in ("trading", "market", "crypto", "bitcoin", "ethereum", "futures")
        )
        return score >= 3 and direct_market_match
    return score >= 3


def dedupe_items(items: list[dict], limit: int = 25) -> list[dict]:
    filtered = [item for item in items if is_relevant_item(item)]
    filtered.sort(
        key=lambda item: (
            relevance_score(item),
            str(item.get("published", "")),
        ),
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for item in filtered:
        key = (item.get("title", "").lower(), item.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def classify_keywords(items: list[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for item in items:
        hay = f"{item.get('title', '')} {item.get('query', '')}".lower()
        for keyword in KEYWORDS:
            if keyword in hay:
                counter[keyword] += 1
    return [name for name, _ in counter.most_common(8)]


def summarize_results_db() -> dict:
    if not DB_FILE.exists():
        return {"top_test": [], "top_train": [], "failure_buckets": {}, "best_timeframes": []}

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        top_test = conn.execute(
            """
            SELECT strategy, symbol, sharpe, return_pct, max_dd_pct, trades
            FROM results
            WHERE period='test' AND status='keep'
            ORDER BY sharpe DESC
            LIMIT 12
            """
        ).fetchall()
        top_train = conn.execute(
            """
            SELECT strategy, symbol, sharpe, return_pct, max_dd_pct, trades
            FROM results
            WHERE period='train' AND status='keep'
            ORDER BY sharpe DESC
            LIMIT 12
            """
        ).fetchall()
        timeframe_rows = conn.execute(
            """
            SELECT substr(strategy, 1, instr(strategy || '_', '_') - 1) AS prefix, COUNT(*) as cnt
            FROM results
            WHERE status='keep'
            GROUP BY prefix
            ORDER BY cnt DESC
            LIMIT 8
            """
        ).fetchall()
        fail_rows = conn.execute(
            """
            SELECT sharpe, trades, max_dd_pct
            FROM results
            WHERE period='train' AND status='discard'
            ORDER BY id DESC
            LIMIT 400
            """
        ).fetchall()
    finally:
        conn.close()

    buckets = {"deep_drawdown": 0, "too_few_trades": 0, "negative_sharpe": 0, "other": 0}
    for row in fail_rows:
        sharpe = float(row["sharpe"] or 0)
        trades = int(row["trades"] or 0)
        dd = float(row["max_dd_pct"] or 0)
        if dd <= MAX_ALLOWED_DRAWDOWN_PCT:
            buckets["deep_drawdown"] += 1
        elif trades < TRAIN_MIN_TRADES:
            buckets["too_few_trades"] += 1
        elif sharpe <= 0:
            buckets["negative_sharpe"] += 1
        else:
            buckets["other"] += 1

    return {
        "top_test": [dict(row) for row in top_test],
        "top_train": [dict(row) for row in top_train],
        "failure_buckets": buckets,
        "best_timeframes": [dict(row) for row in timeframe_rows],
    }


def build_markdown(items: list[dict], db_summary: dict) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    keywords = classify_keywords(items)
    lines = [
        "# Latest Strategy Discovery",
        "",
        f"Generated: {ts}",
        "",
        "## Results-Driven Direction",
        f"- Recent discard buckets: {json.dumps(db_summary.get('failure_buckets', {}), ensure_ascii=False)}",
        f"- Active kept strategy rows: {len(db_summary.get('top_train', []))} sampled train winners, {len(db_summary.get('top_test', []))} sampled test winners",
    ]

    best_timeframes = db_summary.get("best_timeframes", [])
    if best_timeframes:
        lines.append("- Most common kept strategy prefixes:")
        for row in best_timeframes:
            lines.append(f"  - {row['prefix']}: {row['cnt']} kept rows")

    if keywords:
        lines.append(f"- Fresh web themes: {', '.join(keywords)}")

    lines.extend(["", "## Fresh Web Discoveries"])
    if not items:
        lines.append("- No web discoveries fetched this run.")
    else:
        for item in items:
            title = item.get("title", "").replace("\n", " ").strip()
            lines.append(
                f"- [{title}]({item.get('url', '')}) | {item.get('source', '')} | "
                f"{item.get('published', '')} | query=`{item.get('query', '')}`"
            )

    lines.extend(["", "## Current Winners Snapshot"])
    for section_name, rows in (("Train", db_summary.get("top_train", [])), ("Test", db_summary.get("top_test", []))):
        lines.append(f"### {section_name}")
        if not rows:
            lines.append("- None")
            continue
        for row in rows[:8]:
            lines.append(
                f"- {row['strategy']} | {row['symbol']} | Sharpe={float(row['sharpe'] or 0):.3f} | "
                f"Ret={float(row['return_pct'] or 0):+.1f}% | DD={float(row['max_dd_pct'] or 0):.1f}% | "
                f"Trades={int(row['trades'] or 0)}"
            )

    lines.extend([
        "",
        "## How To Use",
        "- `auto_concept_research.py` should read this file and bias concept generation toward fresh web themes that do not contradict current repo winners.",
        "- Use results.db evidence to avoid repeating combinations that already fail by DD, overtrading, or low activity.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(items: list[dict], db_summary: dict, dry_run: bool = False) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "items": items,
        "db_summary": db_summary,
    }
    markdown = build_markdown(items, db_summary)
    if dry_run:
        print(markdown)
        return
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(markdown, encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch fresh public strategy ideas and summarize them for autoresearch.")
    parser.add_argument("--dry-run", action="store_true", help="Print markdown instead of writing files.")
    args = parser.parse_args()

    log("=== Internet Strategy Discovery START ===")
    google_items = fetch_google_news()
    arxiv_items = fetch_arxiv()
    items = dedupe_items(google_items + arxiv_items)
    db_summary = summarize_results_db()
    write_outputs(items, db_summary, dry_run=args.dry_run)
    log(f"discovered {len(items)} web items")
    log("=== Internet Strategy Discovery DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
