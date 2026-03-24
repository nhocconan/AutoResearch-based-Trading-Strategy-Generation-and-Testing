#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
CANDIDATES_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "btc-eth-candidates.json"
OUT_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"

SUPPORTED_INTERVALS = {
    "1": "1m",
    "3": None,
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "45": None,
    "60": "1h",
    "120": None,
    "180": None,
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "1D": "1d",
    "D": "1d",
    "1W": "1w",
    "W": "1w",
}


def score_row(row: dict) -> tuple:
    symbol = (row.get("symbol_name") or "").upper()
    name = (row.get("name") or "").upper()
    likes = row.get("likes_count") or 0
    comments = row.get("comments_count") or 0
    interval = row.get("interval")
    mapped = SUPPORTED_INTERVALS.get(interval)

    btc_eth_symbol = any(tok in symbol for tok in ("BTC", "XBT", "ETH"))
    btc_eth_name = any(tok in name for tok in ("BTC", "ETH"))
    mapped_tf = mapped is not None
    return (
        int(mapped_tf),
        int(btc_eth_symbol),
        int(btc_eth_name),
        likes,
        comments,
    )


def main() -> None:
    rows = json.loads(CANDIDATES_PATH.read_text())

    dedup: dict[str, dict] = {}
    for row in rows:
        url = row["chart_url"]
        interval = row.get("interval")
        repo_tf = SUPPORTED_INTERVALS.get(interval)
        enriched = {
            **row,
            "repo_timeframe": repo_tf,
            "queue_score": score_row(row),
        }
        prev = dedup.get(url)
        if prev is None or tuple(enriched["queue_score"]) > tuple(prev["queue_score"]):
            dedup[url] = enriched

    queue = sorted(dedup.values(), key=lambda row: tuple(row["queue_score"]), reverse=True)
    for idx, row in enumerate(queue, start=1):
        row["queue_rank"] = idx
        row.pop("queue_score", None)

    payload = {
        "source_count": len(rows),
        "deduped_count": len(queue),
        "supported_timeframe_count": sum(1 for row in queue if row["repo_timeframe"]),
        "items": queue,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_count": payload["source_count"],
        "deduped_count": payload["deduped_count"],
        "supported_timeframe_count": payload["supported_timeframe_count"],
        "output": str(OUT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
