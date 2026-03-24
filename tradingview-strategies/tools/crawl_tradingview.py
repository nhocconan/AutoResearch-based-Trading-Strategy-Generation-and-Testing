#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://www.tradingview.com"
LISTING_PATH = "/scripts/"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
PRS_DATA_RE = re.compile(
    r'<script type="application/prs\.init-data\+json">(.*?)</script>',
    re.DOTALL,
)


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def extract_prs_data_blocks(html: str) -> list[dict[str, Any]]:
    matches = list(PRS_DATA_RE.finditer(html))
    if not matches:
        raise RuntimeError("Could not locate TradingView prs init data blocks")
    return [json.loads(match.group(1)) for match in matches]


def find_listing_blob(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        data = node.get("data")
        if isinstance(data, dict) and "feed" in data:
            return {"kind": "feed", "payload": data["feed"]}
        nested = data.get("data") if isinstance(data, dict) else None
        if isinstance(nested, dict) and "ideas" in nested:
            return {"kind": "ideas", "payload": nested["ideas"]}
        if isinstance(nested, dict) and "feed" in nested:
            return {"kind": "feed", "payload": nested["feed"]}
        if isinstance(nested, dict) and "scripts" in nested:
            return {"kind": "scripts", "payload": nested["scripts"]}
            return node
        for value in node.values():
            found = find_listing_blob(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_listing_blob(value)
            if found is not None:
                return found
    return None


def extract_feed_blob(prs_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    for block in prs_blocks:
        found = find_listing_blob(block)
        if found is not None:
            return found
    raise RuntimeError("Could not locate feed blob inside prs data blocks")


def build_page_url(page_number: int, query: dict[str, str]) -> str:
    path = LISTING_PATH if page_number == 1 else f"{LISTING_PATH}page-{page_number}/"
    return f"{BASE_URL}{path}?{urlencode(query)}"


def crawl_pages(
    output_dir: Path,
    start_page: int,
    max_pages: int | None,
    sleep_s: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw-pages"
    raw_dir.mkdir(parents=True, exist_ok=True)

    query = {
        "script_type": "strategies",
        "script_access": "open",
        "sort": "recent_extended",
    }

    manifest: dict[str, Any] = {
        "query": query,
        "pages": [],
        "total_items": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    seen_urls: set[str] = set()
    page = start_page
    page_count = 0

    while True:
        if max_pages is not None and page_count >= max_pages:
            break

        url = build_page_url(page, query)
        html = fetch_text(url)
        (raw_dir / f"page-{page}.html").write_text(html)

        prs_blocks = extract_prs_data_blocks(html)
        blob = extract_feed_blob(prs_blocks)
        payload = blob["payload"]
        if "data" in payload and isinstance(payload["data"], dict):
            payload = payload["data"]
        items = payload.get("entities") or payload.get("items") or []
        next_page = payload.get("next")

        page_items: list[dict[str, Any]] = []
        for item in items:
            page_item = {
                "id": item["id"],
                "name": item["name"],
                "chart_url": item["chart_url"],
                "description": item.get("description", ""),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "script_type": item.get("script_type"),
                "script_access": item.get("script_access"),
                "comments_count": item.get("comments_count"),
                "likes_count": item.get("likes_count"),
                "symbol": item.get("symbol", {}),
                "user": {"id": item["user"]["id"], "username": item["user"]["username"]},
            }
            page_items.append(page_item)
            seen_urls.add(item["chart_url"])

        page_summary = {
            "page": page,
            "url": url,
            "items": page_items,
            "item_count": len(page_items),
            "next": next_page,
        }
        manifest["pages"].append(page_summary)
        manifest["total_items"] += len(page_items)
        page_count += 1

        if not next_page:
            break

        page += 1
        time.sleep(sleep_s)

    manifest["unique_urls"] = len(seen_urls)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl TradingView open-source strategy listing metadata.")
    parser.add_argument(
        "--output-dir",
        default="tradingview-strategies/crawl",
        help="Output directory inside the repo.",
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Crawl at most this many listing pages per run. Default 20 so large crawls proceed in page batches.",
    )
    parser.add_argument("--sleep-s", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = crawl_pages(
        output_dir=output_dir,
        start_page=args.start_page,
        max_pages=args.max_pages,
        sleep_s=args.sleep_s,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    flat_rows: list[dict[str, Any]] = []
    for page in manifest["pages"]:
        for item in page["items"]:
            flat_rows.append(
                {
                    "page": page["page"],
                    "name": item["name"],
                    "chart_url": item["chart_url"],
                    "created_at": item["created_at"],
                    "likes_count": item["likes_count"],
                    "comments_count": item["comments_count"],
                    "symbol_name": item["symbol"].get("name"),
                    "symbol_type": item["symbol"].get("type"),
                    "interval": item["symbol"].get("interval"),
                    "author": item["user"]["username"],
                }
            )
    (output_dir / "strategies.json").write_text(json.dumps(flat_rows, indent=2))

    next_start_page = args.start_page + len(manifest["pages"])
    print(
        json.dumps(
            {
                "pages_crawled": len(manifest["pages"]),
                "total_items": manifest["total_items"],
                "unique_urls": manifest["unique_urls"],
                "output_dir": str(output_dir),
                "next_start_page": next_start_page,
                "note": "Run another batch with --start-page next_start_page to continue in 20-page chunks.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
