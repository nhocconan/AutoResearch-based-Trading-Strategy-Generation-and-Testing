#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
SOURCE_MANIFEST = TV_ROOT / "crawl" / "recent-open-strategies" / "manifest.json"
RAW_ROOT = TV_ROOT / "raw-pine"
CACHE_MANIFEST = RAW_ROOT / "cache-manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_repo_path(rel_file: str | None) -> Path | None:
    if not rel_file:
        return None
    path = Path(rel_file)
    if path.is_absolute():
        return path
    if rel_file.startswith("tradingview-strategies/"):
        return ROOT / rel_file
    return TV_ROOT / rel_file


def normalize_rel_file(rel_file: str | None) -> str | None:
    full_path = resolve_repo_path(rel_file)
    if not full_path or not full_path.exists():
        return rel_file
    if full_path.is_relative_to(TV_ROOT):
        return str(full_path.relative_to(TV_ROOT))
    if full_path.is_relative_to(ROOT):
        return str(full_path.relative_to(ROOT))
    return str(full_path)


def collect_source_rows() -> list[dict[str, Any]]:
    manifest = load_json(SOURCE_MANIFEST, {})
    rows: list[dict[str, Any]] = []
    rank = 1
    for page in manifest.get("pages", []):
        for item in page.get("items", []):
            rows.append(
                {
                    "queue_rank": rank,
                    "page": page.get("page"),
                    "name": item.get("name"),
                    "chart_url": item.get("chart_url"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "likes_count": item.get("likes_count"),
                    "comments_count": item.get("comments_count"),
                    "symbol": item.get("symbol", {}),
                    "author": item.get("user", {}).get("username"),
                }
            )
            rank += 1
    return rows


def collect_existing_cache() -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}

    manifest_paths = [
        RAW_ROOT / "manifest.json",
        RAW_ROOT / "bulk" / "manifest.json",
        *sorted((RAW_ROOT / "bulk").glob("manifest.w*.json")),
    ]
    for path in manifest_paths:
        payload = load_json(path, {"items": []})
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        for item in items:
            url = item.get("chart_url") or item.get("url")
            rel_file = item.get("file")
            if not url or not rel_file:
                continue
            full_path = resolve_repo_path(rel_file)
            if not full_path.exists():
                continue
            rel_file = normalize_rel_file(rel_file)
            by_url[url] = {
                "status": "ok",
                "pine_file": rel_file,
                "line_count": int(item.get("line_count") or 0),
                "last_error": None,
                "source": path.name,
            }

    for path in sorted((TV_ROOT / "results" / "bulk").glob("*.json")):
        if path.name.startswith("summary"):
            continue
        payload = load_json(path, {})
        url = payload.get("chart_url")
        rel_file = payload.get("pine_file")
        if not url or not rel_file:
            continue
        full_path = resolve_repo_path(rel_file)
        if not full_path.exists():
            continue
        by_url.setdefault(
            url,
            {
                "status": "ok",
                "pine_file": normalize_rel_file(rel_file),
                "line_count": 0,
                "last_error": None,
                "source": path.name,
            },
        )

    existing_manifest = load_json(CACHE_MANIFEST, {"items": []})
    for item in existing_manifest.get("items", []):
        url = item.get("chart_url")
        rel_file = item.get("pine_file")
        if not url:
            continue
        full_path = resolve_repo_path(rel_file)
        if full_path and full_path.exists():
            by_url[url] = {
                "status": "ok",
                "pine_file": normalize_rel_file(rel_file),
                "line_count": int(item.get("line_count") or 0),
                "last_error": item.get("last_error"),
                "source": "cache-manifest",
            }
        elif item.get("status") in {"error", "pending"} and url not in by_url:
            by_url[url] = {
                "status": item.get("status") or "pending",
                "pine_file": rel_file,
                "line_count": int(item.get("line_count") or 0),
                "last_error": item.get("last_error"),
                "source": "cache-manifest",
                "extract_attempts": int(item.get("extract_attempts") or 0),
                "updated_at": item.get("updated_at"),
            }
    return by_url


def build_manifest() -> dict[str, Any]:
    source_rows = collect_source_rows()
    existing = collect_existing_cache()
    items = []
    for row in source_rows:
        cached = existing.get(row["chart_url"], {})
        item = {
            **row,
            "status": cached.get("status", "pending"),
            "pine_file": cached.get("pine_file"),
            "line_count": int(cached.get("line_count") or 0),
            "extract_attempts": int(cached.get("extract_attempts") or 0),
            "last_error": cached.get("last_error"),
            "updated_at": cached.get("updated_at"),
        }
        full_path = resolve_repo_path(item["pine_file"])
        if item["status"] == "ok" and item["pine_file"] and (full_path is None or not full_path.exists()):
            item["status"] = "pending"
            item["pine_file"] = None
            item["line_count"] = 0
        items.append(item)
    return {
        "source_manifest": str(SOURCE_MANIFEST.relative_to(TV_ROOT)),
        "generated_at": now_iso(),
        "total_items": len(items),
        "cached_ok": sum(1 for item in items if item["status"] == "ok"),
        "pending": sum(1 for item in items if item["status"] == "pending"),
        "errors": sum(1 for item in items if item["status"] == "error"),
        "items": items,
    }


def main() -> None:
    manifest = build_manifest()
    save_json(CACHE_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": str(CACHE_MANIFEST),
                "total_items": manifest["total_items"],
                "cached_ok": manifest["cached_ok"],
                "pending": manifest["pending"],
                "errors": manifest["errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
