#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
QUEUE_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"
RAW_DIR = TV_ROOT / "raw-pine" / "bulk"
RAW_MANIFEST_PATH = RAW_DIR / "manifest.json"
CLASSIFY_DIR = TV_ROOT / "results" / "bulk"
SUMMARY_PATH = CLASSIFY_DIR / "summary.json"
BACKTESTS_PATH = TV_ROOT / "results" / "bulk-backtests.json"
STATE_PATH = TV_ROOT / "results" / "continuous-pipeline-state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def safe_rank(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 10**9


def queue_rows() -> list[dict[str, Any]]:
    payload = load_json(QUEUE_PATH, {"items": []})
    return [row for row in payload.get("items", []) if row.get("repo_timeframe")]


def merge_manifest_shards() -> dict[str, Any]:
    by_url: dict[str, dict[str, Any]] = {}
    manifest_paths = [RAW_MANIFEST_PATH] + sorted(RAW_DIR.glob("manifest.*.json"))
    for path in manifest_paths:
        if not path.exists():
            continue
        payload = load_json(path, {"items": []})
        for item in payload.get("items", []):
            url = item.get("chart_url")
            if not url:
                continue
            current = by_url.get(url)
            if current is None:
                by_url[url] = item
                continue
            current_ok = current.get("status") == "ok"
            item_ok = item.get("status") == "ok"
            if item_ok and not current_ok:
                by_url[url] = item
                continue
            if item_ok == current_ok:
                current_lines = int(current.get("line_count") or 0)
                item_lines = int(item.get("line_count") or 0)
                if item_lines >= current_lines:
                    by_url[url] = item
    if not by_url:
        for payload in load_reports().values():
            url = payload.get("chart_url")
            if not url:
                continue
            by_url[url] = {
                "chart_url": url,
                "queue_rank": payload.get("queue_rank"),
                "name": payload.get("name"),
                "status": "ok",
                "file": payload.get("pine_file"),
                "line_count": 0,
                "error": None,
            }
    items = sorted(by_url.values(), key=lambda row: safe_rank(row.get("queue_rank")))
    return {"items": items}


def load_reports() -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    if not CLASSIFY_DIR.exists():
        return by_url
    for path in sorted(CLASSIFY_DIR.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        payload = load_json(path, {})
        url = payload.get("chart_url")
        if url:
            by_url[url] = payload
    return by_url


def merge_backtest_shards(reports_by_url: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    backtest_paths = [BACKTESTS_PATH] + sorted((TV_ROOT / "results").glob("bulk-backtests.*.json"))
    for path in backtest_paths:
        if not path.exists():
            continue
        payload = load_json(path, [])
        if not isinstance(payload, list):
            continue
        for row in payload:
            python_file = row.get("python_file")
            symbol = row.get("symbol")
            if not python_file or not symbol:
                continue
            key = (python_file, symbol)
            current = by_key.get(key)
            if current is None or (row.get("generated_at") or "") >= (current.get("generated_at") or ""):
                by_key[key] = row
    for payload in reports_by_url.values():
        for row in payload.get("backtest_results") or []:
            python_file = row.get("python_file") or payload.get("python_file")
            symbol = row.get("symbol")
            if not python_file or not symbol:
                continue
            merged = dict(row)
            merged["python_file"] = python_file
            key = (python_file, symbol)
            current = by_key.get(key)
            if current is None or (merged.get("generated_at") or payload.get("updated_at") or "") >= (
                current.get("generated_at") or ""
            ):
                by_key[key] = merged
    rows = list(by_key.values())
    rows.sort(key=lambda row: (row.get("python_file", ""), row.get("symbol", "")))
    return rows


def build_summary_rows(reports_by_url: dict[str, dict[str, Any]], backtests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backtested_files = {row.get("python_file") for row in backtests if row.get("python_file")}
    rows = []
    for payload in sorted(reports_by_url.values(), key=lambda row: (safe_rank(row.get("queue_rank")), row.get("chart_url", ""))):
        rows.append(
            {
                "slug": payload.get("strategy_name") or payload.get("slug") or "",
                "classification": payload.get("classification"),
                "chart_url": payload.get("chart_url"),
                "python_file": payload.get("python_file"),
                "import_ok": payload.get("import_ok"),
                "backtested": payload.get("python_file") in backtested_files if payload.get("python_file") else False,
            }
        )
    return rows


def merge_state_shards(
    manifest: dict[str, Any],
    reports_by_url: dict[str, dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}
    started_at = ""
    models: set[str] = set()
    state_paths = [STATE_PATH] + sorted((TV_ROOT / "results").glob("continuous-pipeline-state.*.json"))
    for path in state_paths:
        if not path.exists():
            continue
        payload = load_json(path, {})
        if payload.get("started_at"):
            started_at = min(started_at, payload["started_at"]) if started_at else payload["started_at"]
        if payload.get("last_seen_model"):
            models.add(payload["last_seen_model"])
        for url, item in (payload.get("items") or {}).items():
            current = items.get(url)
            if current is None or (item.get("updated_at") or "") >= (current.get("updated_at") or ""):
                items[url] = item
            elif current is not None:
                for field in ("extract_attempts", "classify_attempts", "backtest_attempts"):
                    current[field] = max(int(current.get(field, 0) or 0), int(item.get(field, 0) or 0))
    if not items:
        backtested_files = {row.get("python_file") for row in backtests if row.get("python_file")}
        manifest_by_url = {row.get("chart_url"): row for row in manifest.get("items", []) if row.get("chart_url")}
        for url, report in reports_by_url.items():
            item = {
                "updated_at": report.get("updated_at") or now_iso(),
                "extract_attempts": 1,
                "classify_attempts": 1,
            }
            if manifest_by_url.get(url, {}).get("file"):
                item["raw_file"] = manifest_by_url[url]["file"]
            if report.get("classification") == "unsupported":
                item["last_stage"] = "unsupported"
                item["last_error"] = report.get("reason")
            elif report.get("python_file") in backtested_files:
                item["last_stage"] = "backtest_ok"
                item["python_file"] = report.get("python_file")
                item["backtest_attempts"] = 1
                item["last_error"] = None
            elif report.get("python_file") and report.get("import_ok") is True:
                item["last_stage"] = "convert_ok"
                item["python_file"] = report.get("python_file")
                item["last_error"] = report.get("compile_error") or report.get("conversion_error")
            else:
                item["last_stage"] = "convert_error"
                item["last_error"] = report.get("compile_error") or report.get("conversion_error") or report.get("reason")
            items[url] = item

    manifest_items = manifest.get("items", [])
    unsupported = sum(1 for row in reports_by_url.values() if row.get("classification") == "unsupported")
    converted = sum(
        1
        for row in reports_by_url.values()
        if row.get("classification") != "unsupported" and row.get("python_file") and row.get("import_ok") is True
    )
    state = {
        "started_at": started_at or now_iso(),
        "items": items,
        "last_seen_model": ", ".join(sorted(models)),
        "summary": {
            "queue_supported_timeframes": len(queue_rows()),
            "extracted_ok": sum(1 for row in manifest_items if row.get("status") == "ok"),
            "extracted_error": sum(1 for row in manifest_items if row.get("status") != "ok"),
            "classified": len(reports_by_url),
            "unsupported": unsupported,
            "converted_import_ok": converted,
            "backtested_strategy_files": len({row["python_file"] for row in backtests if row.get("python_file")}),
            "backtest_result_rows": len(backtests),
            "last_updated_at": now_iso(),
        },
    }
    return state


def merge_once() -> None:
    reports_by_url = load_reports()
    manifest = merge_manifest_shards()
    backtests = merge_backtest_shards(reports_by_url)
    summary_rows = build_summary_rows(reports_by_url, backtests)
    state = merge_state_shards(manifest, reports_by_url, backtests)
    save_json(RAW_MANIFEST_PATH, manifest)
    save_json(BACKTESTS_PATH, backtests)
    save_json(SUMMARY_PATH, summary_rows)
    save_json(STATE_PATH, state)
    log(
        "merged workers reports={reports} backtested_files={files} rows={rows}".format(
            reports=len(reports_by_url),
            files=state["summary"]["backtested_strategy_files"],
            rows=state["summary"]["backtest_result_rows"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard-local TradingView worker outputs into canonical files.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-s", type=int, default=20)
    args = parser.parse_args()

    while True:
        merge_once()
        if not args.watch:
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
