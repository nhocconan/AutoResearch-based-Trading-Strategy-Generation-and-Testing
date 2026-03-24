#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
PROGRESS_PATH = TV_ROOT / "results" / "stage-progress.json"
PINE_CACHE_PATH = TV_ROOT / "raw-pine" / "cache-manifest.json"
CONVERSION_DIR = TV_ROOT / "results" / "bulk"
BACKTESTS_PATH = TV_ROOT / "results" / "bulk-backtests.json"
MANUAL_RESULTS_PATH = TV_ROOT / "results" / "backtest_results.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    pine_cache = load_json(PINE_CACHE_PATH, {"items": []})
    reports = []
    if CONVERSION_DIR.exists():
        for path in sorted(CONVERSION_DIR.glob("*.json")):
            if path.name.startswith("summary"):
                continue
            reports.append(load_json(path, {}))
    bulk_backtests = load_json(BACKTESTS_PATH, [])
    manual_results = load_json(MANUAL_RESULTS_PATH, {})

    payload = {
        "generated_at": now_iso(),
        "stage_1_pine_cache": {
            "total_items": len(pine_cache.get("items", [])),
            "cached_ok": sum(1 for item in pine_cache.get("items", []) if item.get("status") == "ok"),
            "pending": sum(1 for item in pine_cache.get("items", []) if item.get("status") == "pending"),
            "errors": sum(1 for item in pine_cache.get("items", []) if item.get("status") == "error"),
        },
        "stage_2_conversion": {
            "reports": len(reports),
            "unsupported": sum(1 for item in reports if item.get("classification") == "unsupported"),
            "converted_import_ok": sum(
                1
                for item in reports
                if item.get("classification") != "unsupported"
                and item.get("python_file")
                and item.get("import_ok") is True
            ),
            "convert_errors": sum(1 for item in reports if item.get("conversion_error") or item.get("compile_error")),
        },
        "stage_3_backtests": {
            "bulk_result_rows": len(bulk_backtests),
            "bulk_strategy_files": len({row.get("python_file") for row in bulk_backtests if row.get("python_file")}),
            "manual_supported": len(manual_results.get("supported", [])),
            "manual_unsupported": len(manual_results.get("unsupported", [])),
        },
    }
    save_json(PROGRESS_PATH, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
