#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
PYTHON = ROOT / ".venv" / "bin" / "python"
BUILD_MANIFEST = TV_ROOT / "tools" / "build_pine_cache_manifest.py"
EXTRACT_BATCH = TV_ROOT / "tools" / "extract_pine_cache_batch.py"
REBUILD_PROGRESS = TV_ROOT / "tools" / "rebuild_stage_progress.py"
CACHE_MANIFEST = TV_ROOT / "raw-pine" / "cache-manifest.json"
STATUS_PATH = TV_ROOT / "results" / "stage1-runner.json"
LOG_PATH = TV_ROOT / "logs" / "stage1-runner.log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_python(script: Path, *args: str) -> str:
    proc = subprocess.run(
        [str(PYTHON), str(script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"{script.name} failed")
    return proc.stdout


def current_counts(max_retries: int) -> dict[str, Any]:
    obj = load_json(CACHE_MANIFEST, {"total_items": 0, "cached_ok": 0, "pending": 0, "errors": 0})
    items = obj.get("items", []) if isinstance(obj, dict) else []
    retryable_errors = 0
    for item in items:
        if item.get("status") != "error":
            continue
        attempts = int(item.get("extract_attempts") or 0)
        if attempts < max_retries:
            retryable_errors += 1
    return {
        "total_items": int(obj.get("total_items") or 0),
        "cached_ok": int(obj.get("cached_ok") or 0),
        "pending": int(obj.get("pending") or 0),
        "errors": int(obj.get("errors") or 0),
        "retryable_errors": retryable_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 Pine cache extraction in repeated 20-item batches.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-agents", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-s", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.batch_size > 20 or args.max_agents > 20:
        raise ValueError("Stage 1 runner enforces max 20 concurrent browser sessions.")

    while True:
        run_python(BUILD_MANIFEST)
        counts_before = current_counts(args.max_retries)
        status = {
            "updated_at": now_iso(),
            "status": "running",
            "last_batch_status": "started",
            "counts_before_batch": counts_before,
            "counts_after_batch": None,
            "batch_started_at": now_iso(),
            "batch_completed_at": None,
            "batch_exit_code": None,
        }
        save_json(STATUS_PATH, status)
        if counts_before["pending"] == 0 and counts_before["retryable_errors"] == 0:
            status["status"] = "completed"
            save_json(STATUS_PATH, status)
            run_python(REBUILD_PROGRESS)
            log("stage1 complete pending=0 retryable_errors=0")
            break

        log(
            "stage1 batch start cached_ok={cached_ok} pending={pending} errors={errors} retryable_errors={retryable_errors}".format(
                **counts_before
            )
        )
        batch_stdout = run_python(
            EXTRACT_BATCH,
            "--batch-size",
            str(args.batch_size),
            "--max-agents",
            str(args.max_agents),
            "--max-retries",
            str(args.max_retries),
        )
        counts_after = current_counts(args.max_retries)
        run_python(REBUILD_PROGRESS)
        status.update(
            {
                "updated_at": now_iso(),
                "status": "running",
                "last_batch_status": "completed",
                "counts_after_batch": counts_after,
                "last_batch_stdout": batch_stdout,
                "batch_completed_at": now_iso(),
                "batch_exit_code": 0,
            }
        )
        save_json(STATUS_PATH, status)
        log(
            "stage1 batch done cached_ok={cached_ok} pending={pending} errors={errors} retryable_errors={retryable_errors}".format(
                **counts_after
            )
        )
        if args.once:
            break
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
