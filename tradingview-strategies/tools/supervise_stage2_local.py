#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
SHARED_RUNNER = TV_ROOT / "tools" / "run_stage2_local_convert.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

QUEUE_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"
RAW_MANIFEST_PATH = TV_ROOT / "raw-pine" / "cache-manifest.json"

LANES_ROOT = TV_ROOT / ".stage2-lanes"
CANONICAL_RESULTS_DIR = TV_ROOT / "results" / "stage2-local"
CANONICAL_PROGRESS_PATH = CANONICAL_RESULTS_DIR / "progress.json"
CANONICAL_SUMMARY_PATH = CANONICAL_RESULTS_DIR / "summary.json"
CANONICAL_ERRORS_PATH = CANONICAL_RESULTS_DIR / "errors.json"
CANONICAL_ERRORS_MD_PATH = TV_ROOT / "reports" / "stage2-local" / "conversion_errors.md"
CANONICAL_RUNNER_PATH = TV_ROOT / "results" / "stage2-local-runner.json"
LOG_PATH = TV_ROOT / "logs" / "stage2-local-supervisor.log"

SHARED_ROOT_FILES = [
    ".env",
    "backtest.py",
    "config.yaml",
    "evaluate.py",
    "llm_client.py",
    "prepare.py",
    "tv_backtest_settings.py",
]

TRANSIENT_ERROR_MARKERS = (
    "error code: 429",
    "quota exceeded",
    "throttling",
    "timed out after",
    "connection error",
    "returned no payload",
    "temporary failure",
    "temporarily unavailable",
)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def classify_transient_error(message: str | None) -> str | None:
    lowered = str(message or "").lower()
    for marker in TRANSIENT_ERROR_MARKERS:
        if marker in lowered:
            return marker
    return None


def normalize_row(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if payload.get("status") == "error":
        marker = classify_transient_error(payload.get("last_error") or payload.get("reason"))
        if marker:
            payload = {**payload}
            payload["status"] = "retryable"
            payload["reason"] = payload.get("reason") or f"retryable transient failure: {marker}"
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log_line(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        safe_unlink(dst)
    dst.symlink_to(src)


def ensure_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        safe_unlink(dst)
    shutil.copy2(src, dst)


def load_queue_rows() -> list[dict[str, Any]]:
    queue_payload = load_json(QUEUE_PATH, {})
    queue_rows = queue_payload.get("items", []) if isinstance(queue_payload, dict) else queue_payload
    queue_by_url = {
        row.get("chart_url"): row
        for row in queue_rows
        if isinstance(row, dict) and row.get("chart_url")
    }
    cache_manifest = load_json(RAW_MANIFEST_PATH, {"items": []})
    rows: list[dict[str, Any]] = []
    for item in cache_manifest.get("items", []):
        if item.get("status") != "ok" or not item.get("pine_file") or not item.get("chart_url"):
            continue
        merged = {**item, **queue_by_url.get(item["chart_url"], {})}
        symbol = merged.get("symbol") or {}
        if isinstance(symbol, dict):
            merged.setdefault("symbol_name", symbol.get("name") or symbol.get("full_name") or symbol.get("short_name"))
            merged.setdefault("interval", symbol.get("interval"))
        rows.append(merged)
    return rows


def queue_rank(row: dict[str, Any]) -> int:
    try:
        return int(row.get("queue_rank") or 10**9)
    except Exception:
        return 10**9


def partition_rows(rows: list[dict[str, Any]], lanes: int) -> dict[int, list[dict[str, Any]]]:
    partitions: dict[int, list[dict[str, Any]]] = {i: [] for i in range(lanes)}
    for row in sorted(rows, key=queue_rank):
        partitions[queue_rank(row) % lanes].append(row)
    return partitions


def write_lane_queue_file(lane_root: Path, rows: list[dict[str, Any]]) -> Path:
    queue_path = lane_root / "tradingview-strategies" / "crawl" / "recent-open-strategies" / "conversion-queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_count": len(rows),
        "deduped_count": len(rows),
        "supported_timeframe_count": len(rows),
        "items": rows,
    }
    save_json(queue_path, payload)
    return queue_path


def seed_lane_results(lane_root: Path) -> None:
    lane_results = lane_root / "tradingview-strategies" / "results" / "stage2-local"
    lane_results.mkdir(parents=True, exist_ok=True)
    if not CANONICAL_RESULTS_DIR.exists():
        return
    for path in CANONICAL_RESULTS_DIR.glob("*.json"):
        shutil.copy2(path, lane_results / path.name)


def build_lane_workspace(lane_id: str, rows: list[dict[str, Any]]) -> Path:
    lane_root = LANES_ROOT / lane_id
    lane_tv_root = lane_root / "tradingview-strategies"
    (lane_tv_root / "tools").mkdir(parents=True, exist_ok=True)
    (lane_tv_root / "crawl" / "recent-open-strategies").mkdir(parents=True, exist_ok=True)
    (lane_tv_root / "results").mkdir(parents=True, exist_ok=True)
    (lane_tv_root / "reports").mkdir(parents=True, exist_ok=True)
    (lane_tv_root / "python-strategies").mkdir(parents=True, exist_ok=True)
    (lane_root / "data").mkdir(parents=True, exist_ok=True)

    for rel in SHARED_ROOT_FILES:
        ensure_symlink(ROOT / rel, lane_root / rel)
    ensure_symlink(ROOT / "data", lane_root / "data")

    # Copy, do not symlink: the runner derives ROOT from its own real path.
    ensure_copy(SHARED_RUNNER, lane_tv_root / "tools" / "run_stage2_local_convert.py")
    ensure_symlink(TV_ROOT / "raw-pine", lane_tv_root / "raw-pine")

    seed_lane_results(lane_root)
    write_lane_queue_file(lane_root, rows)
    return lane_root


def read_lane_progress(lane_root: Path) -> dict[str, Any]:
    return load_json(lane_root / "tradingview-strategies" / "results" / "stage2-local" / "progress.json", {})


def read_lane_summary(lane_root: Path) -> list[dict[str, Any]]:
    payload = load_json(lane_root / "tradingview-strategies" / "results" / "stage2-local" / "summary.json", [])
    return [normalize_row(item) for item in payload] if isinstance(payload, list) else []


def read_lane_errors(lane_root: Path) -> list[dict[str, Any]]:
    payload = load_json(lane_root / "tradingview-strategies" / "results" / "stage2-local" / "errors.json", [])
    return [normalize_row(item) for item in payload] if isinstance(payload, list) else []


def render_error_markdown(errors: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 2 Conversion Errors",
        "",
        f"- Generated at: `{now_iso()}`",
        f"- Error count: `{len(errors)}`",
        "",
    ]
    for item in errors:
        lines.extend(
            [
                f"## {item.get('slug')}",
                "",
                f"- Lane: {item.get('lane_id')}",
                f"- Name: {item.get('name')}",
                f"- URL: {item.get('chart_url')}",
                f"- Attempts used: `{item.get('attempts_used')}`",
                f"- Last error: `{item.get('last_error')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def row_status_rank(row: dict[str, Any]) -> int:
    status = str(row.get("status") or "")
    if status == "converted":
        return 4
    if status == "unsupported":
        return 3
    if status == "error":
        return 2
    if status == "retryable":
        return 1
    return 0


def prefer_row(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return candidate
    existing_rank = row_status_rank(existing)
    candidate_rank = row_status_rank(candidate)
    if candidate_rank != existing_rank:
        return candidate if candidate_rank > existing_rank else existing
    if bool(candidate.get("python_file")) != bool(existing.get("python_file")):
        return candidate if candidate.get("python_file") else existing
    return candidate if int(candidate.get("attempts_used") or 0) >= int(existing.get("attempts_used") or 0) else existing


def load_canonical_item_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not CANONICAL_RESULTS_DIR.exists():
        return rows
    skip_names = {"summary.json", "progress.json", "errors.json"}
    for path in CANONICAL_RESULTS_DIR.glob("*.json"):
        if path.name in skip_names:
            continue
        payload = load_json(path, {})
        payload = normalize_row(payload)
        if not isinstance(payload, dict):
            continue
        if not payload.get("chart_url") or not payload.get("status"):
            continue
        rows.append(payload)
    return rows


def merge_lane_outputs(lane_roots: list[Path], queue_total: int, started_at: str, active_lanes: int) -> dict[str, Any]:
    summary_by_url: dict[str, dict[str, Any]] = {}
    errors_by_url: dict[str, dict[str, Any]] = {}
    lane_payload: dict[str, Any] = {}
    last_progress: dict[str, Any] = {}

    for lane_root in lane_roots:
        lane_id = lane_root.name
        progress = read_lane_progress(lane_root)
        summary_rows = read_lane_summary(lane_root)
        error_rows = read_lane_errors(lane_root)
        last_progress = progress or last_progress
        lane_payload[lane_id] = {
            "lane_root": str(lane_root),
            "progress": progress,
            "summary_rows": len(summary_rows),
            "error_rows": len(error_rows),
        }
        for row in summary_rows:
            url = row.get("chart_url")
            if url:
                summary_by_url[url] = prefer_row(summary_by_url.get(url), {**row, "lane_id": lane_id})
        for row in error_rows:
            url = row.get("chart_url")
            if url:
                errors_by_url[url] = {**row, "lane_id": lane_id}

    for row in load_canonical_item_rows():
        url = row.get("chart_url")
        if not url:
            continue
        summary_by_url[url] = prefer_row(summary_by_url.get(url), {**row, "lane_id": row.get("lane_id") or "canonical"})
        if row.get("status") == "error":
            errors_by_url[url] = {**row, "lane_id": row.get("lane_id") or "canonical"}
        elif row.get("status") in {"converted", "unsupported"}:
            errors_by_url.pop(url, None)

    merged_summary = sorted(
        summary_by_url.values(),
        key=lambda item: (int(item.get("queue_rank") or 10**9), item.get("chart_url") or ""),
    )
    merged_errors = sorted(errors_by_url.values(), key=lambda item: item.get("slug") or "")
    converted = sum(1 for row in merged_summary if row.get("status") == "converted")
    unsupported = sum(1 for row in merged_summary if row.get("status") == "unsupported")
    errors = sum(1 for row in merged_summary if row.get("status") == "error")
    retryable = sum(1 for row in merged_summary if row.get("status") == "retryable")
    terminal_reports = converted + unsupported + errors
    progress = {
        "generated_at": now_iso(),
        "model": last_progress.get("model") or "qwen3.5-plus",
        "provider": last_progress.get("provider") or "openai",
        "base_url_set": bool(last_progress.get("base_url_set", True)),
        "queue_candidates_available": queue_total,
        "processed_reports": len(merged_summary),
        "terminal_reports": terminal_reports,
        "converted": converted,
        "unsupported": unsupported,
        "errors": errors,
        "retryable": retryable,
        "remaining_candidates": max(queue_total - terminal_reports, 0),
        "last_processed_url": last_progress.get("last_processed_url") or "",
        "last_processed_slug": last_progress.get("last_processed_slug") or "",
        "active_lanes": active_lanes,
    }

    save_json(CANONICAL_SUMMARY_PATH, merged_summary)
    save_json(CANONICAL_ERRORS_PATH, merged_errors)
    CANONICAL_ERRORS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL_ERRORS_MD_PATH.write_text(render_error_markdown(merged_errors), encoding="utf-8")
    save_json(CANONICAL_PROGRESS_PATH, progress)
    save_json(
        CANONICAL_RUNNER_PATH,
        {
            "updated_at": now_iso(),
            "status": "running" if progress["remaining_candidates"] > 0 else "completed",
            "started_at": started_at,
            "queue_total": queue_total,
            "lanes": lane_payload,
            "aggregate": progress,
        },
    )
    return progress


def lane_worker_cmd(lane_root: Path, batch_size: int, timeout_s: int, max_retries: int, model: str | None) -> list[str]:
    runner = lane_root / "tradingview-strategies" / "tools" / "run_stage2_local_convert.py"
    cmd = [
        str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)),
        str(runner),
        "--limit",
        str(batch_size),
        "--timeout-s",
        str(timeout_s),
        "--max-retries",
        str(max_retries),
    ]
    if model:
        cmd.extend(["--model", model])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep Stage 2 local conversion running across parallel lanes.")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-s", type=int, default=1)
    parser.add_argument("--max-idle-loops", type=int, default=2)
    parser.add_argument("--model", default="qwen3.5-plus")
    args = parser.parse_args()

    queue_rows = load_queue_rows()
    if not queue_rows:
        save_json(
            CANONICAL_RUNNER_PATH,
            {
                "updated_at": now_iso(),
                "status": "idle",
                "reason": "No supported queue rows available.",
                "lanes": {},
            },
        )
        return

    partitions = partition_rows(queue_rows, args.lanes)
    started_at = now_iso()
    lane_roots = []
    for lane_idx in range(args.lanes):
        lane_id = f"lane-{lane_idx}"
        lane_root = build_lane_workspace(lane_id, partitions.get(lane_idx, []))
        lane_roots.append(lane_root)

    procs: dict[str, subprocess.Popen[str]] = {}
    lane_idle: dict[str, int] = {lane_root.name: 0 for lane_root in lane_roots}
    lane_last_processed: dict[str, int] = {
        lane_root.name: int(read_lane_progress(lane_root).get("terminal_reports") or read_lane_progress(lane_root).get("processed_reports") or 0)
        for lane_root in lane_roots
    }
    lane_queue_total: dict[str, int] = {lane_root.name: len(partitions.get(i, [])) for i, lane_root in enumerate(lane_roots)}

    def shutdown(signum: int, frame: Any) -> None:
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        any_running = False
        for lane_root in lane_roots:
            lane_id = lane_root.name
            proc = procs.get(lane_id)
            if proc is not None and proc.poll() is None:
                any_running = True
                continue

            progress = read_lane_progress(lane_root)
            current_processed = int(progress.get("terminal_reports") or progress.get("processed_reports") or 0)
            remaining = max(lane_queue_total[lane_id] - current_processed, 0)
            if remaining == 0:
                continue
            if lane_idle[lane_id] >= args.max_idle_loops:
                continue

            cmd = lane_worker_cmd(lane_root, args.batch_size, args.timeout_s, args.max_retries, args.model)
            log_line(f"starting {lane_id} command={' '.join(cmd)}")
            stdout_path = lane_root / "tradingview-strategies" / "logs" / f"stage2-local.{lane_id}.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout = stdout_path.open("a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=lane_root,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            procs[lane_id] = proc
            any_running = True

        active_lanes = sum(1 for proc in procs.values() if proc.poll() is None)
        progress = merge_lane_outputs(lane_roots, len(queue_rows), started_at, active_lanes)
        save_json(
            CANONICAL_RUNNER_PATH,
            {
                "updated_at": now_iso(),
                "status": "running" if progress["remaining_candidates"] > 0 else "completed",
                "started_at": started_at,
                "queue_total": len(queue_rows),
                "lanes": {
                    lane_root.name: {
                        "queue_total": lane_queue_total[lane_root.name],
                        "processed_reports": int(read_lane_progress(lane_root).get("processed_reports") or 0),
                        "terminal_reports": int(read_lane_progress(lane_root).get("terminal_reports") or read_lane_progress(lane_root).get("processed_reports") or 0),
                        "remaining": max(
                            lane_queue_total[lane_root.name]
                            - int(read_lane_progress(lane_root).get("terminal_reports") or read_lane_progress(lane_root).get("processed_reports") or 0),
                            0,
                        ),
                        "status": "running" if (procs.get(lane_root.name) and procs[lane_root.name].poll() is None) else "idle",
                        "process_state": "running"
                        if (procs.get(lane_root.name) and procs[lane_root.name].poll() is None)
                        else (f"exited:{procs[lane_root.name].returncode}" if procs.get(lane_root.name) else "idle"),
                        "updated_at": read_lane_progress(lane_root).get("generated_at") or "",
                    }
                    for lane_root in lane_roots
                },
                "aggregate": progress,
            },
        )

        all_done = progress["remaining_candidates"] == 0
        if all_done and not any_running:
            break

        for lane_root in lane_roots:
            lane_id = lane_root.name
            proc = procs.get(lane_id)
            if proc is None:
                continue
            if proc.poll() is None:
                continue
            lane_progress = read_lane_progress(lane_root)
            current_processed = int(lane_progress.get("terminal_reports") or lane_progress.get("processed_reports") or 0)
            if current_processed == lane_last_processed[lane_id]:
                lane_idle[lane_id] += 1
            else:
                lane_idle[lane_id] = 0
                lane_last_processed[lane_id] = current_processed
            if proc.returncode != 0:
                lane_idle[lane_id] += 1
                log_line(f"{lane_id} exited with code {proc.returncode}")
            procs.pop(lane_id, None)

        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
