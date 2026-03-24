#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
RESULTS_DIR = TV_ROOT / "results"
LOG_DIR = TV_ROOT / "logs"
STATUS_PATH = RESULTS_DIR / "conversion-supervisor.json"
PIPELINE = ROOT / ".venv" / "bin" / "python"
PIPELINE_SCRIPT = TV_ROOT / "tools" / "run_conversion_pipeline.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def canonical_summary() -> dict[str, Any]:
    state = load_json(RESULTS_DIR / "continuous-pipeline-state.json", {})
    return state.get("summary", {})


def worker_state(worker_id: str) -> dict[str, Any]:
    return load_json(RESULTS_DIR / f"continuous-pipeline-state.{worker_id}.json", {}).get("summary", {})


def done(summary: dict[str, Any]) -> bool:
    queue = int(summary.get("queue_supported_timeframes") or 0)
    classified = int(summary.get("classified") or 0)
    unsupported = int(summary.get("unsupported") or 0)
    backtested = int(summary.get("backtested_strategy_files") or 0)
    return queue > 0 and classified >= queue and backtested + unsupported >= queue


def start_worker(worker_id: str, rank_mod: int, rank_rem: int, poll_s: int) -> subprocess.Popen[str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"supervisor-{worker_id}.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    cmd = [
        str(PIPELINE),
        str(PIPELINE_SCRIPT),
        "--model",
        "qwen3.5-plus",
        "--worker-id",
        worker_id,
        "--rank-mod",
        str(rank_mod),
        "--rank-rem",
        str(rank_rem),
        "--extract-batch",
        "1",
        "--convert-batch",
        "1",
        "--backtest-batch",
        "1",
        "--timeout-s",
        "240",
        "--repair-attempts",
        "2",
        "--poll-s",
        str(poll_s),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep TradingView conversion workers running and persist status.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-s", type=int, default=15)
    parser.add_argument("--status-path", default=str(STATUS_PATH))
    args = parser.parse_args()

    status_path = Path(args.status_path)
    procs: dict[str, subprocess.Popen[str]] = {}

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
        summary = canonical_summary()
        if done(summary):
            payload = {
                "updated_at": now_iso(),
                "status": "completed",
                "canonical_summary": summary,
                "workers": {},
            }
            save_json(status_path, payload)
            break

        workers_payload: dict[str, Any] = {}
        for rank_rem in range(args.workers):
            worker_id = f"w{rank_rem}"
            proc = procs.get(worker_id)
            if proc is None or proc.poll() is not None:
                proc = start_worker(worker_id, args.workers, rank_rem, args.poll_s)
                procs[worker_id] = proc
                action = "started" if proc.poll() is None else "failed_to_start"
            else:
                action = "running"
            workers_payload[worker_id] = {
                "pid": proc.pid,
                "state": worker_state(worker_id),
                "process_state": action if proc.poll() is None else f"exited:{proc.returncode}",
            }

        payload = {
            "updated_at": now_iso(),
            "status": "running",
            "canonical_summary": summary,
            "workers": workers_payload,
        }
        save_json(status_path, payload)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
