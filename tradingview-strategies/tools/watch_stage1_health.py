#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = TV_ROOT / "tools" / "run_stage1_pine_cache.py"
RUNNER_STATE = TV_ROOT / "results" / "stage1-runner.json"
CACHE_MANIFEST = TV_ROOT / "raw-pine" / "cache-manifest.json"
WATCH_STATE = TV_ROOT / "results" / "stage1-watchdog.json"
WATCH_LOG = TV_ROOT / "logs" / "stage1-watchdog.log"
WATCH_REPORT = TV_ROOT / "reports" / "stage1_watchdog.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_stage1_pids() -> list[int]:
    proc = subprocess.run(
        "ps -ef | grep run_stage1_pine_cache.py | grep -v grep | awk '{print $2}'",
        shell=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def browser_counts() -> dict[str, int]:
    cmds = {
        "agent_browser": "ps -ef | grep agent-browser | grep -v grep | wc -l",
        "chrome_testing": "ps -ef | grep 'Google Chrome for Testing' | grep -v grep | wc -l",
    }
    counts = {}
    for key, cmd in cmds.items():
        proc = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        try:
            counts[key] = int(proc.stdout.strip() or 0)
        except Exception:
            counts[key] = 0
    return counts


def cleanup_browser_processes() -> None:
    for cmd in [
        "pkill -f 'agent-browser-darwin-arm64' || true",
        "pkill -f 'Google Chrome for Testing' || true",
    ]:
        subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True, timeout=30)


def kill_stage1_processes() -> None:
    subprocess.run(
        "pkill -f 'tradingview-strategies/tools/run_stage1_pine_cache.py' || true",
        shell=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def start_stage1_runner(max_agents: int) -> None:
    log(f"starting stage1 runner max_agents={max_agents}")
    subprocess.Popen(
        [
            str(PYTHON),
            str(RUNNER),
            "--batch-size",
            "20",
            "--max-agents",
            str(max_agents),
            "--max-retries",
            "3",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def manifest_counts() -> dict[str, int]:
    obj = load_json(CACHE_MANIFEST, {"total_items": 0, "cached_ok": 0, "pending": 0, "errors": 0})
    return {
        "total_items": int(obj.get("total_items") or 0),
        "cached_ok": int(obj.get("cached_ok") or 0),
        "pending": int(obj.get("pending") or 0),
        "errors": int(obj.get("errors") or 0),
    }


def render_report(payload: dict[str, Any], interval_s: int, stale_minutes: int) -> str:
    counts = payload.get("counts", {})
    runner_state = payload.get("runner_state", {})
    browsers = payload.get("browser_counts", {})
    next_check_at = payload.get("next_check_at", "unknown")
    lines = [
        "# Stage 1 Watchdog",
        "",
        f"- updated_at: {payload.get('updated_at', 'unknown')}",
        f"- action: {payload.get('action', 'none')}",
        f"- next_check_at: {next_check_at}",
        f"- runner_pids: {', '.join(str(pid) for pid in payload.get('runner_pids', [])) or 'none'}",
        f"- stage1_status: {runner_state.get('status', 'unknown')}",
        f"- batch_started_with: cached_ok={runner_state.get('counts_before_batch', {}).get('cached_ok', 0)} pending={runner_state.get('counts_before_batch', {}).get('pending', 0)} errors={runner_state.get('counts_before_batch', {}).get('errors', 0)}",
        f"- latest_manifest: cached_ok={counts.get('cached_ok', 0)} pending={counts.get('pending', 0)} errors={counts.get('errors', 0)} total_items={counts.get('total_items', 0)}",
        f"- browsers: agent_browser={browsers.get('agent_browser', 0)} chrome_testing={browsers.get('chrome_testing', 0)}",
        f"- monitor_interval_seconds: {interval_s}",
        f"- stale_restart_minutes: {stale_minutes}",
        "",
        "This file is rewritten by the 15-minute watchdog. If Stage 1 stalls or the runner exits while pending work remains, the watchdog kills leftover browser processes and restarts Stage 1.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch Stage 1 every 15 minutes and auto-fix stalls.")
    parser.add_argument("--interval-s", type=int, default=900)
    parser.add_argument("--max-agents", type=int, default=4)
    parser.add_argument("--stale-minutes", type=int, default=20)
    args = parser.parse_args()

    while True:
        now = now_iso()
        runner_state = load_json(RUNNER_STATE, {})
        watch_state = load_json(WATCH_STATE, {})
        counts = manifest_counts()
        pids = list_stage1_pids()
        browsers = browser_counts()

        action = "none"
        stale = False
        last_cached_ok = int(watch_state.get("counts", {}).get("cached_ok") or 0)
        updated_at = runner_state.get("updated_at")
        if updated_at:
            updated_ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - updated_ts).total_seconds()
            stale = age_s > args.stale_minutes * 60

        if counts["pending"] > 0 and not pids:
            cleanup_browser_processes()
            start_stage1_runner(args.max_agents)
            action = "restarted_missing_runner"
        elif counts["pending"] > 0 and stale and counts["cached_ok"] <= last_cached_ok:
            kill_stage1_processes()
            cleanup_browser_processes()
            start_stage1_runner(args.max_agents)
            action = "restarted_stale_runner"
        elif counts["pending"] == 0:
            cleanup_browser_processes()
            action = "completed_cleanup"

        payload = {
            "updated_at": now,
            "action": action,
            "runner_pids": pids,
            "runner_state": runner_state,
            "counts": counts,
            "browser_counts": browsers,
            "next_check_at": datetime.fromtimestamp(
                time.time() + args.interval_s, timezone.utc
            ).isoformat(),
        }
        save_json(WATCH_STATE, payload)
        save_text(WATCH_REPORT, render_report(payload, args.interval_s, args.stale_minutes))
        log(f"watch counts={counts} browsers={browsers} action={action}")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
