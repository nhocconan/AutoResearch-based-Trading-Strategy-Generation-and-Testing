#!/usr/bin/env python3
"""
self_improvement_watchdog.py - Keep the research-improvement loop fresh.

Runs auto_concept_research.sh when the supporting discovery/review artifacts are
missing, stale, or enough new experiments have completed since the last cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "logs" / "self_improvement_watchdog_state.json"
LOG_PATH = ROOT / "logs" / "self_improvement_watchdog.log"
SUCCESS_STAMP = ROOT / "logs" / "auto_concept_research.last_success"
AGENT_STATE_PATH = ROOT / "logs" / "agent_research_state.json"
SCRIPT_PATH = ROOT / "auto_concept_research.sh"
ARTIFACTS = [
    ROOT / "docs" / "latest_strategy_discovery.md",
    ROOT / "docs" / "auto_research_review.md",
    SUCCESS_STAMP,
]

CHECK_INTERVAL_S = int(os.environ.get("SELF_IMPROVEMENT_CHECK_INTERVAL_SECONDS", "900"))
STALE_AFTER_S = int(os.environ.get("SELF_IMPROVEMENT_STALE_AFTER_SECONDS", str(12 * 3600)))
MIN_SECONDS_BETWEEN_RUNS = int(os.environ.get("SELF_IMPROVEMENT_MIN_SECONDS_BETWEEN_RUNS", str(4 * 3600)))
EXPERIMENT_DELTA_TRIGGER = int(os.environ.get("SELF_IMPROVEMENT_EXPERIMENT_DELTA_TRIGGER", "75"))
FAILURE_COOLDOWN_S = int(os.environ.get("SELF_IMPROVEMENT_FAILURE_COOLDOWN_SECONDS", str(60 * 60)))


@dataclass
class DueStatus:
    due: bool
    reasons: list[str]
    current_experiment: int
    experiments_since_success: int
    seconds_since_success: int | None
    failed_recently: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_agent_experiment() -> int:
    state = load_json(AGENT_STATE_PATH)
    for key in ("last_completed_experiment_num", "current_experiment_num", "next_experiment_num"):
        value = state.get(key)
        if isinstance(value, int):
            return value
    return 0


def read_state_times(state: dict) -> tuple[float | None, float | None]:
    last_success_ts = state.get("last_success_ts")
    last_attempt_ts = state.get("last_attempt_ts")
    return (
        float(last_success_ts) if isinstance(last_success_ts, (int, float)) else None,
        float(last_attempt_ts) if isinstance(last_attempt_ts, (int, float)) else None,
    )


def artifact_status(now_ts: float) -> tuple[list[str], int | None]:
    reasons: list[str] = []
    ages: list[int] = []
    for path in ARTIFACTS:
        if not path.exists():
            reasons.append(f"missing {path.relative_to(ROOT)}")
            continue
        age = int(now_ts - path.stat().st_mtime)
        ages.append(age)
        if age > STALE_AFTER_S:
            reasons.append(f"stale {path.relative_to(ROOT)} ({age // 3600}h old)")
    return reasons, (max(ages) if ages else None)


def compute_due_status(force: bool = False) -> DueStatus:
    now_ts = time.time()
    state = load_json(STATE_PATH)
    last_success_ts, last_attempt_ts = read_state_times(state)
    last_success_exp = int(state.get("last_success_experiment_num") or 0)
    last_attempt_exit = state.get("last_attempt_exit_code")
    current_exp = read_agent_experiment()
    exp_delta = max(0, current_exp - last_success_exp)

    reasons, max_artifact_age = artifact_status(now_ts)
    failed_recently = (
        last_attempt_exit not in (None, 0)
        and last_attempt_ts is not None
        and (now_ts - last_attempt_ts) < FAILURE_COOLDOWN_S
    )

    if not state:
        reasons.append("watchdog bootstrap state missing")

    seconds_since_success: int | None = None
    if last_success_ts is not None:
        seconds_since_success = int(now_ts - last_success_ts)
    elif SUCCESS_STAMP.exists():
        seconds_since_success = int(now_ts - SUCCESS_STAMP.stat().st_mtime)

    if exp_delta >= EXPERIMENT_DELTA_TRIGGER:
        long_enough_since_success = (
            last_success_ts is None or (now_ts - last_success_ts) >= MIN_SECONDS_BETWEEN_RUNS
        )
        if long_enough_since_success:
            reasons.append(
                f"{exp_delta} experiments since last refresh "
                f"(threshold={EXPERIMENT_DELTA_TRIGGER})"
            )

    if force:
        reasons.append("forced run requested")

    due = bool(reasons)
    if failed_recently and not force:
        due = False

    if max_artifact_age is not None and seconds_since_success is None:
        seconds_since_success = max_artifact_age

    return DueStatus(
        due=due,
        reasons=reasons,
        current_experiment=current_exp,
        experiments_since_success=exp_delta,
        seconds_since_success=seconds_since_success,
        failed_recently=failed_recently,
    )


def format_age(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h {minutes % 60}m"
    return f"{hours // 24}d {hours % 24}h"


def print_status() -> int:
    state = load_json(STATE_PATH)
    due = compute_due_status()
    last_reason = state.get("last_reason") or "n/a"
    last_result = state.get("last_result") or "n/a"
    print("Self-improvement:")
    print(f"  Due now: {'yes' if due.due else 'no'}")
    if due.reasons:
        print(f"  Due reasons: {'; '.join(due.reasons)}")
    if due.failed_recently:
        print(f"  Cooldown: active ({FAILURE_COOLDOWN_S}s after failed attempt)")
    print(f"  Current experiment: {due.current_experiment}")
    print(f"  Experiments since success: {due.experiments_since_success}")
    print(f"  Last success age: {format_age(due.seconds_since_success)}")
    print(f"  Last result: {last_result}")
    print(f"  Last reason: {last_reason}")
    return 0


def run_cycle(force: bool = False) -> int:
    due = compute_due_status(force=force)
    if not due.due:
        log(
            "skip: not due "
            f"(exp_delta={due.experiments_since_success}, last_success_age={format_age(due.seconds_since_success)})"
        )
        return 0

    state = load_json(STATE_PATH)
    state.update(
        {
            "last_attempt_at": iso_now(),
            "last_attempt_ts": time.time(),
            "last_reason": "; ".join(due.reasons),
            "last_attempt_experiment_num": due.current_experiment,
        }
    )
    save_state(state)

    log(f"triggering self-improvement cycle: {'; '.join(due.reasons)}")
    proc = subprocess.run(["bash", str(SCRIPT_PATH)], cwd=ROOT)

    state = load_json(STATE_PATH)
    state.update(
        {
            "last_attempt_at": iso_now(),
            "last_attempt_ts": time.time(),
            "last_attempt_exit_code": int(proc.returncode),
            "last_result": "success" if proc.returncode == 0 else "failed",
            "last_reason": "; ".join(due.reasons),
        }
    )
    if proc.returncode == 0:
        state.update(
            {
                "last_success_at": iso_now(),
                "last_success_ts": time.time(),
                "last_success_experiment_num": read_agent_experiment(),
            }
        )
        log("self-improvement cycle completed successfully")
    else:
        log(f"self-improvement cycle failed with exit code {proc.returncode}")

    save_state(state)
    return proc.returncode


def loop_forever() -> int:
    log(
        "watchdog started "
        f"(interval={CHECK_INTERVAL_S}s, stale_after={STALE_AFTER_S}s, "
        f"experiment_delta={EXPERIMENT_DELTA_TRIGGER}, min_gap={MIN_SECONDS_BETWEEN_RUNS}s)"
    )
    while True:
        run_cycle(force=False)
        time.sleep(CHECK_INTERVAL_S)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch and refresh the self-improvement pipeline")
    parser.add_argument("--once", action="store_true", help="Run one due-check/cycle and exit")
    parser.add_argument("--force", action="store_true", help="Force a cycle even if not due")
    parser.add_argument("--status", action="store_true", help="Print current watchdog status and exit")
    args = parser.parse_args()

    if args.status:
        return print_status()
    if args.once or args.force:
        return run_cycle(force=args.force)
    return loop_forever()


if __name__ == "__main__":
    raise SystemExit(main())
