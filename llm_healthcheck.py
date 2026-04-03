#!/usr/bin/env python3
"""
llm_healthcheck.py - Daily provider liveness check for research workloads.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_client import LLMClient

LOG_PATH = Path("logs/llm_health.log")
STATE_PATH = Path("logs/llm_health_state.json")

DEFAULT_PROVIDERS = ["ollama"]

PROMPTS = {
    "openai": {
        "system": "You are a precise coding assistant. Reply with code only.",
        "message": (
            "Return a valid one-line Python function named ok that returns 1. "
            "Code only."
        ),
    },
    "ollama": {
        "system": "Reply with one short word only.",
        "message": "Say ok.",
    },
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _append_log(payload: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_state(payload: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_provider(provider: str, timeout: int = 45) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now().isoformat()

    try:
        client = LLMClient(provider=provider)
        prompt_cfg = PROMPTS.get(provider, PROMPTS["ollama"])
        response = client.chat(
            prompt_cfg["message"],
            system=prompt_cfg["system"],
            temperature=0.1,
            max_tokens=120,
            timeout=timeout,
        )
        latency = round(time.perf_counter() - started, 3)
        preview = " ".join((response or "").strip().split())[:160]
        return {
            "provider": provider,
            "model": client._get_model(),
            "base_url": client._get_base_url(),
            "started_at": started_at,
            "latency_s": latency,
            "ok": True,
            "preview": preview,
        }
    except Exception as exc:
        latency = round(time.perf_counter() - started, 3)
        model = None
        base_url = None
        try:
            client = LLMClient(provider=provider)
            model = client._get_model()
            base_url = client._get_base_url()
        except Exception:
            pass
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "started_at": started_at,
            "latency_s": latency,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_healthcheck(providers: list[str] | None = None, timeout: int = 45) -> dict[str, Any]:
    providers = providers or list(DEFAULT_PROVIDERS)
    checks = [check_provider(provider, timeout=timeout) for provider in providers]
    payload = {
        "timestamp": _utc_now().isoformat(),
        "overall_ok": all(item["ok"] for item in checks),
        "checks": checks,
    }
    _append_log(payload)
    return payload


def maybe_run_daily_healthcheck(
    providers: list[str] | None = None,
    timeout: int = 45,
    force: bool = False,
) -> dict[str, Any]:
    today = _utc_now().date().isoformat()
    state = _read_state()
    last_attempt = state.get("last_attempt_date")

    if not force and last_attempt == today:
        return {
            "timestamp": _utc_now().isoformat(),
            "skipped": True,
            "reason": f"already ran for {today}",
            "last_attempt_date": last_attempt,
        }

    payload = run_healthcheck(providers=providers, timeout=timeout)
    state.update(
        {
            "last_attempt_date": today,
            "last_overall_ok": payload["overall_ok"],
            "last_timestamp": payload["timestamp"],
        }
    )
    _write_state(state)
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("skipped"):
        print(f"[llm-health] skipped: {payload['reason']}")
        return

    overall = "OK" if payload["overall_ok"] else "DEGRADED"
    print(f"[llm-health] {overall} at {payload['timestamp']}")
    for item in payload["checks"]:
        model = item.get("model") or "?"
        if item["ok"]:
            print(
                f"  - {item['provider']}: ok | model={model} | "
                f"latency={item['latency_s']}s | preview={item.get('preview', '')}"
            )
        else:
            print(
                f"  - {item['provider']}: fail | model={model} | "
                f"latency={item['latency_s']}s | error={item.get('error', '')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM provider healthcheck")
    parser.add_argument("--providers", nargs="+", default=None, help="Providers to test")
    parser.add_argument("--timeout", type=int, default=45, help="Per-provider timeout in seconds")
    parser.add_argument("--daily-if-needed", action="store_true", help="Run at most once per UTC day")
    parser.add_argument("--force", action="store_true", help="Ignore daily state and run now")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any provider fails")
    args = parser.parse_args()

    if args.daily_if_needed:
        payload = maybe_run_daily_healthcheck(
            providers=args.providers,
            timeout=args.timeout,
            force=args.force,
        )
    else:
        payload = run_healthcheck(
            providers=args.providers,
            timeout=args.timeout,
        )

    _print_summary(payload)

    if args.strict and not payload.get("skipped") and not payload["overall_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
