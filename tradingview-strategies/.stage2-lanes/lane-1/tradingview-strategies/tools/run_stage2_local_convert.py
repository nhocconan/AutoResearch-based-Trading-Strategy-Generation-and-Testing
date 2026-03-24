#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LLMClient
from prepare import load_config, load_klines


TV_ROOT = ROOT / "tradingview-strategies"
CACHE_MANIFEST_PATH = TV_ROOT / "raw-pine" / "cache-manifest.json"
QUEUE_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"

RESULTS_DIR = TV_ROOT / "results" / "stage2-local"
REPORTS_DIR = TV_ROOT / "reports" / "stage2-local"
PYTHON_DIR = TV_ROOT / "python-strategies" / "stage2-local"
FAILED_PYTHON_DIR = PYTHON_DIR / "failed"
LOG_PATH = TV_ROOT / "logs" / "stage2-local-convert.log"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
PROGRESS_PATH = RESULTS_DIR / "progress.json"
ERRORS_PATH = RESULTS_DIR / "errors.json"
ERRORS_MD_PATH = REPORTS_DIR / "conversion_errors.md"

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d", "1w"}

RULE_DIGEST = """Repo Pine->Python rules:
- Output a standalone ASCII Python module with module-level `name`, `timeframe`, optional `leverage`, and `generate_signals(prices)`.
- `generate_signals(prices)` must return a numeric numpy.ndarray of exactly len(prices).
- Use only pandas and numpy.
- Allowed columns only: open_time, open, high, low, close, volume.
- Repo `open_time` is datetime-like. Do not blindly coerce with unit='ms' and do not floor-divide datetimes.
- No lookahead, no future indexing, no same-bar fills.
- Convert stops/targets/trailing exits into next-bar target-position changes.
- If Pine uses request.security(...lookahead=barmerge.lookahead_on) or lower-timeframe requests, classify unsupported and mark it for the lookahead blacklist. Do not rescue/adapt it into a tested strategy.
- Do not import repo helpers, read files, or call APIs inside the strategy module.
- Fair-comparison sizing is handled later by the suite; strategy code should emit only direction/position intent.
"""

INITIAL_SYSTEM_PROMPT = """You classify and convert a TradingView Pine Script strategy into repo-compatible Python.
Return strict JSON only with keys:
- classification: direct|partial|unsupported
- timeframe: one of 1m,5m,15m,30m,1h,4h,6h,12h,1d,1w or null
- strategy_name: short slug-safe name
- reason: short explanation
- adaptations: array of short strings
- python_code: full Python module content or null
- notes: array of short strings

Rules:
- Follow the attached rule digest exactly.
- If unsupported, set python_code to null and explain briefly.
- If partial, preserve honest limitations in reason/notes.
- If supported, output a full standalone Python module.
- Output JSON only with no markdown fences."""

REPAIR_SYSTEM_PROMPT = """You are repairing a previously generated Python conversion of a TradingView Pine Script strategy.
Return strict JSON only with keys:
- python_code: full Python module content
- notes: array of short strings

Repair rules:
- Fix the exact reported import/runtime/signal-generation error.
- Preserve the intended strategy logic as much as possible.
- Keep the repo contract: module-level name, timeframe, optional leverage, and generate_signals(prices).
- generate_signals(prices) must return a numpy ndarray with exactly len(prices) elements.
- Use only pandas/numpy and repo OHLCV columns.
- No lookahead, no future indexing, no same-bar fills.
- Keep code ASCII only.
- Output JSON only with no markdown fences."""


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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log_line(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def ensure_llm_env(model: str | None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    effective_model = model or os.environ.get("OPENAI_MODEL") or "qwen3.5-plus"
    os.environ["OPENAI_MODEL"] = effective_model
    return {
        "provider": "openai",
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "model": effective_model,
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    }


def _llm_worker(queue: Any, system_prompt: str, user_prompt: str) -> None:
    try:
        client = LLMClient(provider="openai")
        text = client.chat(user_prompt, system=system_prompt, temperature=0.1, max_tokens=8000)
        queue.put({"ok": True, "text": text})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def llm_call_with_timeout(system_prompt: str, user_prompt: str, timeout_s: int) -> str:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_llm_worker, args=(queue, system_prompt, user_prompt))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"LLM call timed out after {timeout_s}s")
    if queue.empty():
        raise RuntimeError("LLM call returned no payload")
    payload = queue.get()
    if not payload["ok"]:
        raise RuntimeError(payload["error"])
    return payload["text"]


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(raw[start:end + 1])


def sanitize_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "strategy"


def slug_from_url(url: str) -> str:
    return sanitize_slug(url.rstrip("/").split("/")[-1])


def heuristically_unsupported(code: str) -> str | None:
    patterns = [
        (r"lookahead\s*=\s*barmerge\.lookahead_on", "uses higher-timeframe lookahead_on"),
        (r"\brequest\.security_lower_tf\b", "uses lower-timeframe requests"),
        (r"\brequest\.economic\b", "uses TradingView economic data"),
        (r"\brequest\.quandl\b", "uses external Quandl data"),
    ]
    for pattern, reason in patterns:
        if re.search(pattern, code):
            return reason
    return None


def classify_transient_error(message: str | None) -> str | None:
    lowered = str(message or "").lower()
    for marker in TRANSIENT_ERROR_MARKERS:
        if marker in lowered:
            return marker
    return None


def normalize_report_status(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if payload.get("status") == "error":
        last_error = payload.get("last_error") or payload.get("reason")
        transient_marker = classify_transient_error(last_error)
        if transient_marker:
            payload = {**payload}
            payload["status"] = "retryable"
            payload["reason"] = payload.get("reason") or f"retryable transient failure: {transient_marker}"
    return payload


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_symbol(row: dict[str, Any], code: str) -> str:
    text = " ".join(
        [
            str(row.get("symbol_name") or ""),
            str(row.get("name") or ""),
            code[:2000],
        ]
    ).upper()
    if "ETH" in text:
        return "ETHUSDT"
    return "BTCUSDT"


def validate_strategy(py_path: Path, timeframe: str, symbol: str, config: dict[str, Any]) -> dict[str, Any]:
    module = load_module(py_path)
    name = getattr(module, "name", None)
    module_timeframe = getattr(module, "timeframe", None)
    generate_signals = getattr(module, "generate_signals", None)
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("Missing or invalid module-level `name`")
    if not isinstance(module_timeframe, str) or not module_timeframe.strip():
        raise RuntimeError("Missing or invalid module-level `timeframe`")
    if module_timeframe not in SUPPORTED_TIMEFRAMES:
        raise RuntimeError(f"Unsupported module timeframe: {module_timeframe}")
    if not callable(generate_signals):
        raise RuntimeError("Missing callable generate_signals(prices)")

    prices = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2020-01-01",
        end_date="2021-06-01",
        config=config,
    )
    if prices.empty:
        raise RuntimeError(f"No price data loaded for {symbol} {timeframe}")
    signals = generate_signals(prices.copy())
    if not isinstance(signals, np.ndarray):
        raise RuntimeError(f"generate_signals returned {type(signals).__name__}, expected numpy.ndarray")
    if len(signals) != len(prices):
        raise RuntimeError(f"Signal length mismatch: {len(signals)} != {len(prices)}")
    if not np.issubdtype(signals.dtype, np.number):
        raise RuntimeError(f"Signals dtype is not numeric: {signals.dtype}")
    finite_ratio = float(np.isfinite(signals).mean()) if len(signals) else 1.0
    return {
        "module_name": name,
        "module_timeframe": module_timeframe,
        "symbol": symbol,
        "bars_checked": int(len(prices)),
        "non_zero_signals": int(np.count_nonzero(np.nan_to_num(signals))),
        "finite_ratio": finite_ratio,
    }


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
                f"- Name: {item.get('name')}",
                f"- URL: {item.get('chart_url')}",
                f"- Pine file: `{item.get('pine_file')}`",
                f"- Attempts used: `{item.get('attempts_used')}`",
                f"- Last error: `{item.get('last_error')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def row_in_shard(row: dict[str, Any], rank_mod: int, rank_rem: int) -> bool:
    if rank_mod <= 0:
        return True
    try:
        rank = int(row.get("queue_rank") or 0)
    except Exception:
        return False
    return rank % rank_mod == rank_rem


def load_queue_candidates(rank_mod: int = 0, rank_rem: int = 0) -> list[dict[str, Any]]:
    cache_manifest = load_json(CACHE_MANIFEST_PATH, {"items": []})
    queue_payload = load_json(QUEUE_PATH, [])
    queue_rows = queue_payload.get("items", []) if isinstance(queue_payload, dict) else queue_payload
    queue_by_url = {
        row.get("chart_url"): row
        for row in queue_rows
        if isinstance(row, dict) and row.get("chart_url")
    }
    restrict_to_queue_file = ".stage2-lanes" in str(QUEUE_PATH)
    candidates: list[dict[str, Any]] = []
    for cached in cache_manifest.get("items", []):
        if cached.get("status") != "ok" or not cached.get("pine_file"):
            continue
        url = cached.get("chart_url")
        if not url:
            continue
        if restrict_to_queue_file and url not in queue_by_url:
            continue
        merged = {**cached, **queue_by_url.get(url, {})}
        merged["chart_url"] = url
        merged["pine_file"] = cached["pine_file"]
        symbol = merged.get("symbol") or {}
        if isinstance(symbol, dict):
            merged.setdefault("symbol_name", symbol.get("name") or symbol.get("full_name") or symbol.get("short_name"))
            merged.setdefault("interval", symbol.get("interval"))
        if not row_in_shard(merged, rank_mod, rank_rem):
            continue
        candidates.append(merged)
    return candidates


def existing_by_url() -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in {"summary.json", "progress.json", "errors.json"}:
            continue
        payload = normalize_report_status(load_json(path, {}))
        url = payload.get("chart_url")
        if url and payload.get("status") != "retryable":
            by_url[url] = payload
    return by_url


def rebuild_summary_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in {"summary.json", "progress.json", "errors.json"}:
            continue
        payload = normalize_report_status(load_json(path, {}))
        if not isinstance(payload, dict) or not payload.get("chart_url"):
            continue
        summary_rows.append(
            {
                "chart_url": payload.get("chart_url"),
                "slug": payload.get("slug"),
                "name": payload.get("name"),
                "status": payload.get("status"),
                "classification": payload.get("classification"),
                "timeframe": payload.get("timeframe"),
                "python_file": payload.get("python_file"),
                "last_error": payload.get("last_error"),
                "reason": payload.get("reason"),
                "attempts_used": payload.get("attempts_used"),
                "queue_rank": payload.get("queue_rank"),
            }
        )
        if payload.get("status") in {"error", "retryable"}:
            error_rows.append(
                {
                    "chart_url": payload.get("chart_url"),
                    "slug": payload.get("slug"),
                    "name": payload.get("name"),
                    "pine_file": payload.get("pine_file"),
                    "status": payload.get("status"),
                    "attempts_used": payload.get("attempts_used"),
                    "last_error": payload.get("last_error") or payload.get("reason"),
                }
            )
    summary_rows.sort(key=lambda item: (item.get("queue_rank") or 10**9, item.get("slug") or ""))
    error_rows.sort(key=lambda item: item.get("slug") or "")
    return summary_rows, error_rows


def build_prompt_context(row: dict[str, Any], pine_code: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": row.get("name"),
            "chart_url": row.get("chart_url"),
            "symbol_name": row.get("symbol_name"),
            "interval": row.get("interval"),
            "repo_timeframe_hint": row.get("repo_timeframe"),
            "author": row.get("author"),
        },
        "rule_digest": RULE_DIGEST,
        "pine_code": pine_code,
    }


def write_report_markdown(report: dict[str, Any]) -> None:
    lines = [
        f"# {report.get('name')}",
        "",
        f"- Source URL: {report.get('chart_url')}",
        f"- Pine file: `{report.get('pine_file')}`",
        f"- Classification: `{report.get('classification')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Attempts used: `{report.get('attempts_used')}`",
        f"- Result: `{report.get('status')}`",
        f"- Reason: {report.get('reason')}",
        "",
        "## Adaptations",
        "",
    ]
    for note in report.get("adaptations", []) or []:
        lines.append(f"- {note}")
    if not report.get("adaptations"):
        lines.append("- None")
    if report.get("conversion_notes"):
        lines.extend(["", "## Conversion Notes", ""])
        for note in report["conversion_notes"]:
            lines.append(f"- {note}")
    if report.get("last_error"):
        lines.extend(["", "## Last Error", "", f"`{report['last_error']}`"])
    path = REPORTS_DIR / f"{report['slug']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one(row: dict[str, Any], timeout_s: int, max_retries: int, config: dict[str, Any]) -> dict[str, Any]:
    pine_path = TV_ROOT / row["pine_file"]
    pine_code = pine_path.read_text(encoding="utf-8", errors="ignore")
    url_slug = slug_from_url(row["chart_url"])
    symbol = choose_symbol(row, pine_code)
    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "queue_rank": row.get("queue_rank"),
        "page": row.get("page"),
        "name": row.get("name"),
        "slug": url_slug,
        "chart_url": row.get("chart_url"),
        "pine_file": row["pine_file"],
        "symbol_name": row.get("symbol_name"),
        "repo_timeframe_hint": row.get("repo_timeframe"),
        "status": "error",
        "classification": None,
        "timeframe": row.get("repo_timeframe"),
        "reason": None,
        "adaptations": [],
        "attempts_used": 0,
        "attempt_log": [],
        "conversion_notes": [],
        "python_file": None,
        "failed_python_files": [],
        "validation": None,
        "last_error": None,
    }

    unsupported_reason = heuristically_unsupported(pine_code)
    if unsupported_reason:
        report.update(
            {
                "status": "unsupported",
                "classification": "unsupported",
                "reason": unsupported_reason,
            }
        )
        return report

    context = build_prompt_context(row, pine_code)
    initial_payload: dict[str, Any] | None = None
    for attempt in range(1, max_retries + 1):
        report["attempts_used"] = attempt
        try:
            raw = llm_call_with_timeout(INITIAL_SYSTEM_PROMPT, json.dumps(context, ensure_ascii=False), timeout_s)
            initial_payload = extract_json_object(raw)
            break
        except Exception as exc:
            msg = f"initial_attempt_{attempt}: {exc}"
            report["attempt_log"].append(msg)
            report["last_error"] = msg
    if initial_payload is None:
        if classify_transient_error(report["last_error"]):
            report["status"] = "retryable"
            report["reason"] = f"retryable transient failure: {classify_transient_error(report['last_error'])}"
            return report
        report["reason"] = report["last_error"] or "classification_failed"
        return report

    timeframe = initial_payload.get("timeframe") or row.get("repo_timeframe")
    classification_name = initial_payload.get("classification")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        classification_name = "unsupported"
        initial_payload["reason"] = f"unsupported or missing timeframe: {timeframe}"
    report.update(
        {
            "classification": classification_name,
            "timeframe": timeframe,
            "reason": initial_payload.get("reason"),
            "adaptations": initial_payload.get("adaptations", []),
        }
    )
    if classification_name == "unsupported":
        report["status"] = "unsupported"
        return report

    strategy_slug = sanitize_slug(initial_payload.get("strategy_name") or url_slug)
    py_path = PYTHON_DIR / f"{strategy_slug}.py"
    last_python_code = initial_payload.get("python_code") or ""

    for attempt in range(1, max_retries + 1):
        report["attempts_used"] = attempt
        try:
            if attempt == 1:
                python_code = initial_payload.get("python_code")
                notes = initial_payload.get("notes", [])
                if not python_code:
                    raise RuntimeError("initial conversion returned no python_code")
            else:
                repair_payload = {
                    **context,
                    "classification": {
                        "classification": classification_name,
                        "timeframe": timeframe,
                        "reason": report["reason"],
                        "adaptations": report["adaptations"],
                    },
                    "previous_python_code": last_python_code,
                    "last_error": report["last_error"],
                }
                raw = llm_call_with_timeout(REPAIR_SYSTEM_PROMPT, json.dumps(repair_payload, ensure_ascii=False), timeout_s)
                payload = extract_json_object(raw)
                python_code = payload["python_code"]
                notes = payload.get("notes", [])
            py_path.parent.mkdir(parents=True, exist_ok=True)
            py_path.write_text(python_code, encoding="utf-8")
            validation = validate_strategy(py_path, timeframe=timeframe, symbol=symbol, config=config)
            report.update(
                {
                    "status": "converted",
                    "slug": strategy_slug,
                    "python_file": str(py_path.relative_to(TV_ROOT)),
                    "conversion_notes": notes,
                    "validation": validation,
                    "last_error": None,
                }
            )
            return report
        except Exception as exc:
            last_python_code = locals().get("python_code", last_python_code)
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            report["last_error"] = err
            report["attempt_log"].append(f"convert_attempt_{attempt}: {err}")
            failed_path = FAILED_PYTHON_DIR / f"{strategy_slug}.attempt{attempt}.py"
            if last_python_code:
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                failed_path.write_text(last_python_code, encoding="utf-8")
                report["failed_python_files"].append(str(failed_path.relative_to(TV_ROOT)))

    report["reason"] = report["reason"] or "conversion_failed"
    if classify_transient_error(report["last_error"]):
        report["status"] = "retryable"
        report["reason"] = f"retryable transient failure: {classify_transient_error(report['last_error'])}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 local-only TradingView Pine -> Python conversion.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rank-mod", type=int, default=0, help="Optional shard modulus. 0 disables sharding.")
    parser.add_argument("--rank-rem", type=int, default=0, help="Optional shard remainder for rank-mod.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args()

    if args.rank_mod < 0:
        raise SystemExit("--rank-mod must be >= 0")
    if args.rank_mod > 0 and not (0 <= args.rank_rem < args.rank_mod):
        raise SystemExit("--rank-rem must satisfy 0 <= rank-rem < rank-mod")

    env_info = ensure_llm_env(args.model)
    config = load_config()

    candidates = load_queue_candidates(rank_mod=args.rank_mod, rank_rem=args.rank_rem)
    processed = existing_by_url()
    if not args.include_existing:
        candidates = [row for row in candidates if row.get("chart_url") not in processed]
    candidates = candidates[args.offset: args.offset + args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)

    log_line(
        f"stage2-local start candidates={len(candidates)} model={env_info['model']} "
        f"base_url_set={bool(env_info['base_url'])} retries={args.max_retries} "
        f"rank_mod={args.rank_mod} rank_rem={args.rank_rem}"
    )

    for row in candidates:
        log_line(f"processing rank={row.get('queue_rank')} url={row.get('chart_url')}")
        report = process_one(row=row, timeout_s=args.timeout_s, max_retries=args.max_retries, config=config)
        report_path = RESULTS_DIR / f"{report['slug']}.json"
        save_json(report_path, report)
        write_report_markdown(report)
        summary_rows, errors = rebuild_summary_state()
        save_json(SUMMARY_PATH, summary_rows)
        save_json(ERRORS_PATH, errors)
        ERRORS_MD_PATH.write_text(render_error_markdown(errors), encoding="utf-8")

        progress = {
            "generated_at": now_iso(),
            "model": env_info["model"],
            "provider": env_info["provider"],
            "base_url_set": bool(env_info["base_url"]),
            "queue_candidates_available": len(load_queue_candidates(rank_mod=args.rank_mod, rank_rem=args.rank_rem)),
            "processed_reports": len(summary_rows),
            "terminal_reports": sum(1 for item in summary_rows if item.get("status") in {"converted", "unsupported", "error"}),
            "converted": sum(1 for item in summary_rows if item.get("status") == "converted"),
            "unsupported": sum(1 for item in summary_rows if item.get("status") == "unsupported"),
            "errors": sum(1 for item in summary_rows if item.get("status") == "error"),
            "retryable": sum(1 for item in summary_rows if item.get("status") == "retryable"),
            "last_processed_url": report["chart_url"],
            "last_processed_slug": report["slug"],
            "rank_mod": args.rank_mod,
            "rank_rem": args.rank_rem,
        }
        save_json(PROGRESS_PATH, progress)
        log_line(
            f"done slug={report['slug']} status={report['status']} "
            f"timeframe={report['timeframe']} attempts={report['attempts_used']}"
        )

    final_summary = load_json(SUMMARY_PATH, [])
    print(
        json.dumps(
            {
                "processed_this_run": len(candidates),
                "summary_path": str(SUMMARY_PATH),
                "errors_path": str(ERRORS_PATH),
                "progress_path": str(PROGRESS_PATH),
                "total_reports": len(final_summary),
                "terminal_reports": sum(1 for item in final_summary if item.get("status") in {"converted", "unsupported", "error"}),
                "converted": sum(1 for item in final_summary if item.get("status") == "converted"),
                "unsupported": sum(1 for item in final_summary if item.get("status") == "unsupported"),
                "errors": sum(1 for item in final_summary if item.get("status") == "error"),
                "retryable": sum(1 for item in final_summary if item.get("status") == "retryable"),
                "model": env_info["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
