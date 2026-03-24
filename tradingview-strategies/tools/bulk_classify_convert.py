#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LLMClient


TV_ROOT = ROOT / "tradingview-strategies"
RAW_MANIFEST_PATH = TV_ROOT / "raw-pine" / "bulk" / "manifest.json"
CLASSIFY_DIR = TV_ROOT / "results" / "bulk"
STRATEGY_DIR = TV_ROOT / "python-strategies" / "bulk"
REPORT_DIR = TV_ROOT / "reports" / "bulk"

CLASSIFY_SYSTEM_PROMPT = """You classify TradingView Pine Script strategies for repo-compatible Python conversion.
Return strict JSON only with keys:
- classification: direct|partial|unsupported
- timeframe: one of 1m,5m,15m,30m,1h,4h,6h,12h,1d,1w or null
- strategy_name: short slug-safe name
- reason: short explanation
- adaptations: array of short strings

Rules:
- If Pine uses `request.security(...lookahead=barmerge.lookahead_on)` classify unsupported.
- If stop/trailing behavior is approximated to next-bar signals, classify partial and explain.
- Output only valid JSON."""

CONVERT_SYSTEM_PROMPT = """You convert TradingView Pine Script strategies into repo-compatible Python strategies.
Return strict JSON only with keys:
- python_code: full Python file content
- notes: array of short strings

Rules:
- Target interface must expose `name`, `timeframe`, `leverage`, `generate_signals(prices)`.
- Use only pandas/numpy.
- No lookahead, no future indexing, no unsupported intrabar fills.
- Keep code ASCII only.
- Output only valid JSON."""


def heuristically_unsupported(code: str) -> str | None:
    patterns = [
        (r"lookahead\s*=\s*barmerge\.lookahead_on", "uses higher-timeframe lookahead_on"),
        (r"\brequest\.economic\b", "uses TradingView economic data"),
        (r"\brequest\.quandl\b", "uses external Quandl data"),
        (r"\brequest\.security_lower_tf\b", "uses lower-timeframe requests"),
    ]
    for pattern, reason in patterns:
        if re.search(pattern, code):
            return reason
    return None


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_llm_call(system_prompt: str, user_prompt: str) -> str:
    client = LLMClient()
    return client.chat(user_prompt, system=system_prompt, temperature=0.1, max_tokens=7000)


def _llm_worker(q, system_prompt: str, user_prompt: str) -> None:
    try:
        q.put({"ok": True, "text": run_llm_call(system_prompt, user_prompt)})
    except Exception as exc:
        q.put({"ok": False, "error": str(exc)})


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and convert extracted Pine files using repo LLM client.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=90)
    args = parser.parse_args()

    raw_manifest = json.loads(RAW_MANIFEST_PATH.read_text())["items"]
    items = [item for item in raw_manifest if item["status"] == "ok"]
    items = items[args.offset: args.offset + args.limit]

    CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for item in items:
        pine_path = TV_ROOT / item["file"]
        code = pine_path.read_text(encoding="utf-8", errors="ignore")
        try:
            unsupported_reason = heuristically_unsupported(code)
            if unsupported_reason:
                payload = {
                    "classification": "unsupported",
                    "timeframe": item.get("repo_timeframe"),
                    "strategy_name": pine_path.stem,
                    "reason": unsupported_reason,
                    "adaptations": [],
                }
            else:
                user_prompt = json.dumps({
                    "metadata": {
                        "name": item.get("name"),
                        "chart_url": item.get("chart_url"),
                        "repo_timeframe_hint": item.get("repo_timeframe"),
                    },
                    "pine_code": code,
                }, ensure_ascii=True)
                raw = llm_call_with_timeout(CLASSIFY_SYSTEM_PROMPT, user_prompt, args.timeout_s)
                payload = json.loads(raw)
        except Exception as exc:
            payload = {
                "classification": "unsupported",
                "timeframe": item.get("repo_timeframe"),
                "strategy_name": pine_path.stem,
                "reason": f"classification_failed: {exc}",
                "adaptations": [],
            }

        report = {
            "chart_url": item["chart_url"],
            "name": item.get("name"),
            "pine_file": item["file"],
            **payload,
        }

        slug = payload.get("strategy_name") or pine_path.stem
        json_path = CLASSIFY_DIR / f"{slug}.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if args.convert and payload["classification"] != "unsupported":
            try:
                convert_prompt = json.dumps({
                    "metadata": {
                        "name": item.get("name"),
                        "chart_url": item.get("chart_url"),
                        "classification": payload["classification"],
                        "timeframe": payload.get("timeframe") or item.get("repo_timeframe"),
                        "reason": payload["reason"],
                        "adaptations": payload.get("adaptations", []),
                    },
                    "pine_code": code,
                }, ensure_ascii=True)
                convert_raw = llm_call_with_timeout(CONVERT_SYSTEM_PROMPT, convert_prompt, args.timeout_s)
                convert_payload = json.loads(convert_raw)
                report["conversion_notes"] = convert_payload.get("notes", [])
                py_path = STRATEGY_DIR / f"{slug}.py"
                py_path.write_text(convert_payload["python_code"], encoding="utf-8")
                try:
                    mod = load_module(py_path)
                    signals_fn = getattr(mod, "generate_signals", None)
                    ok = callable(signals_fn)
                except Exception as exc:
                    ok = False
                    report["compile_error"] = str(exc)
                report["python_file"] = str(py_path.relative_to(TV_ROOT))
                report["import_ok"] = ok
            except Exception as exc:
                report["conversion_error"] = str(exc)
            json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        md_lines = [
            f"# {report['name']}",
            "",
            f"- Source URL: {report['chart_url']}",
            f"- Pine file: `{report['pine_file']}`",
            f"- Classification: `{report['classification']}`",
            f"- Reason: {report['reason']}",
            "",
            "## Adaptations",
            "",
        ]
        for note in report.get("adaptations", []):
            md_lines.append(f"- {note}")
        if not report.get("adaptations"):
            md_lines.append("- None")
        (REPORT_DIR / f"{slug}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        summary.append({"slug": slug, "classification": report["classification"], "chart_url": report["chart_url"]})

    (CLASSIFY_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"processed": len(summary), "output_dir": str(CLASSIFY_DIR)}, indent=2))


if __name__ == "__main__":
    main()
