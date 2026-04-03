#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import BacktestConfig, BacktestResult, run_backtest
from evaluate import compute_metrics
from llm_client import LLMClient
from prepare import load_config, load_funding_rate, load_klines
from tv_backtest_settings import (
    FIXED_ORDER_FRACTION,
    FIXED_ORDER_SIZE_USD,
    INITIAL_CAPITAL_USD,
    POSITION_SIZING_LABEL,
    apply_fixed_order_size,
)


TV_ROOT = ROOT / "tradingview-strategies"
QUEUE_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"
RAW_DIR = TV_ROOT / "raw-pine" / "bulk"
RAW_MANIFEST_PATH = RAW_DIR / "manifest.json"
CLASSIFY_DIR = TV_ROOT / "results" / "bulk"
SUMMARY_PATH = CLASSIFY_DIR / "summary.json"
STRATEGY_DIR = TV_ROOT / "python-strategies" / "bulk"
REPORT_DIR = TV_ROOT / "reports" / "bulk"
BACKTESTS_PATH = TV_ROOT / "results" / "bulk-backtests.json"
STATE_PATH = TV_ROOT / "results" / "continuous-pipeline-state.json"
LOG_PATH = TV_ROOT / "logs" / "continuous-conversion.log"

CANONICAL_RAW_MANIFEST_PATH = RAW_MANIFEST_PATH
CANONICAL_SUMMARY_PATH = SUMMARY_PATH
CANONICAL_BACKTESTS_PATH = BACKTESTS_PATH
CANONICAL_STATE_PATH = STATE_PATH
CANONICAL_LOG_PATH = LOG_PATH

WORKER_ID = ""
RANK_MOD = 0
RANK_REM = 0
MIN_RANK: int | None = None
MAX_RANK: int | None = None

START_DATE = "2021-01-01"
LOAD_START = "2020-01-01"
LOAD_END = "2030-01-01"

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
- Always return a full standalone Python module starting with `#!/usr/bin/env python3`.
- `generate_signals(prices)` must return a numpy array with exactly `len(prices)` elements.
- Use only columns available from repo klines: `open_time`, `open`, `high`, `low`, `close`, `volume`.
- Do not call external APIs, read files, or import repo-specific helpers inside the strategy file.
- Do not use pandas_ta, ta-lib, scipy, numba, vectorbt, talib, or custom packages.
- Avoid `Series.iloc` inside rolling/apply lambdas that can break with ndarray inputs.
- If Pine uses stops, take-profits, or trailing exits, translate them into next-bar target-position changes.
- If higher-timeframe logic is needed, only preserve it when it maps honestly to the repo timeframe hint.
- Output only valid JSON."""

REPAIR_SYSTEM_PROMPT = """You are repairing a previously generated Python conversion of a TradingView Pine Script strategy.
Return strict JSON only with keys:
- python_code: full Python file content
- notes: array of short strings

Rules:
- Fix the reported import/runtime/backtest error without changing the intended strategy more than necessary.
- Preserve the repo contract: module-level `name`, `timeframe`, optional `leverage`, and `generate_signals(prices)`.
- `generate_signals(prices)` must return a numpy array with exactly `len(prices)` elements.
- Use only pandas/numpy and repo OHLCV columns.
- No lookahead, no future indexing, no same-bar fills.
- Keep code ASCII only.
- Output only valid JSON."""

COMMON_FAILURE_CASES = [
    "Signal length mismatch versus prices length.",
    "Missing module-level `name` or `timeframe`.",
    "generate_signals is missing or not callable.",
    "Using unavailable columns beyond open_time/open/high/low/close/volume.",
    "Returning a pandas Series instead of a numpy array.",
    "NaN-heavy logic that breaks trade state or array comparisons.",
    "Intrabar stop/target logic treated as same-bar fills instead of next-bar signal changes.",
    "request.security higher-timeframe logic aligned dishonestly or requiring lookahead.",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(message: str) -> None:
    worker = f" worker={WORKER_ID}" if WORKER_ID else ""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}]{worker} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def configure_worker_paths(worker_id: str) -> None:
    global RAW_MANIFEST_PATH, SUMMARY_PATH, BACKTESTS_PATH, STATE_PATH, LOG_PATH
    if not worker_id:
        return
    RAW_MANIFEST_PATH = RAW_DIR / f"manifest.{worker_id}.json"
    SUMMARY_PATH = CLASSIFY_DIR / f"summary.{worker_id}.json"
    BACKTESTS_PATH = TV_ROOT / "results" / f"bulk-backtests.{worker_id}.json"
    STATE_PATH = TV_ROOT / "results" / f"continuous-pipeline-state.{worker_id}.json"
    LOG_PATH = TV_ROOT / "logs" / f"continuous-conversion.{worker_id}.log"


def queue_rank(row: dict[str, Any]) -> int:
    try:
        return int(row.get("queue_rank", 10**9))
    except Exception:
        return 10**9


def row_in_shard(row: dict[str, Any]) -> bool:
    rank = queue_rank(row)
    if MIN_RANK is not None and rank < MIN_RANK:
        return False
    if MAX_RANK is not None and rank > MAX_RANK:
        return False
    if RANK_MOD > 0 and rank % RANK_MOD != RANK_REM:
        return False
    return True


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


def ensure_llm_env(model: str) -> dict[str, str]:
    load_dotenv(ROOT / ".env")
    if model:
        os.environ["OLLAMA_MODEL"] = model
    base_url = os.environ.get("OLLAMA_BASE_URL", "")
    is_local = base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")
    return {
        "provider": "ollama",
        "base_url": base_url,
        "model": os.environ.get("OLLAMA_MODEL", ""),
        "api_key_present": bool(os.environ.get("OLLAMA_API_KEY")) or is_local,
    }


def run_cmd(*args: str, timeout_s: int = 180) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def browser_eval(session: str, js: str) -> str:
    return run_cmd("agent-browser", "--session", session, "eval", js, timeout_s=90)


def close_browser_session(session: str) -> None:
    try:
        run_cmd("agent-browser", "--session", session, "close", timeout_s=30)
    except Exception:
        pass


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def session_name(prefix: str, url: str) -> str:
    digest = __import__("hashlib").sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or "strategy"


def extract_code(url: str, session: str) -> dict[str, Any]:
    run_cmd("agent-browser", "--session", session, "open", url, timeout_s=120)
    run_cmd("agent-browser", "--session", session, "click", "#code", timeout_s=60)
    run_cmd("agent-browser", "--session", session, "wait", "1500", timeout_s=30)
    raw = browser_eval(
        session,
        r"""(() => {
            const text = document.body.innerText || "";
            const start = text.indexOf("//@version");
            const endMarker = "Open-source script";
            const end = text.indexOf(endMarker, start >= 0 ? start : 0);
            const code = start >= 0 ? (end > start ? text.slice(start, end) : text.slice(start)) : "";
            return JSON.stringify({
              url: location.href,
              title: document.title,
              code,
              line_count: code ? code.split("\n").length : 0
            });
        })()""",
    )
    payload = json.loads(raw)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


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


def validate_strategy_file(path: Path) -> tuple[bool, str]:
    try:
        module = load_module(path)
    except Exception as exc:
        return False, f"import failed: {exc}"
    if not isinstance(getattr(module, "name", None), str) or not getattr(module, "name", "").strip():
        return False, "missing module-level name"
    if not isinstance(getattr(module, "timeframe", None), str) or not getattr(module, "timeframe", "").strip():
        return False, "missing module-level timeframe"
    if not callable(getattr(module, "generate_signals", None)):
        return False, "generate_signals not callable"
    return True, ""


def run_llm_call(system_prompt: str, user_prompt: str) -> str:
    client = LLMClient(provider="ollama")
    return client.chat(user_prompt, system=system_prompt, temperature=0.1, max_tokens=7000)


def _llm_worker(queue: Any, system_prompt: str, user_prompt: str) -> None:
    try:
        queue.put({"ok": True, "text": run_llm_call(system_prompt, user_prompt)})
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


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def load_queue() -> list[dict[str, Any]]:
    payload = load_json(QUEUE_PATH, {"items": []})
    return [row for row in payload.get("items", []) if row.get("repo_timeframe") and row_in_shard(row)]


def load_raw_manifest() -> dict[str, Any]:
    return load_json(RAW_MANIFEST_PATH, {"items": []})


def merge_manifest_views(*manifests: dict[str, Any]) -> dict[str, Any]:
    by_url: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for row in manifest.get("items", []):
            url = row.get("chart_url")
            if not url:
                continue
            current = by_url.get(url)
            if current is None:
                by_url[url] = row
                continue
            current_ok = current.get("status") == "ok"
            row_ok = row.get("status") == "ok"
            if row_ok and not current_ok:
                by_url[url] = row
                continue
            if row_ok == current_ok and int(row.get("line_count") or 0) >= int(current.get("line_count") or 0):
                by_url[url] = row
    items = list(by_url.values())
    items.sort(key=lambda row: queue_rank(row))
    return {"items": items}


def upsert_manifest_item(manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    items = manifest.setdefault("items", [])
    for idx, row in enumerate(items):
        if row.get("chart_url") == entry.get("chart_url"):
            items[idx] = entry
            return
    items.append(entry)


def load_reports() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_url: dict[str, dict[str, Any]] = {}
    by_python_file: dict[str, dict[str, Any]] = {}
    if not CLASSIFY_DIR.exists():
        return by_url, by_python_file
    for path in CLASSIFY_DIR.glob("*.json"):
        if path.name == "summary.json":
            continue
        try:
            payload = load_json(path, {})
        except Exception:
            continue
        if isinstance(payload, list):
            entries = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            entries = [payload]
        else:
            continue
        for entry in entries:
            chart_url = entry.get("chart_url")
            python_file = entry.get("python_file")
            if chart_url:
                by_url[chart_url] = entry
            if python_file:
                by_python_file[python_file] = entry
    return by_url, by_python_file


def load_backtests() -> list[dict[str, Any]]:
    return load_json(BACKTESTS_PATH, [])


def load_effective_backtests() -> list[dict[str, Any]]:
    rows = []
    if WORKER_ID:
        rows.extend(load_json(CANONICAL_BACKTESTS_PATH, []))
    rows.extend(load_json(BACKTESTS_PATH, []))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        python_file = row.get("python_file")
        symbol = row.get("symbol")
        if not python_file or not symbol:
            continue
        key = (python_file, symbol)
        current = by_key.get(key)
        if current is None or (row.get("generated_at") or "") >= (current.get("generated_at") or ""):
            by_key[key] = row
    merged = list(by_key.values())
    merged.sort(key=lambda row: (row.get("python_file", ""), row.get("symbol", "")))
    return merged


def load_state(model: str) -> dict[str, Any]:
    state = load_json(
        STATE_PATH,
        {
            "started_at": now_iso(),
            "items": {},
        },
    )
    state["last_seen_model"] = model
    return state


def sync_canonical_outputs() -> None:
    if not WORKER_ID:
        return
    try:
        merge_path = Path(__file__).resolve().with_name("merge_worker_outputs.py")
        spec = importlib.util.spec_from_file_location("tv_merge_worker_outputs", merge_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {merge_path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.merge_once()
    except Exception as exc:
        log_line(f"merge sync error worker={WORKER_ID} error={exc}")


def bump_attempt(state: dict[str, Any], url: str, field: str) -> int:
    items = state.setdefault("items", {})
    item = items.setdefault(url, {})
    item[field] = int(item.get(field, 0)) + 1
    item["updated_at"] = now_iso()
    return item[field]


def set_item_state(state: dict[str, Any], url: str, **updates: Any) -> None:
    items = state.setdefault("items", {})
    item = items.setdefault(url, {})
    item.update(updates)
    item["updated_at"] = now_iso()


def write_bulk_summary(reports_by_url: dict[str, dict[str, Any]], backtests: list[dict[str, Any]]) -> None:
    backtested_files = {row["python_file"] for row in backtests}
    summary = []
    report_rows = [row for row in reports_by_url.values() if isinstance(row, dict)]
    for payload in sorted(report_rows, key=lambda row: (queue_rank(row), row.get("chart_url", ""))):
        if not row_in_shard(payload):
            continue
        summary.append(
            {
                "slug": payload.get("strategy_name") or payload.get("slug") or "",
                "classification": payload.get("classification"),
                "chart_url": payload.get("chart_url"),
                "python_file": payload.get("python_file"),
                "import_ok": payload.get("import_ok"),
                "backtested": payload.get("python_file") in backtested_files if payload.get("python_file") else False,
            }
        )
    save_json(SUMMARY_PATH, summary)


def write_state_summary(
    state: dict[str, Any],
    queue_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    reports_by_url: dict[str, dict[str, Any]],
    backtests: list[dict[str, Any]],
) -> None:
    manifest_items = manifest.get("items", [])
    extracted_ok = sum(1 for row in manifest_items if row.get("status") == "ok")
    extracted_error = sum(1 for row in manifest_items if row.get("status") != "ok")
    report_rows = [row for row in reports_by_url.values() if isinstance(row, dict)]
    unsupported = sum(1 for row in report_rows if row.get("classification") == "unsupported")
    converted = sum(
        1
        for row in report_rows
        if row.get("classification") != "unsupported" and row.get("python_file") and row.get("import_ok") is True
    )
    state["summary"] = {
        "queue_supported_timeframes": len(queue_rows),
        "extracted_ok": extracted_ok,
        "extracted_error": extracted_error,
        "classified": len(reports_by_url),
        "unsupported": unsupported,
        "converted_import_ok": converted,
        "backtested_strategy_files": len({row["python_file"] for row in backtests}),
        "backtest_result_rows": len(backtests),
        "last_updated_at": now_iso(),
    }
    save_json(STATE_PATH, state)


def build_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('name') or report.get('strategy_name') or report.get('slug') or 'Strategy'}",
        "",
        f"- Source URL: {report.get('chart_url', '')}",
        f"- Pine file: `{report.get('pine_file', '')}`",
        f"- Classification: `{report.get('classification', '')}`",
        f"- Reason: {report.get('reason', '')}",
    ]
    if report.get("python_file"):
        lines.append(f"- Python file: `{report['python_file']}`")
    if report.get("timeframe"):
        lines.append(f"- Timeframe: `{report['timeframe']}`")
    if report.get("import_ok") is not None:
        lines.append(f"- Import OK: `{report.get('import_ok')}`")
    lines.extend(["", "## Adaptations", ""])
    adaptations = report.get("adaptations") or []
    if adaptations:
        for note in adaptations:
            lines.append(f"- {note}")
    else:
        lines.append("- None")
    notes = report.get("conversion_notes") or []
    if notes:
        lines.extend(["", "## Conversion Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    if report.get("conversion_error"):
        lines.extend(["", "## Conversion Error", "", f"- {report['conversion_error']}"])
    if report.get("compile_error"):
        lines.extend(["", "## Import Error", "", f"- {report['compile_error']}"])
    if report.get("backtest_error"):
        lines.extend(["", "## Backtest Error", "", f"- {report['backtest_error']}"])
    rows = report.get("backtest_results") or []
    if rows:
        lines.extend(
            [
                "",
                "## Backtest Results",
                "",
                "| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            metrics = row["metrics"]
            lines.append(
                "| {symbol} | {timeframe} | {ret:.2f} | {sharpe:.3f} | {dd:.2f} | {trades} | {win:.1f} | {pf:.2f} |".format(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    ret=float(metrics.get("total_return_pct") or 0.0),
                    sharpe=float(metrics.get("sharpe_ratio") or 0.0),
                    dd=float(metrics.get("max_drawdown_pct") or 0.0),
                    trades=int(metrics.get("num_trades") or 0),
                    win=float(metrics.get("win_rate") or 0.0),
                    pf=float(metrics.get("profit_factor") or 0.0),
                )
            )
    return "\n".join(lines) + "\n"


def write_report_files(report: dict[str, Any], slug: str) -> None:
    json_path = CLASSIFY_DIR / f"{slug}.json"
    md_path = REPORT_DIR / f"{slug}.md"
    save_json(json_path, report)
    atomic_write_text(md_path, build_report_markdown(report))


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)):
            out[key] = None if (np.isnan(value) or np.isinf(value)) else float(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def build_convert_prompt(
    row: dict[str, Any],
    report: dict[str, Any],
    pine_code: str,
    repair_error: str | None = None,
    previous_code: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "metadata": {
            "name": row.get("name"),
            "chart_url": row.get("chart_url"),
            "classification": report.get("classification"),
            "timeframe": report.get("timeframe") or row.get("repo_timeframe"),
            "reason": report.get("reason"),
            "adaptations": report.get("adaptations", []),
            "repo_constraints": COMMON_FAILURE_CASES,
        },
        "pine_code": pine_code,
    }
    if repair_error:
        payload["repair_context"] = {
            "error": repair_error,
            "previous_python_code": previous_code or "",
        }
    return json.dumps(payload, ensure_ascii=True)


def generate_conversion(
    row: dict[str, Any],
    report: dict[str, Any],
    pine_code: str,
    timeout_s: int,
    repair_error: str | None = None,
    previous_code: str | None = None,
) -> tuple[dict[str, Any], Path, str]:
    prompt = build_convert_prompt(row, report, pine_code, repair_error=repair_error, previous_code=previous_code)
    system_prompt = REPAIR_SYSTEM_PROMPT if repair_error else CONVERT_SYSTEM_PROMPT
    convert_raw = llm_call_with_timeout(system_prompt, prompt, timeout_s)
    convert_payload = json.loads(convert_raw)
    py_path = STRATEGY_DIR / f"{report['slug']}.py"
    py_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(py_path, convert_payload["python_code"])
    ok, error = validate_strategy_file(py_path)
    return convert_payload, py_path, error if not ok else ""


class BacktestRunner:
    def __init__(self) -> None:
        self.config = load_config()
        self.bt_config = BacktestConfig.from_config(self.config)
        self.price_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.funding_cache: dict[str, pd.DataFrame] = {}

    def prices(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in self.price_cache:
            self.price_cache[key] = load_klines(symbol, timeframe, LOAD_START, LOAD_END, self.config)
        return self.price_cache[key]

    def funding(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.funding_cache:
            self.funding_cache[symbol] = load_funding_rate(symbol, START_DATE, LOAD_END, self.config)
        return self.funding_cache[symbol]

    def run_for_path(self, path: Path) -> list[dict[str, Any]]:
        mod = load_module(path)
        timeframe = getattr(mod, "timeframe")
        leverage = float(getattr(mod, "leverage", 1.0))
        strategy_name = getattr(mod, "name", path.stem)
        results = []
        for symbol in ("BTCUSDT", "ETHUSDT"):
            prices_full = self.prices(symbol, timeframe)
            signals_full = np.asarray(mod.generate_signals(prices_full), dtype=np.float64)
            if len(signals_full) != len(prices_full):
                raise RuntimeError(f"Signal length mismatch for {path.name} on {symbol}")
            mask = prices_full["open_time"] >= pd.Timestamp(START_DATE, tz="UTC")
            prices = prices_full.loc[mask].reset_index(drop=True)
            signals = apply_fixed_order_size(signals_full[mask.to_numpy()])
            funding = self.funding(symbol)
            equity, returns, trades = run_backtest(signals, prices, funding, self.bt_config, leverage=leverage)
            result = BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                period="2021-now",
                equity_curve=equity,
                returns=returns,
                trades=trades,
                backtest_duration_s=0.0,
                num_bars=len(prices),
            )
            metrics = clean_metrics(compute_metrics(result))
            results.append(
                {
                    "python_file": str(path.relative_to(TV_ROOT)),
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "position_sizing": POSITION_SIZING_LABEL,
                    "position_size_fraction": FIXED_ORDER_FRACTION,
                    "position_size_usd": FIXED_ORDER_SIZE_USD,
                    "initial_capital_usd": INITIAL_CAPITAL_USD,
                    "metrics": metrics,
                }
            )
        return results


def select_pending_extract(
    queue_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    state: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    manifest_by_url = {row["chart_url"]: row for row in manifest.get("items", [])}
    pending: list[tuple[int, int, dict[str, Any]]] = []
    for row in queue_rows:
        current = manifest_by_url.get(row["chart_url"])
        if current and current.get("status") == "ok":
            continue
        attempts = int(state.get("items", {}).get(row["chart_url"], {}).get("extract_attempts", 0))
        if attempts >= max_retries:
            continue
        pending.append((attempts, queue_rank(row), row))
    pending.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in pending]


def select_pending_convert(
    manifest: dict[str, Any],
    reports_by_url: dict[str, dict[str, Any]],
    state: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    pending = []
    for row in sorted(manifest.get("items", []), key=queue_rank):
        if row.get("status") != "ok":
            continue
        url = row["chart_url"]
        report = reports_by_url.get(url)
        if report is None:
            pending.append(row)
            continue
        if report.get("classification") == "unsupported":
            continue
        if report.get("python_file") and report.get("import_ok") is True:
            continue
        attempts = int(state.get("items", {}).get(url, {}).get("classify_attempts", 0))
        if attempts < max_retries:
            pending.append(row)
    return pending


def select_pending_backtest(
    reports_by_url: dict[str, dict[str, Any]],
    backtests: list[dict[str, Any]],
    state: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    backtests_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in backtests:
        backtests_by_file.setdefault(row["python_file"], []).append(row)
    pending: list[tuple[int, int, dict[str, Any]]] = []
    for report in sorted(reports_by_url.values(), key=lambda row: (queue_rank(row), row.get("chart_url", ""))):
        if not row_in_shard(report):
            continue
        python_file = report.get("python_file")
        if not python_file or report.get("import_ok") is not True:
            continue
        attempts = int(state.get("items", {}).get(report["chart_url"], {}).get("backtest_attempts", 0))
        if attempts >= max_retries:
            continue
        rows = backtests_by_file.get(python_file, [])
        if len(rows) < 2:
            pending.append((attempts, queue_rank(report), report))
            continue
        report_updated_at = report.get("updated_at") or ""
        last_generated_at = max((row.get("generated_at") or "") for row in rows)
        if report_updated_at and (not last_generated_at or last_generated_at < report_updated_at):
            pending.append((attempts, queue_rank(report), report))
    pending.sort(key=lambda item: (item[0], item[1]))
    return [report for _, _, report in pending]


def process_extract(row: dict[str, Any], manifest: dict[str, Any], state: dict[str, Any], session_prefix: str) -> bool:
    url = row["chart_url"]
    attempt = bump_attempt(state, url, "extract_attempts")
    slug = slug_from_url(url)
    session = session_name(session_prefix, url)
    entry = {**row, "status": "error", "file": None, "line_count": 0, "error": None}
    try:
        payload = extract_code(url, session)
        code = payload.get("code") or ""
        if not code.strip():
            raise RuntimeError("No Pine code extracted from page")
        out_path = RAW_DIR / f"{slug}.pine"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_path, code)
        entry["status"] = "ok"
        entry["file"] = str(out_path.relative_to(TV_ROOT))
        entry["line_count"] = int(payload.get("line_count") or code.count("\n") + 1)
        set_item_state(state, url, last_stage="extract_ok", raw_file=entry["file"], last_error=None)
        log_line(f"extract ok rank={row.get('queue_rank')} attempt={attempt} slug={slug} lines={entry['line_count']}")
        ok = True
    except Exception as exc:
        entry["error"] = str(exc)
        set_item_state(state, url, last_stage="extract_error", last_error=str(exc))
        log_line(f"extract error rank={row.get('queue_rank')} attempt={attempt} slug={slug} error={exc}")
        ok = False
    finally:
        close_browser_session(session)
    upsert_manifest_item(manifest, entry)
    save_json(RAW_MANIFEST_PATH, manifest)
    return ok


def process_convert(
    row: dict[str, Any],
    reports_by_url: dict[str, dict[str, Any]],
    state: dict[str, Any],
    timeout_s: int,
    repair_attempts: int,
) -> dict[str, Any]:
    url = row["chart_url"]
    attempt = bump_attempt(state, url, "classify_attempts")
    pine_path = TV_ROOT / row["file"]
    code = pine_path.read_text(encoding="utf-8", errors="ignore")
    unsupported_reason = heuristically_unsupported(code)
    if unsupported_reason:
        payload = {
            "classification": "unsupported",
            "timeframe": row.get("repo_timeframe"),
            "strategy_name": pine_path.stem,
            "reason": unsupported_reason,
            "adaptations": [],
        }
    else:
        user_prompt = json.dumps(
            {
                "metadata": {
                    "name": row.get("name"),
                    "chart_url": row.get("chart_url"),
                    "repo_timeframe_hint": row.get("repo_timeframe"),
                },
                "pine_code": code,
            },
            ensure_ascii=True,
        )
        raw = llm_call_with_timeout(CLASSIFY_SYSTEM_PROMPT, user_prompt, timeout_s)
        payload = json.loads(raw)
    slug = sanitize_slug(payload.get("strategy_name") or pine_path.stem)
    report = {
        "slug": slug,
        "chart_url": row["chart_url"],
        "name": row.get("name"),
        "pine_file": row["file"],
        "classification": payload.get("classification"),
        "strategy_name": slug,
        "timeframe": payload.get("timeframe") or row.get("repo_timeframe"),
        "reason": payload.get("reason", ""),
        "adaptations": payload.get("adaptations", []),
        "queue_rank": row.get("queue_rank"),
        "updated_at": now_iso(),
    }

    if report["classification"] != "unsupported":
        last_error = ""
        previous_code = ""
        for repair_idx in range(repair_attempts + 1):
            convert_payload, py_path, validation_error = generate_conversion(
                row,
                report,
                code,
                timeout_s,
                repair_error=last_error if repair_idx > 0 else None,
                previous_code=previous_code or None,
            )
            report["conversion_notes"] = convert_payload.get("notes", [])
            report["python_file"] = str(py_path.relative_to(TV_ROOT))
            report["repair_attempts_used"] = repair_idx
            report.pop("backtest_results", None)
            report.pop("backtest_error", None)
            previous_code = convert_payload["python_code"]
            if not validation_error:
                report["import_ok"] = True
                report.pop("compile_error", None)
                report.pop("conversion_error", None)
                break
            last_error = validation_error
            report["import_ok"] = False
            report["compile_error"] = validation_error
        if report.get("import_ok") is not True:
            report["conversion_error"] = report.get("compile_error", "conversion validation failed")
    reports_by_url[url] = report
    write_report_files(report, slug)
    if report["classification"] == "unsupported":
        set_item_state(state, url, last_stage="unsupported", last_error=report["reason"])
        log_line(f"classify unsupported rank={row.get('queue_rank')} attempt={attempt} slug={slug} reason={report['reason']}")
    elif report.get("import_ok") is True:
        set_item_state(state, url, last_stage="convert_ok", python_file=report["python_file"], last_error=None)
        log_line(f"convert ok rank={row.get('queue_rank')} attempt={attempt} slug={slug} tf={report.get('timeframe')}")
    else:
        err = report.get("compile_error") or report.get("conversion_error") or report.get("reason", "")
        set_item_state(state, url, last_stage="convert_error", last_error=err)
        log_line(f"convert error rank={row.get('queue_rank')} attempt={attempt} slug={slug} error={err}")
    return report


def merge_backtests(existing: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return existing
    python_file = rows[0]["python_file"]
    remaining = [row for row in existing if row.get("python_file") != python_file]
    remaining.extend(rows)
    remaining.sort(key=lambda row: (row["python_file"], row["symbol"]))
    return remaining


def process_backtest(
    report: dict[str, Any],
    backtests: list[dict[str, Any]],
    state: dict[str, Any],
    runner: BacktestRunner,
    timeout_s: int,
    repair_attempts: int,
) -> list[dict[str, Any]]:
    url = report["chart_url"]
    attempt = bump_attempt(state, url, "backtest_attempts")
    py_path = TV_ROOT / report["python_file"]
    rows = None
    last_error = ""
    pine_path = TV_ROOT / report["pine_file"]
    pine_code = pine_path.read_text(encoding="utf-8", errors="ignore")
    previous_code = py_path.read_text(encoding="utf-8", errors="ignore")
    row_stub = {
        "name": report.get("name"),
        "chart_url": report.get("chart_url"),
        "repo_timeframe": report.get("timeframe"),
    }
    for repair_idx in range(repair_attempts + 1):
        try:
            rows = runner.run_for_path(py_path)
            report["runtime_repair_attempts_used"] = repair_idx
            report.pop("backtest_error", None)
            break
        except Exception as exc:
            last_error = str(exc)
            report["backtest_error"] = last_error
            if repair_idx >= repair_attempts:
                raise
            convert_payload, py_path, validation_error = generate_conversion(
                row_stub,
                report,
                pine_code,
                timeout_s,
                repair_error=f"Backtest/runtime error: {last_error}",
                previous_code=previous_code,
            )
            previous_code = convert_payload["python_code"]
            report["conversion_notes"] = convert_payload.get("notes", [])
            report["python_file"] = str(py_path.relative_to(TV_ROOT))
            if validation_error:
                report["import_ok"] = False
                report["compile_error"] = validation_error
                raise RuntimeError(f"repair validation failed: {validation_error}")
            report["import_ok"] = True
            report.pop("compile_error", None)
            write_report_files(report, report["slug"])
    if rows is None:
        raise RuntimeError(last_error or "backtest failed")
    generated_at = now_iso()
    for row in rows:
        row["generated_at"] = generated_at
    backtests = merge_backtests(backtests, rows)
    save_json(BACKTESTS_PATH, backtests)
    report["backtest_results"] = rows
    report["updated_at"] = now_iso()
    write_report_files(report, report["slug"])
    set_item_state(state, url, last_stage="backtest_ok", last_error=None)
    log_line(f"backtest ok attempt={attempt} slug={report['slug']} rows={len(rows)}")
    return backtests


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous TradingView Pine -> Python conversion pipeline.")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--rank-mod", type=int, default=0)
    parser.add_argument("--rank-rem", type=int, default=0)
    parser.add_argument("--min-rank", type=int, default=None)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--extract-batch", type=int, default=2)
    parser.add_argument("--convert-batch", type=int, default=1)
    parser.add_argument("--backtest-batch", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--session-prefix", default="tv-cont")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    global WORKER_ID, RANK_MOD, RANK_REM, MIN_RANK, MAX_RANK
    WORKER_ID = args.worker_id.strip()
    RANK_MOD = max(0, int(args.rank_mod or 0))
    RANK_REM = int(args.rank_rem or 0)
    MIN_RANK = args.min_rank
    MAX_RANK = args.max_rank
    if RANK_MOD and not (0 <= RANK_REM < RANK_MOD):
        raise ValueError(f"--rank-rem must satisfy 0 <= rank-rem < rank-mod, got {RANK_REM} vs {RANK_MOD}")
    configure_worker_paths(WORKER_ID)

    env_info = ensure_llm_env(args.model)
    if not env_info["api_key_present"]:
        raise RuntimeError("OLLAMA_API_KEY is not set after loading .env")

    mp.freeze_support()

    log_line(
        "pipeline start model={model} base_url={base_url}".format(
            model=env_info["model"] or "<unset>",
            base_url=env_info["base_url"] or "<unset>",
        )
    )

    runner = BacktestRunner()

    while True:
        progress = False
        queue_rows = load_queue()
        local_manifest = load_raw_manifest()
        effective_manifest = merge_manifest_views(load_json(CANONICAL_RAW_MANIFEST_PATH, {"items": []}), local_manifest)
        reports_by_url, _ = load_reports()
        local_backtests = load_backtests()
        effective_backtests = load_effective_backtests()
        state = load_state(env_info["model"])

        pending_extract = select_pending_extract(queue_rows, effective_manifest, state, args.max_retries)[: args.extract_batch]
        for row in pending_extract:
            progress = process_extract(row, local_manifest, state, args.session_prefix) or progress

        reports_by_url, _ = load_reports()
        effective_manifest = merge_manifest_views(load_json(CANONICAL_RAW_MANIFEST_PATH, {"items": []}), load_raw_manifest())
        pending_convert = select_pending_convert(effective_manifest, reports_by_url, state, args.max_retries)[: args.convert_batch]
        for row in pending_convert:
            try:
                process_convert(row, reports_by_url, state, args.timeout_s, args.repair_attempts)
                progress = True
            except Exception as exc:
                url = row["chart_url"]
                set_item_state(state, url, last_stage="convert_error", last_error=str(exc))
                log_line(f"convert error rank={row.get('queue_rank')} slug={slug_from_url(url)} error={exc}")

        reports_by_url, _ = load_reports()
        local_backtests = load_backtests()
        effective_backtests = load_effective_backtests()
        pending_backtest = select_pending_backtest(reports_by_url, effective_backtests, state, args.max_retries)[: args.backtest_batch]
        for report in pending_backtest:
            try:
                local_backtests = process_backtest(report, local_backtests, state, runner, args.timeout_s, args.repair_attempts)
                progress = True
            except Exception as exc:
                set_item_state(state, report["chart_url"], last_stage="backtest_error", last_error=str(exc))
                log_line(f"backtest error slug={report['slug']} error={exc}")

        reports_by_url, _ = load_reports()
        local_manifest = load_raw_manifest()
        local_backtests = load_backtests()
        write_bulk_summary(reports_by_url, local_backtests)
        write_state_summary(state, queue_rows, local_manifest, reports_by_url, local_backtests)
        sync_canonical_outputs()

        if args.once:
            break
        if not progress:
            log_line(f"idle sleep={args.poll_s}s")
            time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
