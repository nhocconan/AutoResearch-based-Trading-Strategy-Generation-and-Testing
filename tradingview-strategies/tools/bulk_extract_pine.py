#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
QUEUE_PATH = TV_ROOT / "crawl" / "recent-open-strategies" / "conversion-queue.json"
RAW_DIR = TV_ROOT / "raw-pine" / "bulk"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def run_cmd(*args: str) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def session_name(prefix: str, url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def browser_eval(session: str, js: str) -> str:
    return run_cmd("agent-browser", "--session", session, "eval", js)


def extract_code(url: str, session: str) -> dict:
    run_cmd("agent-browser", "--session", session, "open", url)
    run_cmd("agent-browser", "--session", session, "click", "#code")
    run_cmd("agent-browser", "--session", session, "wait", "1500")
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


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"items": []}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk TradingView Pine extractor using agent-browser.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--session-prefix", default="tv-bulk")
    parser.add_argument("--only-supported-timeframes", action="store_true")
    args = parser.parse_args()

    queue = json.loads(QUEUE_PATH.read_text())["items"]
    if args.only_supported_timeframes:
        queue = [row for row in queue if row.get("repo_timeframe")]
    queue = queue[args.offset: args.offset + args.limit]

    manifest = load_manifest()
    done_urls = {item["chart_url"] for item in manifest["items"]}

    for row in queue:
        url = row["chart_url"]
        if url in done_urls:
            continue

        slug = slug_from_url(url)
        session = session_name(args.session_prefix, url)
        entry = {**row, "status": "error", "file": None, "line_count": 0, "error": None}
        try:
            payload = extract_code(url, session)
            code = payload.get("code") or ""
            if not code.strip():
                raise RuntimeError("No Pine code extracted from page")
            out_path = RAW_DIR / f"{slug}.pine"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            entry["status"] = "ok"
            entry["file"] = str(out_path.relative_to(TV_ROOT))
            entry["line_count"] = int(payload.get("line_count") or code.count("\n") + 1)
        except Exception as exc:
            entry["error"] = str(exc)

        manifest["items"].append(entry)
        save_manifest(manifest)
        print(json.dumps({"url": url, "status": entry["status"], "file": entry["file"], "error": entry["error"]}))
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=sys.stderr)
        raise
