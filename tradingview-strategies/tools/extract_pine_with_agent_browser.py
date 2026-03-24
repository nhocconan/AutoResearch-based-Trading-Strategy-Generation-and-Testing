#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(*args: str) -> str:
    proc = subprocess.run(args, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def browser_eval(session: str, js: str) -> str:
    return run_cmd("agent-browser", "--session", session, "eval", js)


def extract_code(url: str, session: str) -> dict[str, str | int]:
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
            const code = start >= 0
              ? (end > start ? text.slice(start, end) : text.slice(start))
              : "";
            return JSON.stringify({
              url: location.href,
              title: document.title,
              code,
              line_count: code ? code.split("\n").length : 0
            });
        })()""",
    )

    try:
        payload = json.loads(raw)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload type: {type(payload).__name__}")
        return payload
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode browser eval output for {url}: {raw[:400]}") from exc


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract open-source TradingView Pine code via agent-browser.")
    parser.add_argument("urls", nargs="+", help="TradingView script URLs")
    parser.add_argument(
        "--output-dir",
        default="tradingview-strategies/raw-pine",
        help="Directory for extracted .pine files",
    )
    parser.add_argument(
        "--session-prefix",
        default="tv-extract",
        help="agent-browser session prefix",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, url in enumerate(args.urls, start=1):
        session = f"{args.session_prefix}-{idx}"
        payload = extract_code(url=url, session=session)
        slug = slug_from_url(url)
        pine_path = output_dir / f"{slug}.pine"
        pine_path.write_text(payload["code"])
        entry = {
            "url": url,
            "title": payload["title"],
            "line_count": payload["line_count"],
            "file": str(pine_path),
        }
        manifest.append(entry)
        print(json.dumps(entry))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=sys.stderr)
        raise
