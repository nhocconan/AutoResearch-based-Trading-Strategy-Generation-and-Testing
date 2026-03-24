#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
TV_ROOT = ROOT / "tradingview-strategies"
RAW_ROOT = TV_ROOT / "raw-pine"
CACHE_MANIFEST = RAW_ROOT / "cache-manifest.json"
OUTPUT_DIR = RAW_ROOT / "all"
LOG_PATH = TV_ROOT / "logs" / "extract-pine-cache.log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_repo_path(rel_file: str | None) -> Path | None:
    if not rel_file:
        return None
    path = Path(rel_file)
    if path.is_absolute():
        return path
    if rel_file.startswith("tradingview-strategies/"):
        return ROOT / rel_file
    return TV_ROOT / rel_file


def run_cmd(*args: str, timeout_s: int = 180) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def session_name(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"tv-cache-{digest}"


def browser_eval(session: str, js: str) -> str:
    return run_cmd("agent-browser", "--session", session, "eval", js, timeout_s=90)


def close_browser_session(session: str) -> None:
    try:
        run_cmd("agent-browser", "--session", session, "close", timeout_s=30)
    except Exception:
        pass


def cleanup_browser_processes() -> None:
    commands = [
        ("pkill", "-f", "agent-browser-darwin-arm64"),
        ("pkill", "-f", "Google Chrome for Testing"),
        ("pkill", "-f", "Chrome for Testing"),
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            pass


def fetch_page_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def _decode_json_string_fragment(fragment: str) -> str:
    decoded = json.loads(f'"{fragment}"')
    if "\\n" in decoded or '\\"' in decoded or "\\u" in decoded:
        try:
            decoded = decoded.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    return decoded


def extract_tree_from_html(html: str) -> str:
    escaped_marker = '\\"tree\\":\\"//@version'
    start = html.find(escaped_marker)
    if start >= 0:
        start += len('\\"tree\\":\\"')
        for end_marker in ['\\"},\\"_metainfo', '\\"},\\"_meta']:
            end = html.find(end_marker, start)
            if end > start:
                try:
                    decoded = _decode_json_string_fragment(html[start:end])
                except Exception:
                    decoded = ""
                if decoded.startswith("//@version"):
                    return decoded

    marker = '"tree":"//@version'
    start = html.find(marker)
    if start < 0:
        return ""
    pos = start + len('"tree":"')
    escaped = False
    buf: list[str] = []
    while pos < len(html):
        ch = html[pos]
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\":
            buf.append(ch)
            escaped = True
        elif ch == '"':
            break
        else:
            buf.append(ch)
        pos += 1
    try:
        decoded = _decode_json_string_fragment("".join(buf))
    except Exception:
        decoded = ""
    return decoded if decoded.startswith("//@version") else ""


def extract_code_http(url: str) -> dict[str, Any]:
    html = fetch_page_html(url)
    code = extract_tree_from_html(html)
    text_lower = html.lower()
    title_start = html.find("<title>")
    title_end = html.find("</title>", title_start + 7) if title_start >= 0 else -1
    title = html[title_start + 7 : title_end].strip() if title_start >= 0 and title_end > title_start else ""
    return {
        "url": url,
        "title": title,
        "code": code,
        "line_count": code.count("\n") + 1 if code else 0,
        "has_open_source_marker": "open-source script" in text_lower,
        "has_source_code_tab": 'id="code"' in html or "source code" in text_lower,
        "has_tree_payload": bool(code),
        "has_protected_marker": "invite-only" in text_lower or "protected script" in text_lower,
    }


def extract_payload(session: str) -> dict[str, Any]:
    raw = browser_eval(
        session,
        r"""(() => {
            const text = document.body.innerText || "";
            const start = text.indexOf("//@version");
            const endMarker = "Open-source script";
            const end = text.indexOf(endMarker, start >= 0 ? start : 0);
            const code = start >= 0 ? (end > start ? text.slice(start, end) : text.slice(start)) : "";
            const bodyLower = text.toLowerCase();
            return JSON.stringify({
              url: location.href,
              title: document.title,
              code,
              line_count: code ? code.split("\n").length : 0,
              has_code_tab_text: bodyLower.includes("code"),
              has_open_source_marker: bodyLower.includes("open-source script"),
              has_protected_marker: bodyLower.includes("invite-only") || bodyLower.includes("protected script")
            });
        })()""",
    )
    payload = json.loads(raw)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def click_code_tab(session: str) -> bool:
    raw = browser_eval(
        session,
        r"""(() => {
            const preferred = [
              document.querySelector("button#code"),
              document.querySelector("[role='tab']#code"),
              document.querySelector("[data-overflow-tooltip-text='Source code']"),
              document.querySelector("[data-overflow-tooltip-text='Code']"),
              document.querySelector("[href='#code']"),
              document.querySelector("[data-name='source-code']"),
              document.querySelector("[data-tab='code']")
            ].filter(Boolean);
            let target = preferred[0] || null;
            if (!target) {
              const selectors = ["button", "a", "[role='tab']", "[role='button']", "div"];
              const seen = new Set();
              const nodes = [];
              for (const selector of selectors) {
                for (const node of document.querySelectorAll(selector)) {
                  if (!seen.has(node)) {
                    seen.add(node);
                    nodes.push(node);
                  }
                }
              }
              target = nodes.find((node) => {
                const text = (node.innerText || node.textContent || "").trim().toLowerCase();
                return text === "code" || text === "source code" || text.includes("source code");
              }) || null;
            }
            if (!target) {
              return JSON.stringify({clicked: false, reason: "code_tab_not_found"});
            }
            target.click();
            return JSON.stringify({clicked: true, reason: "clicked_candidate"});
        })()""",
    )
    payload = json.loads(raw)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return bool(payload.get("clicked"))


def extract_code(url: str, session: str) -> dict[str, Any]:
    try:
        payload = extract_code_http(url)
        if payload.get("code", "").strip():
            return payload
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        log(f"http_extract_fallback url={url} error={exc}")
    run_cmd("agent-browser", "--session", session, "open", url, timeout_s=120)
    run_cmd("agent-browser", "--session", session, "wait", "1500", timeout_s=30)
    payload = extract_payload(session)
    if payload.get("code", "").strip():
        return payload
    clicked = click_code_tab(session)
    if clicked:
        run_cmd("agent-browser", "--session", session, "wait", "1500", timeout_s=30)
        payload = extract_payload(session)
    return payload


def select_batch_items(manifest: dict[str, Any], batch_size: int, max_retries: int) -> list[dict[str, Any]]:
    pending_first_try = []
    retryable = []
    for item in manifest.get("items", []):
        full_path = resolve_repo_path(item.get("pine_file"))
        if item.get("status") == "ok" and item.get("pine_file") and full_path and full_path.exists():
            continue
        attempts = int(item.get("extract_attempts") or 0)
        if attempts >= max_retries:
            continue
        rank = int(item.get("queue_rank") or 10**9)
        if attempts == 0:
            pending_first_try.append((rank, item))
        else:
            retryable.append((attempts, rank, item))
    pending_first_try.sort(key=lambda row: row[0])
    retryable.sort(key=lambda row: (row[0], row[1]))
    retry_quota = min(max(1, batch_size // 5), len(retryable))
    selected = [item for _, _, item in retryable[:retry_quota]]
    remaining = batch_size - len(selected)
    selected.extend(item for _, item in pending_first_try[:remaining])
    if len(selected) < batch_size:
        selected.extend(item for _, _, item in retryable[retry_quota : retry_quota + (batch_size - len(selected))])
    return selected[:batch_size]


def extract_one(item: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    url = item["chart_url"]
    slug = url.rstrip("/").split("/")[-1]
    rel_file = f"raw-pine/all/{slug}.pine"
    out_path = TV_ROOT / rel_file
    attempts_this_batch = 2
    last_error = None
    for local_attempt in range(1, attempts_this_batch + 1):
        session = f"{session_name(url)}-{local_attempt}"
        try:
            payload = extract_code(url, session)
            code = payload.get("code") or ""
            if code.strip():
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(code, encoding="utf-8")
                return {
                    "chart_url": url,
                    "status": "ok",
                    "pine_file": rel_file,
                    "line_count": int(payload.get("line_count") or code.count("\n") + 1),
                    "last_error": None,
                }
            if payload.get("has_protected_marker"):
                last_error = "Protected or invite-only marker shown on page"
            elif payload.get("has_source_code_tab"):
                last_error = "Source code tab exists but Pine tree/code was not extractable"
            elif payload.get("has_open_source_marker"):
                last_error = "Open-source page loaded but Pine code was not visible after code-tab attempts"
            else:
                last_error = "No Pine code extracted from page"
        except Exception as exc:
            last_error = str(exc)
        finally:
            close_browser_session(session)
    return {
        "chart_url": url,
        "status": "error",
        "pine_file": item.get("pine_file"),
        "line_count": int(item.get("line_count") or 0),
        "last_error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a resumable batch of TradingView Pine scripts into the local cache.")
    parser.add_argument("--manifest", default=str(CACHE_MANIFEST))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-agents", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if args.max_agents > 20:
        raise ValueError("--max-agents must be <= 20")
    if args.batch_size > 20:
        raise ValueError("--batch-size must be <= 20")

    cleanup_browser_processes()

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path, {"items": []})
    batch_items = select_batch_items(manifest, batch_size=args.batch_size, max_retries=args.max_retries)
    if not batch_items:
        print(json.dumps({"status": "idle", "message": "No pending Pine extractions in cache manifest."}, indent=2))
        return

    by_url = {item["chart_url"]: item for item in manifest.get("items", [])}
    lock = threading.Lock()

    def apply_result(result: dict[str, Any]) -> None:
        with lock:
            item = by_url[result["chart_url"]]
            item["extract_attempts"] = int(item.get("extract_attempts") or 0) + 1
            item["status"] = result["status"]
            item["pine_file"] = result.get("pine_file")
            item["line_count"] = int(result.get("line_count") or 0)
            item["last_error"] = result.get("last_error")
            item["updated_at"] = now_iso()
            manifest["generated_at"] = now_iso()
            manifest["cached_ok"] = sum(1 for row in manifest["items"] if row.get("status") == "ok")
            manifest["pending"] = sum(1 for row in manifest["items"] if row.get("status") == "pending")
            manifest["errors"] = sum(1 for row in manifest["items"] if row.get("status") == "error")
            save_json(manifest_path, manifest)

    with cf.ThreadPoolExecutor(max_workers=min(args.max_agents, len(batch_items))) as pool:
        future_map = {pool.submit(extract_one, item, Path(args.output_dir)): item for item in batch_items}
        for future in cf.as_completed(future_map):
            item = future_map[future]
            result = future.result()
            apply_result(result)
            log(
                "{status} rank={rank} slug={slug} error={error}".format(
                    status=result["status"],
                    rank=item.get("queue_rank"),
                    slug=item["chart_url"].rstrip("/").split("/")[-1],
                    error=result.get("last_error") or "",
                ).strip()
            )

    cleanup_browser_processes()

    print(
        json.dumps(
            {
                "processed": len(batch_items),
                "cached_ok": manifest["cached_ok"],
                "pending": manifest["pending"],
                "errors": manifest["errors"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
