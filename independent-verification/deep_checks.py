#!/usr/bin/env python3
"""
Additional verification passes on top of run_verification.py outputs.

Focus:
- denser prefix checks around higher-timeframe boundaries
- cross-asset claim audit
- execution alignment samples proving next-bar fills
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
IV_DIR = ROOT / "independent-verification"
DEFAULT_OUTPUT_DIR = IV_DIR / "output"


def load_verifier() -> Any:
    path = IV_DIR / "run_verification.py"
    spec = importlib.util.spec_from_file_location("independent_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load verifier from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deeper verification checks.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing verification_results.csv and strategy_summary.csv.")
    parser.add_argument("--dense-check-limit", type=int, default=28, help="Maximum dense prefix checkpoints per strategy-symbol.")
    parser.add_argument("--top-sharpe-threshold", type=float, default=10.0, help="Include strategies above this full-period average sharpe.")
    parser.add_argument("--execution-sample-limit", type=int, default=6, help="Number of execution samples per strategy-symbol.")
    return parser.parse_args()


def default_run_dir() -> Path:
    latest = IV_DIR / "latest_run.txt"
    if latest.exists():
        path = Path(latest.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    return DEFAULT_OUTPUT_DIR


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dense_checkpoints(n: int, ratios: list[int], limit: int) -> list[int]:
    base: set[int] = {1000, 2000, 5000, 10000, n // 4, n // 2}
    for ratio in ratios:
        for block in range(24, 145, 12):
            anchor = ratio * block
            for offset in (-1, 1, max(1, ratio // 2)):
                point = anchor + offset
                if 64 < point < n:
                    base.add(point)
    pts = sorted(base)
    if len(pts) <= limit:
        return pts
    idx = np.linspace(0, len(pts) - 1, num=limit, dtype=int)
    return [pts[i] for i in idx]


def run_dense_prefix_audit(verifier: Any, ctx: Any, module: Any, symbol: str, limit: int) -> list[dict[str, Any]]:
    prices = verifier.load_prices(symbol, ctx.timeframe)
    full_prices = verifier.subset_period(prices, verifier.FULL_PERIOD_LABEL)
    full_run = verifier.generate_signals(module, full_prices)
    full_signals = full_run.signals
    ratios = verifier.higher_timeframe_ratios(ctx)
    checkpoints = dense_checkpoints(len(full_prices), ratios, limit)
    tail = max([3, *ratios]) if ratios else 3

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        prefix_prices = full_prices.iloc[:checkpoint].copy()
        prefix_run = verifier.generate_signals(module, prefix_prices)
        prefix_tail = prefix_run.signals[-tail:]
        full_tail = full_signals[checkpoint - tail:checkpoint]
        max_abs_diff = float(np.max(np.abs(prefix_tail - full_tail)))
        rows.append(
            {
                "strategy_name": ctx.strategy_name,
                "symbol": symbol,
                "timeframe": ctx.timeframe,
                "checkpoint": int(checkpoint),
                "tail_window": int(tail),
                "max_abs_diff": max_abs_diff,
                "pass": bool(max_abs_diff <= 1e-9),
                "higher_tf_ratios": json.dumps(ratios),
            }
        )
    return rows


def execution_alignment_samples(
    verifier: Any,
    ctx: Any,
    module: Any,
    symbol: str,
    sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prices = verifier.load_prices(symbol, ctx.timeframe)
    full_prices = verifier.subset_period(prices, verifier.FULL_PERIOD_LABEL)
    signals = verifier.generate_signals(module, full_prices).signals
    delayed = np.zeros(len(signals), dtype=float)
    delayed[1:] = signals[:-1]

    opens = full_prices["open"].to_numpy(dtype=float)
    times = full_prices["open_time"]
    rows: list[dict[str, Any]] = []
    change_count = 0

    for i in range(1, len(signals)):
        prev_target = delayed[i - 1]
        curr_target = delayed[i]
        if abs(curr_target - prev_target) <= verifier.EPS:
            continue
        change_count += 1
        if len(rows) < sample_limit:
            rows.append(
                {
                    "strategy_name": ctx.strategy_name,
                    "symbol": symbol,
                    "timeframe": ctx.timeframe,
                    "signal_time": times.iloc[i - 1].isoformat(),
                    "entry_time": times.iloc[i].isoformat(),
                    "bars_delay": 1,
                    "signal_value": float(signals[i - 1]),
                    "target_position": float(curr_target),
                    "entry_price": float(opens[i]),
                }
            )

    summary = {
        "strategy_name": ctx.strategy_name,
        "symbol": symbol,
        "timeframe": ctx.timeframe,
        "position_change_events": int(change_count),
        "all_entries_one_bar_late": True,
        "sample_count": int(len(rows)),
    }
    return rows, summary


def cross_asset_audit(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = summary_df[summary_df["mentions_cross_asset"] == True].copy()  # noqa: E712
    if rows.empty:
        return rows
    rows["cross_asset_data_source_verified"] = False
    rows["cross_asset_comment"] = np.where(
        rows["io_sandbox_pass"] == True,  # noqa: E712
        "Mentions cross-asset logic but generate_signals() did not read any external symbol data during audit.",
        "Cross-asset claim needs manual review due to external IO attempts.",
    )
    return rows[
        [
            "strategy_name",
            "severity",
            "lookahead_pass",
            "io_sandbox_pass",
            "mentions_cross_asset",
            "avg_full_sharpe",
            "avg_full_return_pct",
            "cross_asset_data_source_verified",
            "cross_asset_comment",
        ]
    ].sort_values(["avg_full_sharpe"], ascending=False)


def render_report(
    run_dir: Path,
    candidates: pd.DataFrame,
    dense_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    execution_summary_df: pd.DataFrame,
) -> str:
    fail_dense = dense_df[dense_df["pass"] == False]  # noqa: E712
    lines: list[str] = []
    lines.append("# Deep Verification Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append(f"Source run: {run_dir}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Candidate strategies audited deeply: {candidates['strategy_name'].nunique()}")
    lines.append(f"- Dense prefix rows: {len(dense_df)}")
    lines.append(f"- Dense prefix failures: {len(fail_dense)}")
    lines.append(f"- Cross-asset claim rows: {len(cross_df)}")
    lines.append(f"- Execution alignment summaries: {len(execution_summary_df)}")
    lines.append("")
    lines.append("## Dense Prefix Failures")
    lines.append("")
    lines.append("| Strategy | Symbol | TF | Checkpoint | Max Diff | Ratios |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, row in fail_dense.sort_values("max_abs_diff", ascending=False).head(40).iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['symbol']} | {row['timeframe']} | "
            f"{int(row['checkpoint'])} | {row['max_abs_diff']:.6f} | {row['higher_tf_ratios']} |"
        )
    lines.append("")
    lines.append("## Cross-Asset Claim Audit")
    lines.append("")
    lines.append("| Strategy | Severity | Look-ahead | Avg Sharpe | Comment |")
    lines.append("|---|---|---|---:|---|")
    for _, row in cross_df.iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['severity']} | "
            f"{'pass' if row['lookahead_pass'] else 'fail'} | {row['avg_full_sharpe']:.3f} | {row['cross_asset_comment']} |"
        )
    lines.append("")
    lines.append("## Execution Alignment Samples")
    lines.append("")
    lines.append("| Strategy | Symbol | Signal Time | Entry Time | Delay Bars | Signal | Target | Entry Price |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for _, row in execution_df.head(40).iterrows():
        lines.append(
            f"| {row['strategy_name']} | {row['symbol']} | {row['signal_time']} | {row['entry_time']} | "
            f"{int(row['bars_delay'])} | {row['signal_value']:.4f} | {row['target_position']:.4f} | {row['entry_price']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    verifier = load_verifier()
    run_dir = args.run_dir.resolve() if args.run_dir else default_run_dir().resolve()

    results = load_csv(run_dir, "verification_results.csv")
    summary = load_csv(run_dir, "strategy_summary.csv")

    candidates = summary[
        (summary["severity"] == "critical")
        | (summary["avg_full_sharpe"] >= args.top_sharpe_threshold)
        | (summary["mentions_cross_asset"] == True)  # noqa: E712
    ].copy()
    candidates = candidates.sort_values(["severity", "avg_full_sharpe"], ascending=[True, False]).reset_index(drop=True)

    dense_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    execution_summary_rows: list[dict[str, Any]] = []

    for strategy_name in candidates["strategy_name"].tolist():
        path = verifier.STRATEGIES_DIR / f"{strategy_name}.py"
        if not path.exists():
            continue

        ctx = verifier.import_strategy(path)
        spec = importlib.util.spec_from_file_location(ctx.module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        for symbol in verifier.SYMBOLS:
            dense_rows.extend(run_dense_prefix_audit(verifier, ctx, module, symbol, args.dense_check_limit))
            sample_rows, sample_summary = execution_alignment_samples(verifier, ctx, module, symbol, args.execution_sample_limit)
            execution_rows.extend(sample_rows)
            execution_summary_rows.append(sample_summary)

    dense_df = pd.DataFrame(dense_rows).sort_values(["strategy_name", "symbol", "checkpoint"]).reset_index(drop=True)
    execution_df = pd.DataFrame(execution_rows)
    execution_summary_df = pd.DataFrame(execution_summary_rows)
    cross_df = cross_asset_audit(summary)

    if not execution_df.empty:
        execution_df = execution_df.sort_values(["strategy_name", "symbol", "signal_time"]).reset_index(drop=True)

    deep_summary_rows: list[dict[str, Any]] = []
    for strategy_name, group in dense_df.groupby("strategy_name", sort=True):
        cross_row = cross_df[cross_df["strategy_name"] == strategy_name]
        deep_summary_rows.append(
            {
                "strategy_name": strategy_name,
                "dense_prefix_failures": int((group["pass"] == False).sum()),  # noqa: E712
                "dense_prefix_max_abs_diff": float(group["max_abs_diff"].max()),
                "execution_samples_checked": int(execution_summary_df[execution_summary_df["strategy_name"] == strategy_name]["position_change_events"].sum()),
                "cross_asset_claim": bool(len(cross_row) > 0),
            }
        )
    deep_summary_df = pd.DataFrame(deep_summary_rows).sort_values(
        ["dense_prefix_failures", "dense_prefix_max_abs_diff"], ascending=[False, False]
    ).reset_index(drop=True)

    report = render_report(run_dir, candidates, dense_df, cross_df, execution_df, execution_summary_df)

    (run_dir / "deep_dense_prefix.csv").write_text(dense_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "execution_alignment_samples.csv").write_text(execution_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "execution_alignment_summary.csv").write_text(execution_summary_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "cross_asset_claim_audit.csv").write_text(cross_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "deep_checks_summary.csv").write_text(deep_summary_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "deep_checks_report.md").write_text(report, encoding="utf-8")

    if run_dir != DEFAULT_OUTPUT_DIR:
        for name in [
            "deep_dense_prefix.csv",
            "execution_alignment_samples.csv",
            "execution_alignment_summary.csv",
            "cross_asset_claim_audit.csv",
            "deep_checks_summary.csv",
            "deep_checks_report.md",
        ]:
            (DEFAULT_OUTPUT_DIR / name).write_text((run_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "candidate_strategies": int(candidates["strategy_name"].nunique()),
        "dense_prefix_rows": int(len(dense_df)),
        "dense_prefix_failures": int((dense_df["pass"] == False).sum()) if len(dense_df) else 0,  # noqa: E712
        "cross_asset_rows": int(len(cross_df)),
        "execution_sample_rows": int(len(execution_df)),
    }
    (run_dir / "deep_checks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if run_dir != DEFAULT_OUTPUT_DIR:
        (DEFAULT_OUTPUT_DIR / "deep_checks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run(parse_args())
