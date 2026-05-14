#!/usr/bin/env python3
"""
Audit saved strategies for static-rule violations and prefix look-ahead.

Optionally purge invalid strategies from:
- strategies/
- docs/strategies/
- results.db
"""

import argparse
from pathlib import Path

from results_db import delete_strategy
from validator import run_prefix_lookahead_check, validate_file

STRATEGIES_DIR = Path("strategies")
DOCS_DIR = Path("docs/strategies")


def purge_strategy(strategy_name: str) -> None:
    strategy_path = STRATEGIES_DIR / f"{strategy_name}.py"
    doc_path = DOCS_DIR / f"{strategy_name}.md"
    delete_strategy(strategy_name)
    if strategy_path.exists():
        strategy_path.unlink()
    if doc_path.exists():
        doc_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit saved strategies and optionally purge invalid ones")
    parser.add_argument("--strategy", help="Single strategy name without .py", default=None)
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol used for prefix audit")
    parser.add_argument("--skip-prefix", action="store_true", help="Skip dynamic prefix look-ahead audit")
    parser.add_argument("--purge-invalid", action="store_true", help="Delete invalid strategy files/docs and DB rows")
    args = parser.parse_args()

    if args.strategy:
        paths = [STRATEGIES_DIR / f"{args.strategy}.py"]
    else:
        paths = sorted(STRATEGIES_DIR.glob("*.py"))

    total = 0
    static_fail = 0
    prefix_fail = 0
    kept = 0

    for path in paths:
        if not path.exists():
            continue
        total += 1
        strategy_name = path.stem

        validation = validate_file(str(path))
        if not validation.valid:
            static_fail += 1
            print(f"[INVALID:STATIC] {strategy_name}")
            for err in validation.errors[:5]:
                print(f"  - {err}")
            if args.purge_invalid:
                purge_strategy(strategy_name)
            continue

        if not args.skip_prefix:
            ok, message = run_prefix_lookahead_check(str(path), symbol=args.symbol)
            if not ok:
                prefix_fail += 1
                print(f"[INVALID:PREFIX] {strategy_name}")
                print(f"  - {message}")
                if args.purge_invalid:
                    purge_strategy(strategy_name)
                continue

        kept += 1
        print(f"[OK] {strategy_name}")

    print()
    print(f"Audited: {total}")
    print(f"Static invalid: {static_fail}")
    print(f"Prefix invalid: {prefix_fail}")
    print(f"Valid remaining: {kept}")
    if args.purge_invalid:
        print("Invalid strategies were purged from strategies/, docs/strategies/, and results.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
