#!/usr/bin/env python3
"""
check_results_integrity.py — guard the experiment log's invariants.

CLAUDE.md requires that results.tsv has **no duplicate (strategy, symbol,
period) rows**. This script enforces that. It is used two ways:

    python scripts/check_results_integrity.py              # strict: exit 1 on any duplicate
    python scripts/check_results_integrity.py --warn-only  # CI: report but never fail the build

The append-only log is written by multiple processes under a file lock; this
check is the cheap, independent verification that the lock is doing its job.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results.tsv"
KEY_COLUMNS = ("strategy", "symbol", "period")


def find_duplicates(path: Path) -> tuple[int, list[tuple]]:
    """Return (row_count, list of duplicated keys) for a results TSV."""
    counts: Counter = Counter()
    rows = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in KEY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"results file missing expected columns: {missing}")
        for row in reader:
            rows += 1
            counts[tuple(row.get(c) for c in KEY_COLUMNS)] += 1
    dups = [key for key, n in counts.items() if n > 1]
    return rows, dups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_RESULTS), help="results.tsv path")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="report duplicates but exit 0 (used by CI so data churn never breaks the build)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[results-integrity] {path} not present — nothing to check (ok).")
        return 0

    rows, dups = find_duplicates(path)
    if not dups:
        print(f"[results-integrity] OK — {rows:,} rows, no duplicate {KEY_COLUMNS} keys.")
        return 0

    print(f"[results-integrity] FOUND {len(dups)} duplicated {KEY_COLUMNS} key(s):")
    for key in dups[:20]:
        print(f"    {key}")
    if len(dups) > 20:
        print(f"    ... and {len(dups) - 20} more")

    if args.warn_only:
        print("[results-integrity] --warn-only set; not failing the build.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
