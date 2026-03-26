#!/usr/bin/env python3
"""
validator.py - Strategy Compliance Checker
==========================================
Validates trading strategy code for compliance with research protocol:

1. No look-ahead bias (negative shift, future index access)
2. Required fields present (name, timeframe, leverage, generate_signals)
3. Leverage within safe bounds
4. Timeframe is valid
5. Signal generation function signature is correct

Usage:
    from validator import validate_strategy, validate_file
    result = validate_file("strategy.py")
    print(result.summary())
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_FIELDS = ["name", "timeframe", "leverage", "generate_signals"]
VALID_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d", "1w"]  # 1m excluded: too noisy
MAX_LEVERAGE = 5.0

LOOK_AHEAD_PATTERNS = [
    (r"\.shift\s*\(\s*-\s*\d+", "Negative .shift(-n) detected — reads future bars (look-ahead bias)"),
    (r"prices\.iloc\s*\[(?:[^]]*?)i\s*\+", "Future index prices.iloc[i+...] detected (look-ahead bias)"),
    (r"prices\s*\[\s*i\s*\+", "Future index prices[i+...] detected (look-ahead bias)"),
    (r"\.rolling\s*\([^)]*min_periods\s*=\s*0", "rolling(min_periods=0) may include incomplete windows"),
    (r"df\s*\[\s*['\"].*['\"]\s*\]\s*=.*\.shift\s*\(\s*-", "Column assigned future-shifted value (look-ahead)"),
]

SYNTHETIC_RESAMPLE_PATTERNS = [
    (r"date_range\s*\(\s*(?:start\s*=\s*)?['\"]202", "SYNTHETIC: pd.date_range('202x...') — use mtf_data.get_htf_data() instead"),
    (r"\.resample\s*\(\s*['\"][14]", "Manual .resample() to HTF — use mtf_data.get_htf_data()"),
]

# Patterns that indicate manual positional MTF (i//N) without mtf_data
MANUAL_MTF_PATTERNS = [
    (r"//\s*bars_per_", "Manual MTF via i//bars_per_N — uses unclosed HTF bars (look-ahead). Use mtf_data.get_htf_data()"),
    (r"idx_\d+h\s*=\s*i\s*//", "Manual MTF index mapping — use mtf_data.align_htf_to_ltf()"),
    (r"n_\d+h\s*=.*//", "Manual HTF bar count — use mtf_data.get_htf_data()"),
]

# Patterns that cause frequent runtime TypeErrors (caught before wasting backtest time)
RUNTIME_ERROR_PATTERNS = [
    # Datetime floor-division (639 occurrences in production)
    (r"prices\.index\s*//",
     "TypeError: prices.index // N divides datetime64 — use integer loop variable or get_htf_data()"),
    (r"['\"]open_time['\"]\s*\]\s*//",
     "TypeError: open_time // N divides datetime64 — use integer index, not timestamp column"),
    (r"\.open_time\s*//",
     "TypeError: .open_time // N divides datetime64 — use integer index, not timestamp column"),
    # DatetimeIndex has no .dt accessor (unlike pd.Series) — 33 occurrences
    (r"\.index\.dt\.",
     "AttributeError: DatetimeIndex has no .dt — access .hour/.day/.month directly on the index (no .dt needed)"),
    # Deprecated pandas fillna(method=...) — 6 occurrences
    (r"\.fillna\s*\([^)]*method\s*=",
     "TypeError: fillna(method=) removed in pandas 2.x — use .ffill() or .bfill() instead"),
    # pd.to_datetime(open_time, unit='ms') crashes when open_time is already datetime64
    (r"to_datetime\s*\([^)]*unit\s*=\s*['\"]ms['\"]",
     "TypeError: pd.to_datetime(x, unit='ms') crashes when x is datetime64 — use prices.index.hour or pd.DatetimeIndex(prices['open_time']).hour instead"),
    # open_time[i] // int or timestamp_ms // — datetime64 scalar floor-divide (session filter)
    (r"(?:open_time|timestamp_ms|timestamp_s|ts_ms)\s*\[?\s*(?:i\s*\])?\s*//",
     "TypeError: datetime64 floor-divide — use prices.index.hour for session filters, not timestamp arithmetic"),
    # Manual hour extraction from ms: // (1000 * 60 * 60) or // 3600000
    (r"//\s*\(?\s*1000\s*\*\s*60\s*\*\s*60",
     "TypeError: // (1000*60*60) on datetime64 crashes — use prices.index.hour for UTC hour extraction"),
    (r"//\s*3600(?:000)?(?!\d)",
     "TypeError: // 3600(000) on datetime64 crashes — use prices.index.hour for UTC hour extraction"),
    # Single-divide on datetime index/column (18+9 occurrences)
    (r"prices\.index\s*/[^/=]",
     "TypeError: prices.index / N divides datetime64 — use integer loop variable instead"),
    (r"['\"]open_time['\"]\s*\]\s*/[^/=]",
     "TypeError: open_time / N divides datetime64 — use integer index instead"),
]


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }

    def summary(self) -> str:
        lines = []
        status = "PASS" if self.valid else "FAIL"
        lines.append(f"[{status}] Compliance Check")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        for i in self.info:
            lines.append(f"  INFO:  {i}")
        return "\n".join(lines)


def validate_strategy(code: str) -> ValidationResult:
    """Run all compliance checks on strategy source code."""
    result = ValidationResult(valid=True)

    # --- 1. Syntax check ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.errors.append(f"Syntax error: {e}")
        result.valid = False
        return result  # can't do AST checks without valid syntax

    # --- 2. Required fields ---
    for req in REQUIRED_FIELDS:
        if req not in code:
            result.errors.append(f"Missing required element: '{req}'")
            result.valid = False

    # --- 3. Look-ahead bias (regex) ---
    for pattern, msg in LOOK_AHEAD_PATTERNS:
        if re.search(pattern, code):
            result.errors.append(msg)
            result.valid = False

    # --- 3b. Synthetic resampling (causes alignment bugs) ---
    for pattern, msg in SYNTHETIC_RESAMPLE_PATTERNS:
        if re.search(pattern, code):
            result.errors.append(msg)
            result.valid = False

    # --- 3c. Manual positional MTF (i//N) without mtf_data ---
    if "get_htf_data" not in code and "mtf_data" not in code:
        for pattern, msg in MANUAL_MTF_PATTERNS:
            if re.search(pattern, code):
                result.errors.append(msg)
                result.valid = False

    # --- 3d. Runtime error patterns (saves backtest time on known-bad patterns) ---
    for pattern, msg in RUNTIME_ERROR_PATTERNS:
        if re.search(pattern, code):
            result.errors.append(msg)
            result.valid = False

    # --- 3e. Undefined bare variables that cause NameError at runtime ---
    _check_undefined_bare_names(code, result)

    # --- 4. AST: extract metadata values ---
    leverage_found = None
    timeframe_found = None
    name_found = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            if target.id == "leverage":
                val = _extract_constant(node.value)
                if val is not None:
                    leverage_found = float(val)
                    if leverage_found > MAX_LEVERAGE:
                        result.errors.append(
                            f"Leverage {leverage_found}x exceeds max allowed {MAX_LEVERAGE}x"
                        )
                        result.valid = False
                    elif leverage_found > 3.0:
                        result.warnings.append(
                            f"High leverage: {leverage_found}x (max is {MAX_LEVERAGE}x)"
                        )
                    else:
                        result.info.append(f"Leverage: {leverage_found}x")

            elif target.id == "timeframe":
                val = _extract_constant(node.value)
                if val is not None:
                    timeframe_found = str(val)
                    if timeframe_found not in VALID_TIMEFRAMES:
                        result.errors.append(
                            f"Invalid timeframe '{timeframe_found}' — must be one of {VALID_TIMEFRAMES}"
                        )
                        result.valid = False
                    else:
                        result.info.append(f"Timeframe: {timeframe_found}")

            elif target.id == "name":
                val = _extract_constant(node.value)
                if val is not None:
                    name_found = str(val)
                    result.info.append(f"Strategy name: {name_found}")

    # --- 5. Check generate_signals function signature ---
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
            args = [a.arg for a in node.args.args]
            if "prices" not in args:
                result.warnings.append(
                    "generate_signals() should accept 'prices' as argument"
                )
            result.info.append(
                f"generate_signals({', '.join(args)}) — function present"
            )
            break
    else:
        if "generate_signals" in code:
            pass  # might be defined but not as a def (edge case)

    # --- 6. T+1 fill and costs — enforced by engine ---
    result.info.append("Signal fill delay (t+1): enforced by backtest engine")
    result.info.append("Costs (fee 0.04% + slippage 0.01% + funding): enforced by backtest engine")

    if result.valid and not result.errors:
        result.info.append("✓ No compliance violations detected")

    return result


def _check_undefined_bare_names(code: str, result: ValidationResult):
    """AST check: catch common NameErrors before backtesting.

    Detects variables used as bare names inside generate_signals() that are
    never assigned there — the top recurring runtime errors are 'close' (159x)
    and 'j' (78x) in production logs.
    """
    BARE_NAMES = {"close", "high", "low", "open", "volume", "j", "period", "n"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # already caught above

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "generate_signals"):
            continue
        # Collect all names assigned within this function
        assigned = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for t in ast.walk(child):
                    if isinstance(t, ast.Name) and isinstance(child, ast.Assign) and t in ast.walk(child.targets[0] if child.targets else ast.parse("x")):
                        assigned.add(t.id)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(getattr(child, 'target', None), ast.Name):
                    assigned.add(child.target.id)
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    assigned.add(child.target.id)
            elif isinstance(child, ast.NamedExpr):
                if isinstance(child.target, ast.Name):
                    assigned.add(child.target.id)
        # Also include function arguments as "assigned"
        for arg in node.args.args:
            assigned.add(arg.arg)

        # Check if any bare name from BARE_NAMES is used but not assigned
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in BARE_NAMES and child.id not in assigned:
                if isinstance(child.ctx, ast.Load):
                    result.errors.append(
                        f"NameError: '{child.id}' used but never assigned in generate_signals() — "
                        f"use prices['{child.id}'].values or prices['{child.id}'] instead"
                    )
                    result.valid = False
                    BARE_NAMES.discard(child.id)  # report each name only once
        break


def validate_file(path: str) -> ValidationResult:
    """Validate a strategy file."""
    try:
        code = Path(path).read_text()
        return validate_strategy(code)
    except FileNotFoundError:
        r = ValidationResult(valid=False)
        r.errors.append(f"File not found: {path}")
        return r


def _extract_constant(node) -> object:
    """Extract a constant value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # Python <3.8 compat
        return node.n
    if isinstance(node, ast.Str):  # Python <3.8 compat
        return node.s
    return None


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "strategy.py"
    result = validate_file(path)
    print(result.summary())
    sys.exit(0 if result.valid else 1)
