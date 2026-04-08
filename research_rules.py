#!/usr/bin/env python3
"""
Single source of truth for pass/fail rules used across research pipelines.
"""

from __future__ import annotations

TRAIN_MIN_SHARPE = 0.0
TRAIN_MIN_TRADES = 5
TEST_MIN_SHARPE = 0.0
TEST_MIN_TRADES = 3
MAX_ALLOWED_DRAWDOWN_PCT = -50.0


def _metric(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0.0) or 0.0)


def train_symbol_pass(metrics: dict) -> bool:
    return (
        _metric(metrics, "sharpe_ratio") > TRAIN_MIN_SHARPE
        and int(metrics.get("num_trades", 0) or 0) >= TRAIN_MIN_TRADES
        and _metric(metrics, "max_drawdown_pct") > MAX_ALLOWED_DRAWDOWN_PCT
    )


def test_symbol_pass(metrics: dict) -> bool:
    return (
        _metric(metrics, "sharpe_ratio") > TEST_MIN_SHARPE
        and int(metrics.get("num_trades", 0) or 0) >= TEST_MIN_TRADES
        and _metric(metrics, "max_drawdown_pct") > MAX_ALLOWED_DRAWDOWN_PCT
    )


def row_is_active(status: str) -> bool:
    return str(status) == "keep"


def rules_summary() -> dict:
    return {
        "train_min_sharpe": TRAIN_MIN_SHARPE,
        "train_min_trades": TRAIN_MIN_TRADES,
        "test_min_sharpe": TEST_MIN_SHARPE,
        "test_min_trades": TEST_MIN_TRADES,
        "max_allowed_drawdown_pct": MAX_ALLOWED_DRAWDOWN_PCT,
    }
