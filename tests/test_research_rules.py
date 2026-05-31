#!/usr/bin/env python3
"""Pass/fail threshold tests — the single source of truth that decides which
strategies are kept. Mirrors the rules documented in CLAUDE.md."""

from __future__ import annotations

from research_rules import (
    MAX_ALLOWED_DRAWDOWN_PCT,
    row_is_active,
    rules_summary,
)
# Alias away the ``test_`` prefix so pytest does not collect the imported
# rule function itself as a test case.
from research_rules import test_symbol_pass as check_test_pass
from research_rules import train_symbol_pass


def _m(sharpe, trades, dd=-10.0):
    return {"sharpe_ratio": sharpe, "num_trades": trades, "max_drawdown_pct": dd}


def test_train_pass_requires_all_three():
    assert train_symbol_pass(_m(0.5, 5, -10.0))


def test_train_fail_on_low_sharpe():
    assert not train_symbol_pass(_m(0.0, 10, -10.0))      # sharpe must be > 0
    assert not train_symbol_pass(_m(-0.1, 10, -10.0))


def test_train_fail_on_too_few_trades():
    assert not train_symbol_pass(_m(1.0, 4, -10.0))       # needs >= 5


def test_train_fail_on_excessive_drawdown():
    assert not train_symbol_pass(_m(1.0, 10, MAX_ALLOWED_DRAWDOWN_PCT))   # not strictly greater
    assert not train_symbol_pass(_m(1.0, 10, -60.0))


def test_test_pass_threshold_is_looser_on_trades():
    assert check_test_pass(_m(0.1, 3, -10.0))            # test needs only >= 3
    assert not check_test_pass(_m(0.1, 2, -10.0))


def test_zero_trade_strategy_never_passes():
    assert not train_symbol_pass(_m(0.0, 0, 0.0))
    assert not check_test_pass(_m(0.0, 0, 0.0))


def test_row_is_active():
    assert row_is_active("keep")
    assert not row_is_active("discard")
    assert not row_is_active("")


def test_rules_summary_shape():
    s = rules_summary()
    for key in ("train_min_sharpe", "train_min_trades", "test_min_sharpe",
                "test_min_trades", "max_allowed_drawdown_pct"):
        assert key in s
