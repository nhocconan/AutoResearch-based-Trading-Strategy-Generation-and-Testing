#!/usr/bin/env python3
"""Validator tests: the compliance gate is the first line of defence against
look-ahead bias and unsafe code. Each rule gets an explicit pass/fail case."""

from __future__ import annotations

from validator import validate_strategy

VALID = '''
name = "unit_test_strategy"
timeframe = "1h"
leverage = 1.0

import numpy as np

def generate_signals(prices):
    close = prices["close"].values
    sma = __import__("pandas").Series(close).rolling(20, min_periods=20).mean().values
    sig = np.zeros(len(close))
    sig[close > sma] = 0.25
    return sig
'''


def test_valid_strategy_passes():
    r = validate_strategy(VALID)
    assert r.valid, r.errors


def test_negative_shift_rejected():
    code = VALID + "\n# future = prices['close'].shift(-1)\n"
    code = code.replace("# future", "future")
    r = validate_strategy(code)
    assert not r.valid
    assert any("shift" in e.lower() for e in r.errors)


def test_future_index_rejected():
    bad = VALID.replace("sig[close > sma] = 0.25", "x = prices.iloc[i+1]\n    sig[close > sma] = 0.25")
    r = validate_strategy(bad)
    assert not r.valid


def test_missing_name_rejected():
    bad = VALID.replace('name = "unit_test_strategy"', "")
    r = validate_strategy(bad)
    assert not r.valid
    assert any("name" in e.lower() for e in r.errors)


def test_invalid_timeframe_rejected():
    bad = VALID.replace('timeframe = "1h"', 'timeframe = "1m"')
    r = validate_strategy(bad)
    assert not r.valid
    assert any("timeframe" in e.lower() for e in r.errors)


def test_excessive_leverage_rejected():
    bad = VALID.replace("leverage = 1.0", "leverage = 10.0")
    r = validate_strategy(bad)
    assert not r.valid
    assert any("leverage" in e.lower() for e in r.errors)


def test_synthetic_resample_rejected():
    bad = VALID.replace(
        "    sig = np.zeros(len(close))",
        "    htf = prices.resample('4h').last()\n    sig = np.zeros(len(close))",
    )
    r = validate_strategy(bad)
    assert not r.valid


def test_manual_mtf_index_rejected():
    bad = VALID.replace(
        "    sig = np.zeros(len(close))",
        "    idx_4h = i // bars_per_4h\n    sig = np.zeros(len(close))",
    )
    r = validate_strategy(bad)
    assert not r.valid


def test_datetime_floordiv_rejected():
    bad = VALID.replace(
        "    sig = np.zeros(len(close))",
        "    hour = prices['open_time'] // 3600000\n    sig = np.zeros(len(close))",
    )
    r = validate_strategy(bad)
    assert not r.valid


def test_fractal_alignment_without_delay_rejected():
    bad = '''
name = "fractal_test"
timeframe = "1h"
leverage = 1.0
import numpy as np
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractal_levels
def generate_signals(prices):
    df = get_htf_data(prices, "4h")
    bear_fractal, bull_fractal = compute_williams_fractal_levels(df["high"].values, df["low"].values)
    aligned = align_htf_to_ltf(prices, df, bull_fractal)
    return np.zeros(len(prices))
'''
    r = validate_strategy(bad)
    assert not r.valid
    assert any("fractal" in e.lower() for e in r.errors)


def test_fractal_alignment_with_delay_passes():
    ok = '''
name = "fractal_test"
timeframe = "1h"
leverage = 1.0
import numpy as np
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractal_levels
def generate_signals(prices):
    df = get_htf_data(prices, "4h")
    bear_fractal, bull_fractal = compute_williams_fractal_levels(df["high"].values, df["low"].values)
    aligned = align_htf_to_ltf(prices, df, bull_fractal, additional_delay_bars=2)
    return np.zeros(len(prices))
'''
    r = validate_strategy(ok)
    assert r.valid, r.errors


def test_current_strategy_file_is_compliant():
    """The shipped strategy.py must always pass its own validator."""
    from validator import validate_file
    r = validate_file("strategy.py")
    assert r.valid, r.errors
