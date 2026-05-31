#!/usr/bin/env python3
"""Multi-timeframe alignment tests — the subtlest source of look-ahead bias.
A higher-timeframe value may only become visible to a lower-timeframe bar
*after the HTF candle has closed*. These tests prove that boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtf_data import align_htf_to_ltf, compute_williams_fractals


def _ltf(n=24, start="2021-01-01", freq="1h"):
    return pd.DataFrame({"open_time": pd.date_range(start, periods=n, freq=freq)})


def _htf(n=6, start="2021-01-01", freq="4h"):
    return pd.DataFrame({"open_time": pd.date_range(start, periods=n, freq=freq)})


def test_completed_bar_only_no_lookahead():
    """The 4h candle opening at 04:00 closes at 08:00. Its value must NOT
    appear on any 1h bar before 08:00."""
    ltf = _ltf(n=24)            # 1h bars 00:00..23:00
    htf = _htf(n=6)             # 4h bars 00:00,04:00,08:00,...
    htf_values = np.arange(len(htf), dtype=np.float64)  # value == bar index

    aligned = align_htf_to_ltf(ltf, htf, htf_values, use_completed_only=True)

    # 1h bar at 04:00 is index 4. The 04:00 4h candle (value 1) has NOT closed
    # yet, so the visible completed value is the 00:00 candle (value 0).
    assert aligned[4] == 0.0
    # By 08:00 (1h index 8) the 04:00 candle has closed → value 1 visible.
    assert aligned[8] == 1.0
    # Before the first candle closes (00:00..03:00) nothing is available.
    assert np.isnan(aligned[0])


def test_completed_value_is_never_from_the_future():
    ltf = _ltf(n=24)
    htf = _htf(n=6)
    htf_values = np.arange(len(htf), dtype=np.float64)
    aligned = align_htf_to_ltf(ltf, htf, htf_values, use_completed_only=True)

    htf_open = pd.to_datetime(htf["open_time"]).values
    ltf_open = pd.to_datetime(ltf["open_time"]).values
    duration = np.timedelta64(4, "h")
    for i, t in enumerate(ltf_open):
        if np.isnan(aligned[i]):
            continue
        # The HTF bar whose value we used must have CLOSED at or before t.
        used_idx = int(aligned[i])
        assert htf_open[used_idx] + duration <= t


def test_additional_delay_pushes_visibility_later():
    ltf = _ltf(n=24)
    htf = _htf(n=6)
    htf_values = np.arange(len(htf), dtype=np.float64)

    base = align_htf_to_ltf(ltf, htf, htf_values, use_completed_only=True)
    delayed = align_htf_to_ltf(ltf, htf, htf_values, use_completed_only=True,
                               additional_delay_bars=2)
    # With 2 extra 4h bars of delay, fewer/later values are visible.
    base_visible = np.sum(~np.isnan(base))
    delayed_visible = np.sum(~np.isnan(delayed))
    assert delayed_visible < base_visible


def test_length_mismatch_raises():
    ltf = _ltf(n=10)
    htf = _htf(n=5)
    try:
        align_htf_to_ltf(ltf, htf, np.arange(4, dtype=np.float64))
    except ValueError:
        return
    raise AssertionError("expected ValueError on htf value length mismatch")


def test_williams_fractal_marks_local_extreme():
    # A clear peak at index 4: 1,2,3,4,9,4,3,2,1
    high = np.array([1, 2, 3, 4, 9, 4, 3, 2, 1], dtype=np.float64)
    low = np.array([9, 8, 7, 6, 1, 6, 7, 8, 9], dtype=np.float64)
    bearish, bullish = compute_williams_fractals(high, low)
    assert bearish[4] == 1.0  # high peak
    assert bullish[4] == 1.0  # low trough
    # Edges (need 2 bars either side) are never fractals.
    assert bearish[0] == 0.0 and bearish[-1] == 0.0
