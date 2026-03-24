#!/usr/bin/env python3
"""
ZigZag Elliott Wave Strategy (Demo) - Pine conversion (partial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

name = "ZigZag Elliott Wave Strategy (Demo)"
timeframe = "4h"
leverage = 1


def _pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or left < 1 or right < 1:
        return out
    for i in range(left + right, n):
        c = i - right
        window = high[c - left : c + right + 1]
        center = high[c]
        if np.isfinite(center) and np.isfinite(window).all() and center == np.max(window):
            out[i] = center
    return out


def _pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or left < 1 or right < 1:
        return out
    for i in range(left + right, n):
        c = i - right
        window = low[c - left : c + right + 1]
        center = low[c]
        if np.isfinite(center) and np.isfinite(window).all() and center == np.min(window):
            out[i] = center
    return out


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    n = len(prices)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    high = prices["high"].to_numpy(dtype=np.float64)
    low = prices["low"].to_numpy(dtype=np.float64)
    close = prices["close"].to_numpy(dtype=np.float64)

    zigzag_len = 12
    fib_retr_min = 0.50
    fib_retr_max = 0.618
    fib_wave3_ext = 1.618
    fib_wave4_min = 0.382
    fib_wave4_max = 0.50
    partial_tp = 0.90

    ph = _pivot_high(high, zigzag_len, zigzag_len)
    pl = _pivot_low(low, zigzag_len, zigzag_len)

    signals = np.zeros(n, dtype=np.float64)
    position = 0.0

    last_low = np.nan
    last_high = np.nan
    wave1_low = np.nan
    wave1_high = np.nan
    partial_done = False

    for i in range(n):
        if np.isfinite(pl[i]):
            last_low = pl[i]
        if np.isfinite(ph[i]):
            last_high = ph[i]

        wave1_valid = np.isfinite(last_low) and np.isfinite(last_high) and last_high > last_low
        if wave1_valid:
            wave1_low = last_low
            wave1_high = last_high

        if not (np.isfinite(wave1_low) and np.isfinite(wave1_high) and wave1_high > wave1_low):
            signals[i] = position
            continue

        wave1_range = wave1_high - wave1_low
        fib50 = wave1_high - wave1_range * fib_retr_min
        fib618 = wave1_high - wave1_range * fib_retr_max
        wave2_zone = close[i] <= fib50 and close[i] >= fib618

        wave3_target = wave1_low + wave1_range * fib_wave3_ext
        wave4_fib_min = wave3_target - wave1_range * fib_wave4_min
        wave4_fib_max = wave3_target - wave1_range * fib_wave4_max
        wave4_zone = close[i] <= wave4_fib_min and close[i] >= wave4_fib_max
        partial_close_price = wave3_target * partial_tp

        # Wave 2 entry.
        if wave2_zone and position == 0.0:
            position = 1.0
            partial_done = False

        # Wave 4 add-on approximated as increased target position.
        if wave4_zone and position > 0.0:
            position = min(1.5, position + 0.5)

        # Partial take profit approximation.
        if close[i] >= partial_close_price and position > 0.0 and not partial_done:
            position = max(0.5, position * 0.5)
            partial_done = True

        # Full exit at wave 3 extension.
        if close[i] >= wave3_target and position > 0.0:
            position = 0.0
            partial_done = False

        signals[i] = position

    return signals
