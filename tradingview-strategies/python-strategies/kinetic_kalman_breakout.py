#!/usr/bin/env python3
"""
TradingView conversion: Kinetic Kalman Breakout
Compatibility: direct
Source Pine: nd8EpyQ5-Kinetic-Kalman-Breakout.pine
"""

import numpy as np
import pandas as pd

name = "tv_kinetic_kalman_breakout"
timeframe = "15m"
leverage = 1.0

PROCESS_NOISE_POS = 0.05
PROCESS_NOISE_VEL = 0.0001
MEASUREMENT_NOISE = 250.0
BAND_LOOKBACK = 200
BAND_MULTIPLIER = 2.6


def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    valid = np.isfinite(a) & np.isfinite(b)
    prev_valid = valid[:-1] & valid[1:]
    out[1:] = prev_valid & (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    valid = np.isfinite(a) & np.isfinite(b)
    prev_valid = valid[:-1] & valid[1:]
    out[1:] = prev_valid & (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].to_numpy(dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    kalman_price = np.full(n, np.nan, dtype=np.float64)

    x_p = close[0]
    x_v = 0.0
    p00 = 1.0
    p01 = 0.0
    p10 = 0.0
    p11 = 1.0

    for i in range(n):
        z = close[i]

        p_prime = x_p + x_v
        v_prime = x_v

        a00 = p00 + p10
        a01 = p01 + p11
        a10 = p10
        a11 = p11

        p00_ = a00 + a01
        p01_ = a01
        p10_ = a10 + a11
        p11_ = a11

        p00_ += PROCESS_NOISE_POS
        p11_ += PROCESS_NOISE_VEL

        y = z - p_prime
        s = p00_ + MEASUREMENT_NOISE
        k0 = p00_ / s
        k1 = p10_ / s

        x_p = p_prime + k0 * y
        x_v = v_prime + k1 * y

        i00 = 1.0 - k0
        i10 = -k1

        pp00 = i00 * p00_
        pp01 = i00 * p01_
        pp10 = i10 * p00_ + p10_
        pp11 = i10 * p01_ + p11_

        p00 = pp00
        p01 = pp01
        p10 = pp10
        p11 = pp11
        kalman_price[i] = x_p

    abs_diff = np.abs(close - kalman_price)
    mae = (
        pd.Series(abs_diff)
        .rolling(window=BAND_LOOKBACK, min_periods=BAND_LOOKBACK)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    upper_band = kalman_price + BAND_MULTIPLIER * mae
    lower_band = kalman_price - BAND_MULTIPLIER * mae

    bull_signal = _crossover(close, upper_band)
    bear_signal = _crossunder(close, lower_band)

    signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    for i in range(n):
        if bull_signal[i]:
            position = 1.0
        elif bear_signal[i]:
            position = -1.0
        signals[i] = position

    return signals
