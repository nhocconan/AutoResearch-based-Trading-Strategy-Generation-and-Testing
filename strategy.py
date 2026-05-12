#!/usr/bin/env python3
# 6h_Supertrend_Filter_1dTrend_VolumeBreakout
# Hypothesis: 6h Supertrend (10,3) filters trend direction, combined with 1d EMA20 for higher timeframe trend alignment, and volume spike breakouts for entry. Exits when Supertrend flips or volume drops. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drift. Works in bull/bear by requiring alignment between 6s momentum, 1d trend, and volume confirmation.

name = "6h_Supertrend_Filter_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 6h data for Supertrend and price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate ATR for Supertrend
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate Supertrend
    basic_ub = (high_6h + low_6h) / 2 + 3 * atr
    basic_lb = (high_6h + low_6h) / 2 - 3 * atr
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    supertrend = np.zeros_like(close_6h)
    direction = np.ones_like(close_6h)  # 1 for uptrend, -1 for downtrend

    for i in range(1, len(close_6h)):
        final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_6h[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_6h[i-1] < final_lb[i-1]) else final_lb[i-1]

        if direction[i-1] == 1:
            supertrend[i] = final_ub[i] if close_6h[i] <= final_ub[i] else final_lb[i]
            direction[i] = -1 if close_6h[i] <= final_ub[i] else 1
        else:
            supertrend[i] = final_lb[i] if close_6h[i] >= final_lb[i] else final_ub[i]
            direction[i] = 1 if close_6h[i] >= final_lb[i] else -1

    # Align Supertrend and direction to LTF
    supertrend_aligned = align_htf_to_ltf(prices, df_6h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_6h, direction)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Supertrend uptrend, price above 1d EMA, and volume spike
            if direction_aligned[i] == 1 and close[i] > ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend, price below 1d EMA, and volume spike
            elif direction_aligned[i] == -1 and close[i] < ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend flips to downtrend or volume drops below average
            if direction_aligned[i] == -1 or volume[i] < volume_sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend flips to uptrend or volume drops below average
            if direction_aligned[i] == 1 or volume[i] < volume_sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals