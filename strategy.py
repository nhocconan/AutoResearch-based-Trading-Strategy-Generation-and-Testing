#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend_1d
# Hypothesis: 4h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation.
# Works in bull markets (breakouts in uptrends) and bear markets (breakouts in downtrends).
# Uses 1d EMA50 for trend filter (institutional bias) and volume spike for conviction.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend_1d"
timeframe = "4h"
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

    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest[i] = np.max(high[i - lookback + 1:i + 1])
        lowest[i] = np.min(low[i - lookback + 1:i + 1])

    # Volume spike detection: volume > 1.5 * 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period MA
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + uptrend + volume spike
            if close[i] > highest[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + downtrend + volume spike
            elif close[i] < lowest[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (breakdown)
            if close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (breakout)
            if close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals