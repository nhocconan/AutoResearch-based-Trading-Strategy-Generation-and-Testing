#!/usr/bin/env python3
# 1h_4H_Donchian_Breakout_20_1dTrend_VolumeSpike_v2
# Hypothesis: Use 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
# Enter long when price breaks above 4h upper band with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below 4h lower band with volume spike and 1d EMA50 downtrend.
# Exit when price returns to 4h midline (average of upper/lower band).
# Uses 1h timeframe for entry timing, 4h for structure, 1d for trend filter.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "1h_4H_Donchian_Breakout_20_1dTrend_VolumeSpike_v2"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate 4h Donchian(20) channels
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i])
        lower_4h[i] = np.min(low_4h[i-20:i])
    mid_4h = (upper_4h + lower_4h) / 2.0

    # Align Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    mid_aligned = align_htf_to_ltf(prices, df_4h, mid_4h)

    # Volume confirmation: current volume > 2.0 x 24-period average (4h equivalent)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data is not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above 4h upper band with volume spike and 1d EMA50 uptrend
            if close[i] > upper_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: break below 4h lower band with volume spike and 1d EMA50 downtrend
            elif close[i] < lower_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to 4h midline
            if close[i] <= mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price returns to 4h midline
            if close[i] >= mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals