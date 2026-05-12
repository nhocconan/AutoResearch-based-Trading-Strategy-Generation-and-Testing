#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 12h with 1d EMA50 trend filter and volume spike (>1.5x 20-period avg) provides high-probability directional moves in both bull and bear markets. 
Long when price breaks above upper Donchian + price > 1d EMA50 + volume spike. 
Short when price breaks below lower Donchian + price < 1d EMA50 + volume spike.
Exit when price reverses to touch the opposite Donchian band or trend changes.
Designed for very low trade frequency (<20/year) to minimize fee decay while capturing strong trends in BTC/ETH.
"""

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Donchian channels (20-period) on 12h
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + 1d uptrend + volume spike
            if close[i] > high_max_20[i-1] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below lower Donchian + 1d downtrend + volume spike
            elif close[i] < low_min_20[i-1] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or goes below lower Donchian OR trend turns down
            if close[i] <= low_min_20[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price touches or goes above upper Donchian OR trend turns up
            if close[i] >= high_max_20[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals