#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_VolumeTrend
Hypothesis: Breakouts from 4h price channels (Donchian or Keltner) with volume confirmation and higher timeframe trend filter work in both bull and bear markets. Uses 4h Donchian(20) breakout + volume > 1.5x 20-period average + 1d EMA50 trend filter. Designed for low trade frequency (~25-40/year) to minimize fee drag while capturing strong directional moves.
"""

name = "4h_PriceChannel_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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

    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + uptrend + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low + downtrend + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR trend turns down
            if close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR trend turns up
            if close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals