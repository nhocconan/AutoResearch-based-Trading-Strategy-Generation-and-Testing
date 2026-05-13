#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1wTrend_Volume
# Hypothesis: Donchian(20) breakout on 12h with 1-week trend filter (price > SMA50 for long, < SMA50 for short) and volume confirmation (>1.5x average). 
# Works in bull via breakout above resistance and in bear via breakout below support. 
# Uses weekly trend filter to avoid counter-trend trades. Target: 15-30 trades/year.

name = "12h_Donchian20_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Weekly SMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    sma50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + price above weekly SMA50 + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > sma50_1w_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price below weekly SMA50 + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < sma50_1w_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below midpoint) or weekly trend turns bearish
            mid = (high_max_20[i] + low_min_20[i]) / 2.0
            if close[i] < mid or close[i] < sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above midpoint) or weekly trend turns bullish
            mid = (high_max_20[i] + low_min_20[i]) / 2.0
            if close[i] > mid or close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals