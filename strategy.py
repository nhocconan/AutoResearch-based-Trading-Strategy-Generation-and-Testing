# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_VolumeBreakout_12hTrend
Hypothesis: On 4h timeframe, buy when price breaks above 20-period high with volume >2x average and 12h EMA50 trending up; sell when price breaks below 20-period low with volume >2x average and 12h EMA50 trending down. Uses volume confirmation and trend filter to avoid false breakouts, targeting low trade frequency (<40/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

name = "4h_VolumeBreakout_12hTrend"
timeframe = "4h"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 20-period high/low for breakout levels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 20-period high + 12h uptrend + volume spike
            if (close[i] > high_max_20[i-1] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below 20-period low + 12h downtrend + volume spike
            elif (close[i] < low_min_20[i-1] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low OR trend turns down
            if close[i] < low_min_20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high OR trend turns up
            if close[i] > high_max_20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals