#!/usr/bin/env python3
"""
12h_PriceChannel_Breakout_Volume_1dTrend
Hypothesis: On 12h timeframe, breakouts from price channels (high/low of last 20 periods)
with volume > 1.5x 20-period average and daily EMA50 trend filter capture strong momentum.
This works in bull markets by riding breakouts and in bear markets by filtering out
false breakouts against the trend. Targets 12-37 trades/year (50-150 total over 4 years)
with low turnover to minimize fee drag.
"""

name = "12h_PriceChannel_Breakout_Volume_1dTrend"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Price channel: high/low of last 20 periods
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        ema50 = ema50_1d_aligned[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(high_max) or 
            np.isnan(low_min) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above channel high + above EMA50 + volume surge
            if (close[i] > high_max and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below channel low + below EMA50 + volume surge
            elif (close[i] < low_min and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below channel low or below EMA50
            if (close[i] < low_min or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above channel high or above EMA50
            if (close[i] > high_max or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals