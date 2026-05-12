#!/usr/bin/env python3
# 12h_Donchian20_Breakout_Volume_Trend
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter. 
# Long when price breaks above 20-period high with volume > 1.5x 20-period average and price > 1d EMA50. 
# Short when price breaks below 20-period low with volume > 1.5x 20-period average and price < 1d EMA50. 
# Exit when price crosses back below/above the 10-period moving average in the opposite direction. 
# Targets 15-30 trades/year to minimize fee decay and work in both bull/bear markets via trend filter.

name = "12h_Donchian20_Breakout_Volume_Trend"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Exit condition: 10-period SMA
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(sma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above 20-period high + volume surge + price > 1d EMA50
            if (close[i] > high_max_20[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below 20-period low + volume surge + price < 1d EMA50
            elif (close[i] < low_min_20[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 10-period SMA
            if close[i] < sma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 10-period SMA
            if close[i] > sma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals