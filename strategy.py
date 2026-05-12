#!/usr/bin/env python3
"""
12h_DonchianBreakout_1dTrend_Volume
Hypothesis: On 12h timeframe, Donchian(20) breakout with 1d EMA50 trend and volume > 2x 20-period average
provides high-probability entries. Works in bull via breakout momentum and bear via mean-reversion
at extremes with trend filter. Targets 15-35 trades/year (60-140 total over 4 years).
"""

name = "12h_DonchianBreakout_1dTrend_Volume"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get values for current 12h bar
        ema50 = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(vol_avg_val) or 
            np.isnan(dch_high) or np.isnan(dch_low)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + price above EMA50 + volume surge
            if (close[i] > dch_high and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price below EMA50 + volume surge
            elif (close[i] < dch_low and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or price below EMA50
            if (close[i] < dch_low or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or price above EMA50
            if (close[i] > dch_high or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals