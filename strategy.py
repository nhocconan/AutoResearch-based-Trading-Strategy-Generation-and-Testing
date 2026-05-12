#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_Volume_Trend
Hypothesis: On daily timeframe, Donchian(20) breakouts with volume confirmation and 
weekly EMA200 trend filter capture strong trends while avoiding false breakouts in ranges.
Works in bull via upward breakouts and bear via downward breakouts with trend filter.
Targets 7-25 trades/year (30-100 total over 4 years) with low turnover.
"""

name = "1d_Donchian20_Breakout_Volume_Trend"
timeframe = "1d"
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

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Calculate daily Donchian channels (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-day average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get current values
        ema200 = ema200_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema200) or np.isnan(upper) or 
            np.isnan(lower) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + above weekly EMA200 + volume surge
            if (close[i] > upper and 
                close[i] > ema200 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + below weekly EMA200 + volume surge
            elif (close[i] < lower and 
                  close[i] < ema200 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or below weekly EMA200
            if (close[i] < lower or close[i] < ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or above weekly EMA200
            if (close[i] > upper or close[i] > ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals