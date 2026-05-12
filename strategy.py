#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_1wTrend_Filter
Hypothesis: Weekly Donchian(20) breakouts capture long-term momentum, with trend filter from weekly EMA50 and volume confirmation to reduce false signals. Works in bull (breakouts continue) and bear (breakouts fail quickly, limiting losses). Targets 15-25 trades/year.
"""

name = "1d_WeeklyDonchian_Breakout_1wTrend_Filter"
timeframe = "1d"
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

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values

    # Weekly Donchian(20) channels
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: 1.5x 20-period average (daily)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current daily bar
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(donch_high) or np.isnan(donch_low) or 
            np.isnan(ema50) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high + price above weekly EMA50 + volume surge
            if (close[i] > donch_high and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + price below weekly EMA50 + volume surge
            elif (close[i] < donch_low and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low or price below weekly EMA50
            if (close[i] < donch_low or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high or price above weekly EMA50
            if (close[i] > donch_high or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals