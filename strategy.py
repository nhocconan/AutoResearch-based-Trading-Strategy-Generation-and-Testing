#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend
# Hypothesis: Enter long when price breaks above 4h Donchian upper channel (20-period high) during high volume and bullish 1d trend (price above EMA50).
# Enter short when price breaks below 4h Donchian lower channel (20-period low) during high volume and bearish 1d trend (price below EMA50).
# Donchian breakouts capture volatility expansion; volume confirms institutional participation; 1d EMA50 filters for higher timeframe trend alignment.
# Works in bull (breakouts above upper channel in uptrend) and bear (breakdowns below lower channel in downtrend).
# Low frequency due to combined breakout+volume+trend conditions.

name = "4h_Donchian_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Donchian + volume filter + bullish 1d trend
            if close[i] > high_20[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian + volume filter + bearish 1d trend
            elif close[i] < low_20[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below 10-period low or trend reversal
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if close[i] < low_10[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above 10-period high or trend reversal
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if close[i] > high_10[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals