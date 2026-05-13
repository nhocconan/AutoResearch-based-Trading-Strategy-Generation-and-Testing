#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Use 20-period Donchian channel breakouts with volume confirmation and 12h EMA50 trend filter.
In trending markets, breakouts capture momentum; in ranging markets, volume filter reduces false signals.
12h trend filter aligns with higher timeframe momentum to avoid counter-trend trades.
Target: 20-40 trades/year per symbol (~80-160 total over 4 years).
Works in both bull and bear markets by only taking trades in direction of higher timeframe trend.
"""

name = "4h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above upper band, 12h uptrend, volume confirmation
            if close[i] > high_max[i] and uptrend_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower band, 12h downtrend, volume confirmation
            elif close[i] < low_min[i] and downtrend_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below midpoint or trend reverses
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] < midpoint or not uptrend_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above midpoint or trend reverses
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] > midpoint or not downtrend_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals