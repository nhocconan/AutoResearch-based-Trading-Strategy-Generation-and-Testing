#!/usr/bin/env python3
"""
1d_WP_Trend_With_Volume_Spike
Hypothesis: Use weekly pivot levels from 1w for breakout entries in the direction of 1d EMA trend, filtered by volume spike.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year). 
Weekly pivots provide strong institutional levels; EMA trend filters direction; volume spike confirms breakout strength.
Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).
"""

name = "1d_WP_Trend_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and daily data for trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Standard pivot point formula: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    pivot = (high_prev + low_prev + close_prev) / 3
    r1 = (2 * pivot) - high_prev
    s1 = (2 * pivot) - low_prev
    
    # Align weekly pivot levels to daily timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get daily price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1 week), daily EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above weekly R1 with volume spike
            if close[i] > ema_34_aligned[i] and high[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below weekly S1 with volume spike
            elif close[i] < ema_34_aligned[i] and low[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR trend turns bearish
            if low[i] < pivot_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR trend turns bullish
            if high[i] > pivot_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals