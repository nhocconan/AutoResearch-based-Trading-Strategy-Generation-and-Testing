#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend
Hypothesis: Weekly pivot points on 1w timeframe act as strong support/resistance.
Long when price breaks above R1 with volume confirmation and weekly EMA10 uptrend.
Short when price breaks below S1 with volume confirmation and weekly EMA10 downtrend.
Uses volume spike (>2x average) to filter low-quality breakouts.
Targets 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

name = "1d_WeeklyPivot_Breakout_Trend"
timeframe = "1d"
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
    
    # Get 1w data for weekly pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous 1w bar
    # Using previous week's HLC to avoid look-ahead
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate pivot levels
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)  # R1 = C + (H-L) * 1.1/12
    s1 = prev_close - (rng * 1.1 / 12)  # S1 = C - (H-L) * 1.1/12
    r2 = prev_close + (rng * 1.1 / 6)   # R2 = C + (H-L) * 1.1/6
    s2 = prev_close - (rng * 1.1 / 6)   # S2 = C - (H-L) * 1.1/6
    
    # Align 1w levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume confirmation (20-period MA on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA10 (10) and volume MA (20)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_1w = close[i] > ema_10_1w_aligned[i]
        downtrend_1w = close[i] < ema_10_1w_aligned[i]
        
        # Volume confirmation (>2x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above R1 + volume confirmation
            if uptrend_1w and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S1 + volume confirmation
            elif downtrend_1w and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend_1w or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend_1w or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals