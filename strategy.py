#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend
Hypothesis: Use daily Camarilla pivot R1 and S1 levels for breakout entries, filtered by weekly EMA100 trend direction. Go long when price breaks above R1 with volume confirmation and weekly trend up, short when price breaks below S1 with volume confirmation and weekly trend down. Camarilla levels provide high-probability reversal/breakout points, and weekly trend filter avoids counter-trend trades. Designed for 1d timeframe to limit trades (<20/year) and avoid fee drag. Works in both bull and bear markets by following the weekly trend.
"""

name = "1d_Camarilla_R1S1_Breakout_1wTrend"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to minute timeframe (no extra delay needed)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and weekly trend up
            if close[i-1] <= r1_aligned[i-1] and close[i] > r1_aligned[i] and vol_spike and close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and weekly trend down
            elif close[i-1] >= s1_aligned[i-1] and close[i] < s1_aligned[i] and vol_spike and close[i] < ema_100_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or weekly trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or weekly trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals