#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter
Hypothesis: Trade 12h price breakouts above/below 1-day Camarilla R1/S1 levels with volume confirmation and 1-week trend filter.
Long when price breaks above 1d R1 with volume spike and 1w uptrend; short when breaks below 1d S1 with volume spike and 1w downtrend.
Uses daily Camarilla pivot levels (calculated from prior day's high, low, close) and volume > 1.5x 20-period average for confirmation.
1-week trend filter avoids counter-trend trades. Designed for 12h timeframe to capture multi-day moves with reduced noise.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: 1w trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter"
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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (using prior day's high, low, close)
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    rang = high_1d - low_1d
    r1_1d = close_1d + rang * 1.1 / 12.0
    s1_1d = close_1d - rang * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 12h timeframe (already delayed by one bar via align_htf_to_ltf)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_1w = ema(close_1w, 20)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready (20 for EMA + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d R1 with volume filter AND 1w uptrend (close > EMA20)
            if close[i] > r1_1d_aligned[i] and volume_filter[i] and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S1 with volume filter AND 1w downtrend (close < EMA20)
            elif close[i] < s1_1d_aligned[i] and volume_filter[i] and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 1d S1 OR 1w trend turns down
            if close[i] < s1_1d_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 1d R1 OR 1w trend turns up
            if close[i] > r1_1d_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals