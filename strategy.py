#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume_1d_Trend_Filter
Hypothesis: Trade Camarilla pivot R1/S1 breakouts on 12h with volume confirmation and 1d trend filter.
Long when price breaks above R1 with volume spike and 1d uptrend; short when breaks below S1 with volume spike and 1d downtrend.
Uses volume > 2.0x 20-period average for strong breakout confirmation to reduce trade frequency.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, high volume threshold filters false breakouts.
"""

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_1d_Trend_Filter"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    def camarilla(high_val, low_val, close_val):
        range_val = high_val - low_val
        r1 = close_val + range_val * 1.1 / 12
        s1 = close_val - range_val * 1.1 / 12
        return r1, s1
    
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        r1[i], s1[i] = camarilla(high_1d[i], low_1d[i], close_1d[i])
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume spike (volume > 2.0x 20-period average for strict confirmation)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 1d uptrend (price > EMA50)
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND 1d downtrend (price < EMA50)
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 1d trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 1d trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals