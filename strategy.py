#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_With_Trend_Filter
Hypothesis: Trade weekly pivot R1/S1 breakouts on daily timeframe with volume confirmation, filtered by 1-week trend direction (EMA20).
Long when price breaks above R1 with volume spike and 1w uptrend; short when breaks below S1 with volume spike and 1w downtrend.
Targets 10-25 trades/year per symbol (40-100 total over 4 years) with position size 0.25.
Works in bull/bear: 1w trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""

name = "1d_WeeklyPivot_R1S1_Breakout_With_Trend_Filter"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
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
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate weekly pivot levels (using previous week's data)
    high_shift = np.roll(high_1w, 1)
    low_shift = np.roll(low_1w, 1)
    close_shift = np.roll(close_1w, 1)
    high_shift[0] = high_1w[0]
    low_shift[0] = low_1w[0]
    close_shift[0] = close_1w[0]
    
    # Previous week's range
    range_prev = high_shift - low_shift
    
    # Weekly pivot levels (standard formula)
    pivot = (high_shift + low_shift + close_shift) / 3.0
    R1 = pivot + (high_shift - low_shift) * 1.1 / 12  # R1 = pivot + (high-low)*1.1/12
    S1 = pivot - (high_shift - low_shift) * 1.1 / 12  # S1 = pivot - (high-low)*1.1/12
    
    # Align pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 1w uptrend (price > EMA20)
            if close[i] > R1_aligned[i] and volume_spike[i] and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND 1w downtrend (price < EMA20)
            elif close[i] < S1_aligned[i] and volume_spike[i] and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 1w trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 1w trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals