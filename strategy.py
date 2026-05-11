#!/usr/bin/env python3
name = "6h_WeeklyPivotBias_TriangleBreakout_12hVolFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels from 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (PP) and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.0)
    s1_1w = pivot_1w - (range_1w * 1.0)
    r2_1w = pivot_1w + (range_1w * 2.0)
    s2_1w = pivot_1w - (range_1w * 2.0)
    r3_1w = pivot_1w + (range_1w * 3.0)
    s3_1w = pivot_1w - (range_1w * 3.0)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 12h volume filter: current volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # Triangle pattern detection: higher lows and lower highs
    # Higher lows: current low > previous low
    # Lower highs: current high < previous high
    higher_lows = low > np.roll(low, 1)
    lower_highs = high < np.roll(high, 1)
    # Valid triangle: both conditions true for last 3 periods
    triangle_condition = (
        higher_lows & lower_highs &
        np.roll(higher_lows, 1) & np.roll(lower_highs, 1) &
        np.roll(higher_lows, 2) & np.roll(lower_highs, 2)
    )
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 3)  # volume MA20 and triangle needs 3 bars
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or \
           np.isnan(s3_1w_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Triangle breakout above R2 with volume, bias above weekly pivot
            if (triangle_condition[i] and 
                close[i] > r2_1w_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Triangle breakout below S2 with volume, bias below weekly pivot
            elif (triangle_condition[i] and 
                  close[i] < s2_1w_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or triangle breaks down
            if close[i] < s1_1w_aligned[i] or not triangle_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or triangle breaks down
            if close[i] > r1_1w_aligned[i] or not triangle_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals