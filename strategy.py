#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with 12h Trend Filter
# - Uses 12h EMA50 as trend filter (long when price > EMA50, short when price < EMA50)
# - Entries at Camarilla R3/S3 reversals from 1d pivots (mean reversion in range)
# - Exits at R4/S4 breakouts (trend continuation) or opposite R3/S3 touch
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Works in both bull/bear via trend filter + reversal logic at key levels
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # R1, R2, R3, R4 and S1, S2, S3, S4
    range_hl = high_1d - low_1d
    r1 = close_1d + range_hl * 1.1 / 12
    r2 = close_1d + range_hl * 1.1 / 6
    r3 = close_1d + range_hl * 1.1 / 4
    r4 = close_1d + range_hl * 1.1 / 2
    s1 = close_1d - range_hl * 1.1 / 12
    s2 = close_1d - range_hl * 1.1 / 6
    s3 = close_1d - range_hl * 1.1 / 4
    s4 = close_1d - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h price data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or \
           np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        uptrend = close_6h[i] > ema_50_12h_aligned[i]
        downtrend = close_6h[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: price touches S3 in uptrend (bounce from support)
            if uptrend and low_6h[i] <= s3_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches R3 in downtrend (rejection at resistance)
            elif downtrend and high_6h[i] >= r3_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: stop loss at S4 break, take profit at R3 touch, or trend reversal
            if low_6h[i] <= s4_6h[i] or high_6h[i] >= r3_6h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss at R4 break, take profit at S3 touch, or trend reversal
            if high_6h[i] >= r4_6h[i] or low_6h[i] <= s3_6h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Reversal_12hEMA50Filter"
timeframe = "6h"
leverage = 1.0