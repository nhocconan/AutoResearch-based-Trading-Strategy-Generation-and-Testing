#!/usr/bin/env python3
name = "6H_Weekly_Pivot_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Key levels: R1, S1 for breakout, R2/S2 for confirmation
    r1_1w = pivot_1w + (range_1w * 1.0)
    s1_1w = pivot_1w - (range_1w * 1.0)
    r2_1w = pivot_1w + (range_1w * 2.0)
    s2_1w = pivot_1w - (range_1w * 2.0)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Weekly trend filter: price relative to pivot
    trend_up = close_1w > pivot_1w
    trend_down = close_1w < pivot_1w
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + weekly uptrend + volume confirmation
            if close[i] > r1_aligned[i] and trend_up_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + weekly downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and trend_down_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or weekly trend turns down
            if close[i] < s1_aligned[i] or trend_down_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or weekly trend turns up
            if close[i] > r1_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals