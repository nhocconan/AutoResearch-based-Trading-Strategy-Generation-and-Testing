#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_River_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Previous weekly OHLC for pivot calculation
    prev_close_w = df_w['close'].shift(1).values
    prev_high_w = df_w['high'].shift(1).values
    prev_low_w = df_w['low'].shift(1).values
    
    # Calculate weekly pivot and support/resistance levels
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    r1_w = pivot_w + (range_w * 1.0) / 2
    s1_w = pivot_w - (range_w * 1.0) / 2
    
    # Align weekly levels to daily
    pivot_d = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_d = align_htf_to_ltf(prices, df_w, r1_w)
    s1_d = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Daily volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_d[i]) or np.isnan(r1_d[i]) or np.isnan(s1_d[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with bullish bias
            if (close[i] > r1_d[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S1 with bearish bias
            elif (close[i] < s1_d[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot
            if close[i] < pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot
            if close[i] > pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals