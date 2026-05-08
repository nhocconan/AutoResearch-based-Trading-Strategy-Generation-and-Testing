#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PivotReversal_Bias_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Daily pivot and support/resistance levels
    pivot_1d = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    range_1d = high_1d_prev - low_1d_prev
    s1_1d = pivot_1d - (range_1d * 1)
    r1_1d = pivot_1d + (range_1d * 1)
    s2_1d = pivot_1d - (range_1d * 2)
    r2_1d = pivot_1d + (range_1d * 2)
    
    # Align daily levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # Daily trend bias: close vs 50-period EMA
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s2_val = s2_1d_aligned[i]
        r2_val = r2_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long setup: price near S2 + bullish bias + volume spike
            if (close[i] >= s2_val * 0.995 and close[i] <= s2_val * 1.005 and
                close[i] > ema50_val and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short setup: price near R2 + bearish bias + volume spike
            elif (close[i] >= r2_val * 0.995 and close[i] <= r2_val * 1.005 and
                  close[i] < ema50_val and
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches pivot or R1
            if close[i] >= pivot_val or close[i] >= r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches pivot or S1
            if close[i] <= pivot_val or close[i] <= s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals