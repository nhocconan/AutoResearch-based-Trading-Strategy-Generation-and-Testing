#!/usr/bin/env python3
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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for Pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Pivot levels (using previous day's OHLC) ===
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        pivot[i] = (h + l + c) / 3.0
        r1[i] = 2 * pivot[i] - l
        s1[i] = 2 * pivot[i] - h
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    # Align all HTF data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for pivot calculations
    warmup = 20
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        p = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1
            if price < s1_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1
            if price > r1_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume
            if price > r1_val and vol_ratio > 1.3:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below S1 with volume
            elif price < s1_val and vol_ratio > 1.3:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0