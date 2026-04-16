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
    
    # === Daily data for 1D pivot levels (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day values for pivot calculation
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Initialize first values
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Calculate daily pivot point and S1/R1 (Fibonacci 0.382)
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    prev_range = prev_high_1d - prev_low_1d
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation (4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stop) or at opposite level R1 (mean reversion)
            if price < s1_val or price > r1_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stop) or at opposite level S1 (mean reversion)
            if price > r1_val or price < s1_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation
            if (price > r1_val) and (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume confirmation
            elif (price < s1_val) and (vol_ratio_val > 1.8):
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

name = "4h_FibPivot_R1_S1_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0