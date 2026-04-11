#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = typical_price
    # Support and resistance levels
    range_hl = df_1d['high'] - df_1d['low']
    r1 = close.shift(1) + range_hl * 1.1 / 12
    r2 = close.shift(1) + range_hl * 1.1 / 6
    r3 = close.shift(1) + range_hl * 1.1 / 4
    r4 = close.shift(1) + range_hl * 1.1 / 2
    s1 = close.shift(1) - range_hl * 1.1 / 12
    s2 = close.shift(1) - range_hl * 1.1 / 6
    s3 = close.shift(1) - range_hl * 1.1 / 4
    s4 = close.shift(1) - range_hl * 1.1 / 2
    
    # Align levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # 1d volume confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Price levels
        price = close[i]
        
        # Entry conditions: price touches S3 or R3 with volume confirmation
        long_signal = vol_confirm and (price <= s3_aligned[i])
        short_signal = vol_confirm and (price >= r3_aligned[i])
        
        # Exit conditions: price reaches S1/R1 or opposite S3/R3
        long_exit = (price >= s1_aligned[i]) or (price <= s4_aligned[i])
        short_exit = (price <= r1_aligned[i]) or (price >= r4_aligned[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals