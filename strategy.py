#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_Pivot_Breakout_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Formula: 
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.500
    # R3 = C + Range * 1.250
    # R2 = C + Range * 1.166
    # R1 = C + Range * 1.083
    # S1 = C - Range * 1.083
    # S2 = C - Range * 1.166
    # S3 = C - Range * 1.250
    # S4 = C - Range * 1.500
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R3 = close_1d + range_1d * 1.250
    S3 = close_1d - range_1d * 1.250
    R4 = close_1d + range_1d * 1.500
    S4 = close_1d - range_1d * 1.500
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: current volume > 1.5x 24-period average (6h * 4 = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume
            if price > R4_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume
            elif price < S4_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below R3 (mean reversion)
            if price < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above S3 (mean reversion)
            if price > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals