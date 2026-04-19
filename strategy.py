#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Level_Touch"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Camarilla pivot levels for previous day
    # R4, R3, R2, R1, S1, S2, S3, S4
    # Based on previous day's high, low, close
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    R4 = close_1d + range_hl * 1.500
    R3 = close_1d + range_hl * 1.250
    R2 = close_1d + range_hl * 1.166
    R1 = close_1d + range_hl * 1.083
    S1 = close_1d - range_hl * 1.083
    S2 = close_1d - range_hl * 1.166
    S3 = close_1d - range_hl * 1.250
    S4 = close_1d - range_hl * 1.500
    
    # Align levels to 4h timeframe (they are fixed for the day)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price touches S1 level with volume
            if price <= S1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price touches R1 level with volume
            elif price >= R1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot or reaches S2 (stop)
            if price >= pivot_aligned[i] or price <= S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot or reaches R2 (stop)
            if price <= pivot_aligned[i] or price >= R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals