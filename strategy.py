#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and Choppiness Filter.
Long when price touches S3 with volume spike in choppy market.
Short when price touches R3 with volume spike in choppy market.
Exit when price reaches S2/R2 or crosses back to pivot.
Uses Camarilla pivot levels from 1d timeframe for precision levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_reversal_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Choppiness Index (14) for regime filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    range_sum = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / (range_sum + 1e-10)) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when undefined
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Load 1d data for Camarilla pivots (HIGHER TIMEFRAME) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    r2 = pivot + range_hl * 1.1 / 6
    s2 = pivot - range_hl * 1.1 / 6
    
    # Align to 4h timeframe (already shifted by get_htf_data + shift(1) above)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(chop_values[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S2 or crosses below pivot
            if close[i] <= s2_aligned[i] or close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R2 or crosses above pivot
            if close[i] >= r2_aligned[i] or close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Choppy market filter (chop > 50 = ranging)
            if chop_values[i] <= 50:
                signals[i] = 0.0
                continue
            
            # Volume spike confirmation
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Price touches S3/R3 with volume spike
            # Allow small tolerance for touching the level
            tolerance = 0.001 * close[i]  # 0.1% tolerance
            
            if abs(close[i] - s3_aligned[i]) <= tolerance:
                # Touched S3 -> long
                position = 1
                signals[i] = 0.25
            elif abs(close[i] - r3_aligned[i]) <= tolerance:
                # Touched R3 -> short
                position = -1
                signals[i] = -0.25
    
    return signals