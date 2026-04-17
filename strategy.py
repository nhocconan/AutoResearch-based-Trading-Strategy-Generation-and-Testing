#!/usr/bin/env python3
"""
4h Camarilla Pivot + Volume Spike + Chop Filter
Long: Price > S1 + volume > 1.5x 4h volume SMA(20) + CHOP > 61.8 (range)
Short: Price < R1 + volume > 1.5x 4h volume SMA(20) + CHOP > 61.8 (range)
Exit: Price crosses midline (PP) or volume drops
Uses Camarilla levels from 1d, volume confirmation, and chop regime filter.
Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: PP, S1, R1, S2, R2, S3, R3
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # R2 = C + (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # R3 = C + (H - L) * 1.1 / 4
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    s1 = close_1d - range_1d * 1.1 / 12.0
    r1 = close_1d + range_1d * 1.1 / 12.0
    s2 = close_1d - range_1d * 1.1 / 6.0
    r2 = close_1d + range_1d * 1.1 / 6.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    r3 = close_1d + range_1d * 1.1 / 4.0
    
    # Align pivots to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14) - range detection
    # CHOP = 100 * log10(sum(ATR) / (n * (max_high - min_low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = high[0] - low[0]  # First bar TR is just high-low
    tr3[0] = high[0] - low[0]
    atr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    chop = np.full(n, 50.0)  # Default neutral
    for i in range(14, n):
        if range_14[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_14[i]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 14)  # Need volume SMA and chop
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(vol_sma_4h[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        chop_val = chop[i]
        pp_val = pp_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        
        # Only trade in ranging markets (CHOP > 61.8)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > S1 + volume spike
            if price > s1_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price < R1 + volume spike
            elif price < r1_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price < PP or volume drops
            if price < pp_val or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price > PP or volume drops
            if price > pp_val or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0