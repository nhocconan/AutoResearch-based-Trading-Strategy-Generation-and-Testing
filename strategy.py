#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter"
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
    
    # Get weekly data for pivot and Donchian calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1_1w = 2 * pivot_1w - low_1w  # R1
    s1_1w = 2 * pivot_1w - high_1w  # S1
    r2_1w = pivot_1w + (high_1w - low_1w)  # R2
    s2_1w = pivot_1w - (high_1w - low_1w)  # S2
    
    # Weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # 6h volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        donchian_high = high_max_20_aligned[i]
        donchian_low = low_min_20_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above weekly R2 AND weekly Donchian high with volume
            if price > r2 and price > donchian_high and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 AND weekly Donchian low with volume
            elif price < s2 and price < donchian_low and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price falls below weekly pivot OR weekly Donchian low
            if price < pivot or price < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above weekly pivot OR weekly Donchian high
            if price > pivot or price > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals