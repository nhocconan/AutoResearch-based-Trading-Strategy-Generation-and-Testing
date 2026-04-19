#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_Breakout_Volume_Spread"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot points (standard calculation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Pivot levels
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume moving average (20-period on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or \
           np.isnan(s2_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.2x 20-period average
        volume_ok = vol > 1.2 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if price > r1_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif price < s1_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or breaks S1 (reversal)
            if price < pivot_1d_aligned[i] or price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or breaks R1 (reversal)
            if price > pivot_1d_aligned[i] or price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals