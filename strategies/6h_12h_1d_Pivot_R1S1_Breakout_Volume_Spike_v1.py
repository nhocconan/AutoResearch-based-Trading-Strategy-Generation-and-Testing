#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Pivot_R1S1_Breakout_Volume_Spike_v1"
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
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h pivot levels from previous 12h bar
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h = np.roll(high_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h = np.roll(low_12h, 1)
    prev_low_12h[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_12h = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_12h = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12.0
    
    # Calculate 1d pivot levels from previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    # Align to 6h timeframe
    pivot_12h_6h = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_6h = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_6h = align_htf_to_ltf(prices, df_12h, s1_12h)
    pivot_1d_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_12h_6h[i]) or np.isnan(r1_12h_6h[i]) or np.isnan(s1_12h_6h[i]) or \
           np.isnan(pivot_1d_6h[i]) or np.isnan(r1_1d_6h[i]) or np.isnan(s1_1d_6h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 12h R1 with volume spike and above 1d pivot
            if price > r1_12h_6h[i] and volume_spike and price > pivot_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h S1 with volume spike and below 1d pivot
            elif price < s1_12h_6h[i] and volume_spike and price < pivot_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 12h S1 (reversal signal)
            if price < s1_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 12h R1 (reversal signal)
            if price > r1_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals