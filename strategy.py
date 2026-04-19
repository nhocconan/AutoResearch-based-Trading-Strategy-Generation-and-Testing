#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R4S4_Breakout_Volume_Spike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from previous weekly bar
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w = np.roll(high_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w = np.roll(low_1w, 1)
    prev_low_1w[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 2.0
    
    # Calculate daily pivot levels from previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    
    # Align to 6h timeframe
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_6h = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_6h = align_htf_to_ltf(prices, df_1w, s4_1w)
    pivot_1d_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_6h[i]) or np.isnan(r4_1w_6h[i]) or np.isnan(s4_1w_6h[i]) or \
           np.isnan(pivot_1d_6h[i]) or np.isnan(r4_1d_6h[i]) or np.isnan(s4_1d_6h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly R4 with volume spike and above daily pivot
            if price > r4_1w_6h[i] and volume_spike and price > pivot_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S4 with volume spike and below daily pivot
            elif price < s4_1w_6h[i] and volume_spike and price < pivot_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly S4 (reversal signal)
            if price < s4_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly R4 (reversal signal)
            if price > r4_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals