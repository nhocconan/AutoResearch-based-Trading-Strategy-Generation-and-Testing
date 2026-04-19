#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Pivot_R4S4_Breakout_Volume_Confirmation"
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
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily pivot levels from previous day
    prev_close_d = np.roll(close_1d, 1)
    prev_close_d[0] = np.nan
    prev_high_d = np.roll(high_1d, 1)
    prev_high_d[0] = np.nan
    prev_low_d = np.roll(low_1d, 1)
    prev_low_d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 2.0
    
    # Calculate weekly pivot levels from previous week
    prev_close_w = np.roll(close_1w, 1)
    prev_close_w[0] = np.nan
    prev_high_w = np.roll(high_1w, 1)
    prev_high_w[0] = np.nan
    prev_low_w = np.roll(low_1w, 1)
    prev_low_w[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 2.0
    
    # Align to 6h timeframe
    pivot_d_6h = align_htf_to_ltf(prices, df_1d, pivot_d)
    r4_d_6h = align_htf_to_ltf(prices, df_1d, r4_d)
    s4_d_6h = align_htf_to_ltf(prices, df_1d, s4_d)
    pivot_w_6h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r4_w_6h = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_6h = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_d_6h[i]) or np.isnan(r4_d_6h[i]) or np.isnan(s4_d_6h[i]) or \
           np.isnan(pivot_w_6h[i]) or np.isnan(r4_w_6h[i]) or np.isnan(s4_w_6h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        # Use daily R4/S4 as breakout levels, weekly as trend filter
        if position == 0:
            # Long: Price breaks above daily R4 with volume spike and above weekly pivot
            if price > r4_d_6h[i] and volume_spike and price > pivot_w_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S4 with volume spike and below weekly pivot
            elif price < s4_d_6h[i] and volume_spike and price < pivot_w_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below daily S4 (reversal signal)
            if price < s4_d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above daily R4 (reversal signal)
            if price > r4_d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals