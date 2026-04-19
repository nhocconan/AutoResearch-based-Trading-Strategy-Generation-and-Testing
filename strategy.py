#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Pivot_R1S1_Breakout_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from previous week
    prev_close_w = np.roll(close_1w, 1)
    prev_close_w[0] = np.nan
    prev_high_w = np.roll(high_1w, 1)
    prev_high_w[0] = np.nan
    prev_low_w = np.roll(low_1w, 1)
    prev_low_w[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 12.0
    
    # Calculate daily pivot levels from previous day
    prev_close_d = np.roll(close_1d, 1)
    prev_close_d[0] = np.nan
    prev_high_d = np.roll(high_1d, 1)
    prev_high_d[0] = np.nan
    prev_low_d = np.roll(low_1d, 1)
    prev_low_d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 12.0
    
    # Align to 12h timeframe
    pivot_w_12h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_12h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_12h = align_htf_to_ltf(prices, df_1w, s1_w)
    pivot_d_12h = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_12h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_12h = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_w_12h[i]) or np.isnan(r1_w_12h[i]) or np.isnan(s1_w_12h[i]) or \
           np.isnan(pivot_d_12h[i]) or np.isnan(r1_d_12h[i]) or np.isnan(s1_d_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        # Use weekly R1/S1 as primary levels, daily as secondary filter
        if position == 0:
            # Long: Price breaks above weekly R1 with volume spike and above daily pivot
            if price > r1_w_12h[i] and volume_spike and price > pivot_d_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume spike and below daily pivot
            elif price < s1_w_12h[i] and volume_spike and price < pivot_d_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly S1 (reversal signal)
            if price < s1_w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly R1 (reversal signal)
            if price > r1_w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals