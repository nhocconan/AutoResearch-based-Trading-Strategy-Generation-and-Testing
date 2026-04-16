#!/usr/bin/env python3
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
    
    # 4h data (primary)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 1d data (HTF for trend and pivot)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h Donchian channel (20-period) ===
    donchian_high = np.full_like(close_4h, np.nan)
    donchian_low = np.full_like(close_4h, np.nan)
    for i in range(len(close_4h)):
        if i < 20:
            donchian_high[i] = np.max(high_4h[:i+1])
            donchian_low[i] = np.min(low_4h[:i+1])
        else:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # === 1d daily pivot points ===
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        pivot_1d[i] = pp
        r1_1d[i] = 2 * pp - low_1d[i-1]
        s1_1d[i] = 2 * pp - high_1d[i-1]
    
    # === 1d volume ratio for confirmation ===
    vol_ma_10_1d = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = volume_4h / vol_ma_10_1d
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and pivots
    warmup = 20
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        pt = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 (stop loss)
            if price < s1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 (stop loss)
            if price > r1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume expansion
            if price > r1 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price breaks below S1 with volume expansion
            elif price < s1 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0