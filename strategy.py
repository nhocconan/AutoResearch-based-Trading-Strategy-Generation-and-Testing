#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data for price action ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 6h Donchian (20-period) ===
    highest_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_6h = align_htf_to_ltf(prices, df_6h, highest_6h)
    donchian_low_6h = align_htf_to_ltf(prices, df_6h, lowest_6h)
    
    # === 1d ATR (14-period) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d volume MA (20-period) for volume spike ===
    vol_ma_1d = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values  # using 6h volume for consistency
    volume_spike = volume_6h > (1.8 * vol_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_6h, volume_spike)
    
    # === Weekly pivot points (from weekly data) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + range_1w
    s1_1w = pivot_1w - range_1w
    r2_1w = pivot_1w + 2 * range_1w
    s2_1w = pivot_1w - 2 * range_1w
    
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 80
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_channel = donchian_high_6h[i]
        lower_channel = donchian_low_6h[i]
        atr = atr_1d_aligned[i]
        vol_spike = volume_spike_aligned[i]
        weekly_pivot = pivot_1w_aligned[i]
        weekly_r1 = r1_1w_aligned[i]
        weekly_s1 = s1_1w_aligned[i]
        weekly_r2 = r2_1w_aligned[i]
        weekly_s2 = s2_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches or crosses weekly pivot (mean reversion to weekly pivot)
            if price <= weekly_pivot:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses weekly pivot (mean reversion to weekly pivot)
            if price >= weekly_pivot:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 6h Donchian high with volume spike and above weekly R1
            if price > upper_channel and vol_spike and price > weekly_r1:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 6h Donchian low with volume spike and below weekly S1
            elif price < lower_channel and vol_spike and price < weekly_s1:
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

name = "6h_Donchian20_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0