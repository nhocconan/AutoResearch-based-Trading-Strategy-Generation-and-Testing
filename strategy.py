#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for 20-period Donchian channel ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian(20) upper/lower bands
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # === Daily data for pivot and volume context ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Daily Pivot Points (using standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_hl_1d = high_1d - low_1d
    r1_1d = pivot_1d + range_hl_1d * 0.382
    s1_1d = pivot_1d - range_hl_1d * 0.382
    r2_1d = pivot_1d + range_hl_1d * 0.618
    s2_1d = pivot_1d - range_hl_1d * 0.618
    
    # === Daily volume spike detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # === Align HTF data to 6h timeframe ===
    donchian_upper_6h = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_lower_6h = align_htf_to_ltf(prices, df_1w, low_20)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i]) or
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper_6h[i]
        lower = donchian_lower_6h[i]
        r1 = r1_6h[i]
        s1 = s1_6h[i]
        r2 = r2_6h[i]
        s2 = s2_6h[i]
        vol_spike = volume_spike_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1 or breaks below weekly lower band
            if price < s1 or price < lower:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or breaks above weekly upper band
            if price > r1 or price > upper:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly upper band with volume spike, above R1 pivot
            if price > upper and vol_spike and price > r1:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly lower band with volume spike, below S1 pivot
            elif price < lower and vol_spike and price < s1:
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

name = "6h_Donchian20_Weekly_PivotR1S1_VolumeSpike"
timeframe = "6h"
leverage = 1.0