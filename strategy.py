#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w Camarilla pivot direction + volume confirmation
# Entry on 6h Donchian breakout only when aligned with 1w Camarilla pivot bias:
#   Long when price > weekly R4 (bullish bias), short when price < weekly S4 (bearish bias)
# Volume confirmation: current 6h volume > 1.5x 20-period average of 6h volume
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Weekly pivot provides structural bias that works in both bull and bear markets via mean reversion at extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data for Donchian, volume ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === Donchian Channel (20-period) on 6h ===
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 1w Camarilla Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for weekly timeframe
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2.0)
    s4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2.0)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # === EXIT LOGIC (Donchian opposite break) ===
        if position == 1:  # Long position
            # Exit when price breaks below 6h Donchian lower band
            if price < lower_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above 6h Donchian upper band
            if price > upper_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper band AND price > weekly R4 AND volume confirmation
            if price > upper_val and price > r4_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below lower band AND price < weekly S4 AND volume confirmation
            elif price < lower_val and price < s4_val and vol_confirm:
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

name = "6h_Donchian20_1wCamarilla_R4S4_VolumeConfirm"
timeframe = "6h"
leverage = 1.0