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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (HTF for trend) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 12h Donchian Channel (20-period) ===
    # Upper band: 20-period high
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle line: (upper + lower) / 2
    middle_line = (highest_high_20 + lowest_low_20) / 2
    
    # Align Donchian bands with proper delay (wait for 12h bar to close)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_20)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_20)
    middle_12h_aligned = align_htf_to_ltf(prices, df_12h, middle_line)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === 12h ATR for stop loss (14-period) ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and ATR
    warmup = 40
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(middle_12h_aligned[i]) or np.isnan(vol_ratio_4h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = upper_12h_aligned[i]
        lower = lower_12h_aligned[i]
        middle = middle_12h_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        atr = atr_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below middle line or stop loss hit
            if price < middle or price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above middle line or stop loss hit
            if price > middle or price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper Donchian band with volume confirmation
            if price > upper and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: price breaks below lower Donchian band with volume confirmation
            elif price < lower and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_20_12h_Trend_Volume"
timeframe = "4h"
leverage = 1.0