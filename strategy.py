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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (HTF for trend context) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 12h EMA(50) for trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Donchian(20) breakout ===
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and EMA
    warmup = 50
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_12h_aligned[i]
        upper = highest_high_20[i]
        lower = lowest_low_20[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price crosses below 12h EMA50 or volume dries up
            if price < ema_50 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price crosses above 12h EMA50 or volume dries up
            if price > ema_50 or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above Donchian upper + above 12h EMA50 + volume surge
            if price > upper and price > ema_50 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: price breaks below Donchian lower + below 12h EMA50 + volume surge
            elif price < lower and price < ema_50 and vol_ratio > 1.5:
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

name = "6h_Donchian_EMA50_Volume"
timeframe = "6h"
leverage = 1.0