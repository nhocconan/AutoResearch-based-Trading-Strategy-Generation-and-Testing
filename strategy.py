#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF for trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1d 20-period Donchian channels ===
    # Calculate rolling max/min for 20 periods
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1w 34-period EMA for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1d volume ratio for confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # Align HTF indicators to 1d
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1w)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for 20-period Donchian and EMA34
    warmup = 35
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donchian_high = high_max_20_aligned[i]
        donchian_low = low_min_20_aligned[i]
        ema_34 = ema_34_1w_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if price <= donchian_low or price < ema_34:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if price >= donchian_high or price > ema_34:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above Donchian high, above weekly EMA, volume confirmation
            if price > donchian_high and price > ema_34 and vol_ratio > 1.8:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, below weekly EMA, volume confirmation
            elif price < donchian_low and price < ema_34 and vol_ratio > 1.8:
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

name = "1d_Donchian_20_1w_EMA34_Volume"
timeframe = "1d"
leverage = 1.0