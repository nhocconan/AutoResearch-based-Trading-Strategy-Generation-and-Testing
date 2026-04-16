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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR on 4h for stop loss
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate ATR on 12h for volatility regime
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 12h ATR ratio for volatility regime (current ATR vs 50-period average) ===
    atr_ma_50 = pd.Series(atr_12h_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_12h_aligned / atr_ma_50
    
    # === 4h Donchian Channel (20) for breakout signals ===
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    
    # === 4h Volume spike detection ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_4h / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]
        atr_4h_val = atr_4h_aligned[i]
        atr_ratio_val = atr_ratio[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR volatility increases significantly
            if (price < donchian_lower[i]) or (atr_ratio_val > 1.5):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR volatility increases significantly
            if (price > donchian_upper[i]) or (atr_ratio_val > 1.5):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND low volatility regime AND volume spike
            if (price > donchian_upper[i]) and (atr_ratio_val < 0.8) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND low volatility regime AND volume spike
            elif (price < donchian_lower[i]) and (atr_ratio_val < 0.8) and (vol_ratio_val > 2.0):
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

name = "4h_Donchian_Breakout_LowVol_Volume_12hFilter"
timeframe = "4h"
leverage = 1.0