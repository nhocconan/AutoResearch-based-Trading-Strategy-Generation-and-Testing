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
    
    # === 1w Donchian Channel (20-period) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian upper/lower bands
    donch_upper = np.full_like(high_1w, np.nan)
    donch_lower = np.full_like(low_1w, np.nan)
    period = 20
    for i in range(len(high_1w)):
        if i >= period - 1:
            donch_upper[i] = np.max(high_1w[i-(period-1):i+1])
            donch_lower[i] = np.min(low_1w[i-(period-1):i+1])
        elif i > 0:
            donch_upper[i] = np.max(high_1w[0:i+1])
            donch_lower[i] = np.min(low_1w[0:i+1])
        else:
            donch_upper[i] = high_1w[0]
            donch_lower[i] = low_1w[0]
    
    # === 12h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === Align Donchian levels to 12h timeframe ===
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    
    # === 12h Close price relative to Donchian channels ===
    price_vs_upper = close > donch_upper_aligned
    price_vs_lower = close < donch_lower_aligned
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Close breaks above Donchian upper band + volume confirmation
            if price_vs_upper[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below Donchian lower band + volume confirmation
            elif price_vs_lower[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility drop
        elif position == 1:
            # Exit long: Close breaks below Donchian lower band OR volume drops
            if price_vs_lower[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close breaks above Donchian upper band OR volume drops
            if price_vs_upper[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0