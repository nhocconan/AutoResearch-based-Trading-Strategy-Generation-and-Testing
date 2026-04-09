#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channel on 12h (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_12h[i-19:i+1])
            donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: 4-period average on 6h (24h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR volume drops
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if close[i] < donchian_low_6h[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR volume drops
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if close[i] > donchian_high_6h[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if close[i] > donchian_high_6h[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low with volume confirmation
            elif close[i] < donchian_low_6h[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals