#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_TopBottom_Close_Reversal_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day high/low for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian high/low on daily
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume filter: current volume > 1.5x 20-period 4h average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high_20_aligned[i]
        lower = donch_low_20_aligned[i]
        
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long reversal: close near 20-day low with volume
            if price <= lower * 1.005 and volume_ok:  # within 0.5% of lower band
                signals[i] = 0.25
                position = 1
            # Short reversal: close near 20-day high with volume
            elif price >= upper * 0.995 and volume_ok:  # within 0.5% of upper band
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close back above midpoint or stop loss
            midpoint = (upper + lower) * 0.5
            if price >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close back below midpoint or stop loss
            midpoint = (upper + lower) * 0.5
            if price <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals