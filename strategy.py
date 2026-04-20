#!/usr/bin/env python3
"""
4h_Donchian20_1dVolume_Conservative_v1
Concept: 4h Donchian(20) breakout with 1d volume confirmation only.
- Long: Price > Donchian high(20) AND 1d volume > 1.5x 20-period average
- Short: Price < Donchian low(20) AND 1d volume > 1.5x 20-period average
- Exit: Price crosses back through Donchian midpoint
- Position sizing: 0.25
- Target: 75-200 total trades over 4 years (19-50/year)
- Works in bull/bear: Volume surge confirms institutional interest behind breakout
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_Conservative_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # === 4h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Get values
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        current_vol = vol_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or 
            np.isnan(vol_ma) or np.isnan(current_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_condition = current_vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge
            if close[i] > dh and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge
            elif close[i] < dl and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if close[i] < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if close[i] > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals