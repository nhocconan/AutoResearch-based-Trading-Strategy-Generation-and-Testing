#!/usr/bin/env python3
"""
6h Donchian breakout + weekly pivot direction + volume confirmation
Long when price breaks above Donchian(20) high and weekly pivot > prior close
Short when price breaks below Donchian(20) low and weekly pivot < prior close
Exit when price crosses back through Donchian midpoint
Designed to capture breakouts with institutional bias from weekly pivot
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === Volume Filter (20-period average) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    # Typical price for weekly pivot calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly pivot point
    weekly_pivot = (df_1w['high'].iloc[0] + df_1w['low'].iloc[0] + df_1w['close'].iloc[0]) / 3  # Simplified: use first bar's typical price
    # Actually compute properly for each weekly bar
    weekly_pivot_vals = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_avg[i]) or np.isnan(weekly_pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: break above Donchian high with weekly pivot bias up
            if (close[i] > highest_high[i] and 
                weekly_pivot_aligned[i] > close[i-1] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian low with weekly pivot bias down
            elif (close[i] < lowest_low[i] and 
                  weekly_pivot_aligned[i] < close[i-1] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals