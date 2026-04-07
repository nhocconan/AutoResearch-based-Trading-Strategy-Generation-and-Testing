#!/usr/bin/env python3
"""
12H Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Long when price breaks above weekly Donchian channel (20-period high) with expanding volume AND weekly trend up
Short when price breaks below weekly Donchian channel (20-period low) with expanding volume AND weekly trend down
Exit when price crosses back to the midline of the weekly Donchian channel
Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channel (20-period) ===
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align weekly Donchian levels to 12h timeframe (with shift(1) for completed bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # === Volume confirmation (on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below midline
            if close[i] < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above midline
            if close[i] > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly Donchian breakout with volume confirmation
            if close[i] > donchian_high_aligned[i]:
                # Breakout above weekly Donchian high -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low_aligned[i]:
                # Breakdown below weekly Donchian low -> short
                position = -1
                signals[i] = -0.25
    
    return signals