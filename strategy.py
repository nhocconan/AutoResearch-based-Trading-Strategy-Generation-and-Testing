#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with volume confirmation on 1-hour timeframe.
Long when price breaks above 20-period 4h high AND 1h volume > 1.5x average volume.
Short when price breaks below 20-period 4h low AND 1h volume > 1.5x average volume.
Exit when price crosses 20-period 4h mid-point or volume drops below average.
4h provides directional bias (trend), 1h provides precise entry/exit timing.
Volume confirmation ensures institutional participation and reduces false breakouts.
Works in both bull and bear markets by following breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian channels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Volume filter: 1h volume > 1.5x average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below 4h Donchian mid-point
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above 4h Donchian mid-point
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "4H_Donchian_Breakout_1hVolume_Volume"
timeframe = "1h"
leverage = 1.0