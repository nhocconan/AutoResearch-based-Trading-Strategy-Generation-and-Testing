#!/usr/bin/env python3
"""
Hypothesis: 4-hour price position relative to Donchian channel combined with 1-day volume confirmation.
Long when price breaks above 4h Donchian upper band (20) with above-average 1d volume.
Short when price breaks below 4h Donchian lower band (20) with above-average 1d volume.
Exit when price returns to the 4h Donchian midline (average of upper/lower).
Designed for low-to-moderate trade frequency (~15-35/year) to capture breakouts while avoiding whipsaws.
Volume filter ensures breakouts have participation, reducing false signals.
Works in both bull (catching breakouts) and bear (catching breakdowns) markets.
"""

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
    
    # Load 4-hour data for Donchian channel - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1-day data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-period)
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1h volume > 1.5x average 1d volume (scaled to hourly)
        # Approximate: 1d volume / 24 = average hourly volume
        volume_threshold = avg_volume_1d_aligned[i] / 24.0 * 1.5
        volume_ok = volume[i] > volume_threshold
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume confirmation
            if close[i] > donchian_upper_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower with volume confirmation
            elif close[i] < donchian_lower_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian midline
                if close[i] <= donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian midline
                if close[i] >= donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "4H_Donchian_Breakout_1dVolumeConfirmation"
timeframe = "1h"
leverage = 1.0