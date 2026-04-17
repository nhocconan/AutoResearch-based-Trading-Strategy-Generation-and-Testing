#!/usr/bin/env python3
"""
12h_DonchianBreakout_Volume_Confirmation
Breakout strategy using 1d Donchian(20) channels for direction, 12h price breakout for entry, and 1d volume surge confirmation.
Long when: Price breaks above 1d Donchian upper channel + 1d volume > 1.5x 20-day average.
Short when: Price breaks below 1d Donchian lower channel + 1d volume > 1.5x 20-day average.
Exit when price returns to the 1d Donchian midpoint.
Position size: 0.25. Target: 15-35 trades/year.
Uses 1d for trend structure and volume confirmation, 12h for entry timing. Works in bull/bear: breakouts capture momentum in both directions.
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2.0
    
    # Calculate 20-day average volume on 1d
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # warmup for Donchian channels
        # Skip if any required data is not available
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(donch_mid_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or
            np.isnan(volume_1d_current[i])):
            signals[i] = 0.0
            continue
        
        volume_filter = volume_1d_current[i] > (1.5 * volume_ma20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper channel + volume surge
            if close[i] > donch_high_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel + volume surge
            elif close[i] < donch_low_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if close[i] > donch_mid_1d_aligned[i]:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if close[i] < donch_mid_1d_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DonchianBreakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0