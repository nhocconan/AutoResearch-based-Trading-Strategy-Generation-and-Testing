#!/usr/bin/env python3
# 12h_1w_donchian_breakout_volume_v1
# Hypothesis: Trade weekly Donchian channel breakouts with volume confirmation on 12h timeframe.
# Long when price breaks above 20-period weekly high with volume surge.
# Short when price breaks below 20-period weekly low with volume surge.
# Uses 12-hour timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Volume filter ensures breakout strength, reducing false signals.
# Works in bull markets (catching breakouts) and bear markets (catching breakdowns).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period weekly Donchian channels
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Volume confirmation: volume > 2x 24-period average (12 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_24[i] if vol_ma_24[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period weekly low
            if close[i] < low_min_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period weekly high
            if close[i] > high_max_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period weekly high with volume surge
            if (close[i] > high_max_20_aligned[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period weekly low with volume surge
            elif (close[i] < low_min_20_aligned[i] and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals