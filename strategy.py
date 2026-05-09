#!/usr/bin/env python3
# 12h_Range_Breakout_With_Volume_Filter
# Hypothesis: Uses 12h price action with 1d range (high-low) as dynamic support/resistance.
# Enters long when price breaks above 1d high with volume confirmation (>1.5x avg).
# Enters short when price breaks below 1d low with volume confirmation.
# Exits when price returns to 1d midpoint. Designed to capture breakouts from daily ranges
# in both trending and ranging markets. Volume filter reduces false breakouts.
# Target: 15-30 trades/year per symbol.

name = "12h_Range_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily high and low
    daily_high = high_1d
    daily_low = low_1d
    daily_mid = (high_1d + low_1d) / 2
    
    # Align daily levels to 12h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_mid_aligned = align_htf_to_ltf(prices, df_1d, daily_mid)
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or \
           np.isnan(daily_mid_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above daily high with volume confirmation
            if close[i] > daily_high_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below daily low with volume confirmation
            elif close[i] < daily_low_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to or below daily midpoint
            if close[i] <= daily_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to or above daily midpoint
            if close[i] >= daily_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals