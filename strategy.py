#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h structure (Donchian breakout) and volume confirmation.
Trade breakouts of 12h Donchian channels (20-period) with volume filter.
Use 4h only for entry timing to keep trade frequency low (target: 20-50/year).
Works in bull markets via trend-following breakouts and in bear via mean-reversion at 12h structure.
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
    
    # Get 12h data for structure (Donchian channels)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h data to 4h
    high_max_20_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # Volume filter: current volume > 1.5x 20-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume
            if close[i] > high_max_20_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low with volume
            elif close[i] < low_min_20_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian20_Volume"
timeframe = "4h"
leverage = 1.0