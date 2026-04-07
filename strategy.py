#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1d Volume Confirmation
# Hypothesis: Breakouts of 20-period Donchian channels on 12h timeframe, confirmed by above-average volume on 1d,
# capture significant moves while avoiding whipsaws in ranging markets.
# Uses 1d volume filter to ensure breakouts have institutional participation.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

name = "12h_donchian20_1d_volume_confirm_v1"
timeframe = "12h"
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 20-period Donchian channels on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_vol_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below midpoint of Donchian channel
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above midpoint of Donchian channel
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current 12h volume > 20-period average 1d volume
            vol_confirm = volume[i] > avg_vol_20_aligned[i]
            
            # Long breakout: price closes above 20-period high
            if close[i] > highest_high[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakout: price closes below 20-period low
            elif close[i] < lowest_low[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals