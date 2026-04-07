#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1d Volume Confirmation
# Hypothesis: Price breakouts above/below 4h Donchian(20) channels, confirmed by 1d volume spikes,
# capture breakout moves in both bull and bear markets. Volume ensures breakout validity.
# Donchian provides objective trend structure, reducing whipsaw.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "4h_donchian_breakout_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 4h
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 4h Donchian channels (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_vol_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average volume
        volume_spike = volume[i] > 1.5 * avg_vol_20_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or volume dries up
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or volume dries up
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian high with volume spike
            if close[i] > donch_high[i] and volume_spike:
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below Donchian low with volume spike
            elif close[i] < donch_low[i] and volume_spike:
                position = -1
                signals[i] = -0.25
    
    return signals