#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Donchian breakout and volume confirmation.
# Uses weekly Donchian channels (20-period) to identify breakouts in either direction.
# Volume filter ensures breakout strength. Works in both bull and bear markets by
# capturing breakouts in either direction. Target: 30-100 total trades over 4 years.
name = "1d_1w_Donchian_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels on weekly timeframe (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume filter: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above weekly Donchian high with volume
            if close[i] > donchian_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below weekly Donchian low with volume
            elif close[i] < donchian_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals