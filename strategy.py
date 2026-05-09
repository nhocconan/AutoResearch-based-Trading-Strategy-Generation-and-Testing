#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 12h Donchian breakout and volume confirmation.
# Uses 12h Donchian channels (20-period) for structural breakouts and 12h volume spike for confirmation.
# Donchian breakouts capture momentum in both trending and ranging markets, while volume filters ensure
# institutional participation. This combination has shown robustness across market regimes.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Donchian20_12hVolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for Donchian channels and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h average volume (20-period)
    avg_volume_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # Breakout conditions: price breaks Donchian channel
    breakout_up = close > donchian_high_aligned
    breakout_down = close < donchian_low_aligned
    
    # Volume filter: current volume > 2.0x 12h average volume
    volume_spike = volume > (2.0 * avg_volume_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike
            if breakout_up[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume spike
            elif breakout_down[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low or opposite breakout
            if close[i] <= donchian_low_aligned[i] or breakout_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high or opposite breakout
            if close[i] >= donchian_high_aligned[i] or breakout_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals