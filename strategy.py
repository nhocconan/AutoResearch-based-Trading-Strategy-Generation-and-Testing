#!/usr/bin/env python3
# Hypothesis: 12h timeframe with daily Williams Alligator (3 SMAs) for trend direction and 1d volume spike for confirmation.
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trends: Lips > Teeth > Jaw = uptrend, reverse for downtrend.
# Volume spike (>2x 20-period average) confirms momentum. Williams Alligator works in both bull and bear markets by
# clearly defining trend direction and avoiding whipsaws during sideways periods.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WilliamsAlligator_1dVolumeSpike"
timeframe = "12h"
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
    
    # Calculate Williams Alligator components from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Trend conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    trend_up = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    trend_down = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Volume filter: current volume > 2.0x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams Alligator uptrend + volume spike
            if trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams Alligator downtrend + volume spike
            elif trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal (Lips cross below Teeth)
            if lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal (Lips cross above Teeth)
            if lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals