#!/usr/bin/env python3
# 6h_LongTerm_Swing_With_Volume_Filter
# Hypothesis: 6h swing trading using 20-period EMA crossover with volume confirmation.
# Long when price crosses above EMA20 and volume > 1.5x average volume.
# Short when price crosses below EMA20 and volume > 1.5x average volume.
# Volume filter reduces whipsaw by requiring institutional participation.
# Works in both bull and bear markets by following intermediate-term trends.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_LongTerm_Swing_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period EMA
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 20 to ensure EMA and volume data is available
    for i in range(20, n):
        # Skip if EMA or volume data is not available
        if np.isnan(ema20[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume must be > 1.5x average volume
        volume_filter = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Look for EMA crossover with volume confirmation
            if close[i] > ema20[i] and close[i-1] <= ema20[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < ema20[i] and close[i-1] >= ema20[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below EMA20
            if close[i] < ema20[i] and close[i-1] >= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above EMA20
            if close[i] > ema20[i] and close[i-1] <= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals