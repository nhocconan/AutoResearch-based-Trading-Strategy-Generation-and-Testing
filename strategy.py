#!/usr/bin/env python3
"""
4h_PriceAction_With_Volume_Momentum
Hypothesis: Trade 4h breakouts of 20-period high/low with volume momentum confirmation.
Long when price breaks above 20-period high + volume > 1.5x average volume.
Short when price breaks below 20-period low + volume > 1.5x average volume.
Exit when price crosses 10-period EMA in opposite direction.
Volume momentum filters false breakouts, EMA exit reduces whipsaw.
Works in bull/bear: breakouts capture trends, volume filter avoids fakeouts.
Target: 100-200 total trades over 4 years (25-50/year) with position size 0.25.
"""

name = "4h_PriceAction_With_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

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
    
    # 20-period highest high and lowest low
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # 20-period average volume for momentum filter
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 10-period EMA for exit
    ema10 = np.full(n, np.nan)
    if n >= 10:
        multiplier = 2.0 / (10 + 1)
        ema10[9] = np.mean(close[:10])
        for i in range(10, n):
            ema10[i] = multiplier * close[i] + (1 - multiplier) * ema10[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema10[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-period high + volume momentum
            if close[i] > highest_high[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low + volume momentum
            elif close[i] < lowest_low[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-period EMA
            if close[i] < ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-period EMA
            if close[i] > ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals