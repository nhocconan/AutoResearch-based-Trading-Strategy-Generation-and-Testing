#!/usr/bin/env python3
"""
12h_ThreeLineBreak_Trend_With_Volume
Strategy: 12h trend using 3-line break chart with volume confirmation.
Long: 3-line break bullish pattern (3 consecutive closes > prior high) with volume > 1.5x 20-period average.
Short: 3-line break bearish pattern (3 consecutive closes < prior low) with volume > 1.5x 20-period average.
Exit: Opposite pattern or volume drop below 0.8x average.
Uses volume to filter false breakouts and reduce overtrading.
Target: 20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via trend-following + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 3-line break calculation
    # Track the last three closes to determine trend
    line1 = np.full(n, np.nan)
    line2 = np.full(n, np.nan)
    line3 = np.full(n, np.nan)
    
    # Initialize first three values
    if n >= 3:
        line1[0] = close[0]
        line2[1] = close[1]
        line3[2] = close[2]
    
    # Calculate 3-line break values
    for i in range(3, n):
        # Bullish: three consecutive closes above prior high
        if close[i] > line3[i-1] and close[i-1] > line2[i-1] and close[i-2] > line1[i-1]:
            line1[i] = line2[i-1]
            line2[i] = line3[i-1]
            line3[i] = close[i]
        # Bearish: three consecutive closes below prior low
        elif close[i] < line1[i-1] and close[i-1] < line2[i-1] and close[i-2] < line3[i-1]:
            line1[i] = close[i]
            line2[i] = line1[i-1]
            line3[i] = line2[i-1]
        # No change - carry forward
        else:
            line1[i] = line1[i-1]
            line2[i] = line2[i-1]
            line3[i] = line3[i-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    vol_exit = volume < 0.8 * vol_ma  # Exit when volume drops significantly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data is not available
        if np.isnan(line1[i]) or np.isnan(line2[i]) or np.isnan(line3[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish 3-line break with volume confirmation
            if (close[i] > line3[i] and close[i-1] > line2[i] and close[i-2] > line1[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish 3-line break with volume confirmation
            elif (close[i] < line1[i] and close[i-1] < line2[i] and close[i-2] < line3[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish 3-line break or volume drop
            if (close[i] < line1[i] and close[i-1] < line2[i] and close[i-2] < line3[i]) or vol_exit[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish 3-line break or volume drop
            if (close[i] > line3[i] and close[i-1] > line2[i] and close[i-2] > line1[i]) or vol_exit[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ThreeLineBreak_Trend_With_Volume"
timeframe = "12h"
leverage = 1.0