#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted Breakout with 12h Volume Trend Filter
# Uses 4h price breaking above/below 12h VWAP with 4h volume > 1.5x 20-bar median.
# Trend filter: 12h VWAP slope > 0 for longs, < 0 for shorts.
# Works in bull markets (breakouts above rising VWAP) and bear markets (breakouts below falling VWAP).
# Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for VWAP and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP (typical price * volume) / volume
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_numerator = np.cumsum(typical_price_12h * volume_12h)
    vwap_denominator = np.cumsum(volume_12h)
    vwap_12h = vwap_numerator / (vwap_denominator + 1e-10)
    
    # Calculate 12h VWAP slope (5-period change)
    vwap_slope = vwap_12h - np.roll(vwap_12h, 5)
    vwap_slope[:5] = 0
    
    # Align 12h VWAP and slope to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    vwap_slope_aligned = align_htf_to_ltf(prices, df_12h, vwap_slope)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(vwap_slope_aligned[i])):
            continue
        
        # Long entry: price above 12h VWAP + volume confirmation + rising VWAP slope
        if (close[i] > vwap_12h_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            vwap_slope_aligned[i] > 0 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below 12h VWAP + volume confirmation + falling VWAP slope
        elif (close[i] < vwap_12h_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              vwap_slope_aligned[i] < 0 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse price relative to VWAP or volume drops below average
        elif position == 1 and (close[i] < vwap_12h_aligned[i] or 
                                volume[i] < 0.5 * np.median(volume[max(0, i-20):i+1])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > vwap_12h_aligned[i] or 
                                 volume[i] < 0.5 * np.median(volume[max(0, i-20):i+1])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_VWAP_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0