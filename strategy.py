# 6h_DailyPivot_Breakout_Confluence
# Hypothesis: 6-hour breakout above/below daily pivot levels with volume confirmation and time-of-day filter (UTC 8-20).
# Works in bull markets (breakouts above pivot + high volume) and bear markets (breakdowns below pivot + high volume).
# Uses daily pivot levels (calculated from prior day's OHLC) as support/resistance.
# Volume filter requires current volume > 1.5x 20-period average.
# Time filter restricts trading to active market hours (UTC 8-20) to avoid low-volume periods.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 6h timeframe (use previous day's levels)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: UTC 8-20 (active trading hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Time filter: only trade during active hours (UTC 8-20)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if close[i] > r1_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif close[i] < s1_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot
            if close[i] < pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot
            if close[i] > pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_Breakout_Confluence"
timeframe = "6h"
leverage = 1.0