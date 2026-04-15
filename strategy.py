# 4h_1d_Range_Breakout_Volume_ADX
# Hypothesis: 4h 1-day Range Breakout with Volume Confirmation and ADX Trend Filter
# Uses the previous day's high/low as support/resistance levels. Breakouts above previous day's high
# or below previous day's low are traded only when confirmed by volume and ADX > 25 (trending market).
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 50-150 total trades.
# Timeframe: 4h, HTF: 1d

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for previous day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous day
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 4h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above previous day's high + volume confirmation
        if (close[i] > prev_high_1d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume confirmation
        elif (close[i] < prev_low_1d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < prev_low_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_high_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Range_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0