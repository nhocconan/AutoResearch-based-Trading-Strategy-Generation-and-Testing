#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week Bollinger Band breakout and 1-day volume confirmation
# Strategy buys when price breaks above upper Bollinger Band (20, 2) on weekly chart with volume spike
# Sells when price breaks below lower Bollinger Band with volume spike
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Uses 12h timeframe for execution, 1w for trend, 1d for volume filter
# Target: 50-150 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly data
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb.values)
    
    # Load 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on daily data
    avg_volume_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(avg_volume_20d_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Bollinger Band + volume spike
        if (close[i] > upper_bb_aligned[i] and
            volume[i] > 1.5 * avg_volume_20d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Bollinger Band + volume spike
        elif (close[i] < lower_bb_aligned[i] and
              volume[i] > 1.5 * avg_volume_20d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to middle of Bollinger Bands (SMA)
        # We'll use a simple exit: reverse signal or time-based exit
        elif position == 1 and close[i] < (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Bollinger_Breakout_Volume"
timeframe = "12h"
leverage = 1.0