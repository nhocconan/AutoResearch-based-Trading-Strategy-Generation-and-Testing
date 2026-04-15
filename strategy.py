#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter (EMA200) and 1d volume filter
# Uses 1h price action with 4h EMA200 for trend direction and 1d volume surge for confirmation.
# Designed to work in both bull and bear markets by only taking trades in the direction
# of the 4h trend, reducing whipsaws. Volume filter ensures momentum behind moves.
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    
    # Calculate EMA200 on 4h for trend
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period average volume on 1d
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size (20%)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price above 4h EMA200 + volume surge
        if (close[i] > ema200_4h_aligned[i] and
            volume[i] > 2.0 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below 4h EMA200 + volume surge
        elif (close[i] < ema200_4h_aligned[i] and
              volume[i] > 2.0 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < ema200_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema200_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_EMA200_Trend_Volume_Surge"
timeframe = "1h"
leverage = 1.0