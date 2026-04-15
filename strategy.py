#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Volume Confirmation and 1D Trend Filter
# Uses 12-hour Donchian(20) breakouts for entries, confirmed by volume spike (>1.5x median)
# and filtered by 1D EMA50 trend direction (price > EMA50 for longs, < EMA50 for shorts).
# Works in bull markets (breakouts up in uptrend) and bear markets (breakouts down in downtrend).
# Target: 50-150 total trades over 4 years. Timeframe: 12h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + uptrend (price > EMA50)
        if (close[i] > highest_20[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + downtrend (price < EMA50)
        elif (close[i] < lowest_20[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout
        elif position == 1 and close[i] < lowest_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_20[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_EMA50"
timeframe = "12h"
leverage = 1.0