#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA trend filter
# Breakouts above 20-period high or below 20-period low are traded only when:
#   - Volume > 1.5x median volume of last 20 bars (confirmation)
#   - Price is above/below 100-period EMA on 1d (trend filter)
# Works in bull markets (breakouts above EMA) and bear markets (breakouts below EMA)
# Target: 50-150 total trades over 4 years. Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 100-period EMA on 1d
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_100_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above 20-period high + volume confirmation + price > 1d EMA100
        if (close[i] > high_20[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_100_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 20-period low + volume confirmation + price < 1d EMA100
        elif (close[i] < low_20[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_100_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < low_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_20[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_EMA100"
timeframe = "4h"
leverage = 1.0