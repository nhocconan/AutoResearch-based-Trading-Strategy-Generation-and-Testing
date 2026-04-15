#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + EMA200 filter
# Uses 4h Donchian channel breakouts with volume > 1.5x 20-period median and price > EMA200 for long,
# price < EMA200 for short. Works in trending markets with volume confirmation to avoid false breakouts.
# EMA200 filter ensures we trade with the higher timeframe trend, reducing whipsaw in ranging markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA200 filter (4h timeframe)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: > 1.5x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema200[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_median[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + price > EMA200
        if (close[i] > highest_high[i] and
            volume[i] > 1.5 * vol_median[i] and
            close[i] > ema200[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + price < EMA200
        elif (close[i] < lowest_low[i] and
              volume[i] > 1.5 * vol_median[i] and
              close[i] < ema200[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout
        elif position == 1 and close[i] < lowest_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_high[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA200"
timeframe = "4h"
leverage = 1.0