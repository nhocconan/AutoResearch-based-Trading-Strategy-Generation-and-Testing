#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy using Donchian channel (20-period) breakout with 4-hour EMA20 trend filter and volume confirmation (>1.5x 20-period average volume).
# Donchian breakouts capture momentum bursts; EMA20 filter ensures trades align with intermediate trend.
# Volume confirmation filters false breakouts. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (upward breakouts) and bear markets (downward breakdowns) by taking both long and short signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) - calculate directly on 4h data
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above upper Donchian band with uptrend and volume
        if (close[i] > high_roll_max[i] and 
            close[i] > ema20[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below lower Donchian band with downtrend and volume
        elif (close[i] < low_roll_min[i] and 
              close[i] < ema20[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or trend reversal
        elif position == 1 and (close[i] < ema20[i] or close[i] < low_roll_min[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > ema20[i] or close[i] > high_roll_max[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA20_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0