#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Volume Filter + 1w EMA Trend
# Elder Ray measures bull/bear power via EMA13. In bull markets: buy when Bull Power > 0 and rising.
# In bear markets: sell when Bear Power < 0 and falling. Volume confirms conviction.
# 1w EMA40 acts as regime filter - only trade in direction of weekly trend.
# Works in both bull and bear by adapting to market regime via weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Elder Ray calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13_1d  # Higher highs relative to EMA13
    bear_power = low_1d - ema13_1d   # Lower lows relative to EMA13
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA40 on 1w for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align all indicators to 1d timeframe first (since Elder Ray is calculated on 1d)
    bull_power_aligned_1d = bull_power  # Already on 1d
    bear_power_aligned_1d = bear_power  # Already on 1d
    vol_avg_aligned_1d = vol_avg_1d     # Already on 1d
    ema40_1w_aligned_1d = align_htf_to_ltf(df_1d, df_1w, ema40_1w)
    
    # Now align from 1d to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_aligned_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_aligned_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_aligned_1d)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1d, ema40_1w_aligned_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema40_1w_aligned[i])):
            continue
        
        # Long entry: Bull Power > 0 and rising + volume above average + price above weekly EMA40
        if (bull_power_aligned[i] > 0 and 
            bull_power_aligned[i] > bull_power_aligned[i-1] and
            volume[i] > vol_avg_aligned[i] and
            close[i] > ema40_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 and falling + volume above average + price below weekly EMA40
        elif (bear_power_aligned[i] < 0 and 
              bear_power_aligned[i] < bear_power_aligned[i-1] and
              volume[i] > vol_avg_aligned[i] and
              close[i] < ema40_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or loss of momentum
        elif position == 1 and (bull_power_aligned[i] <= 0 or volume[i] < vol_avg_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] >= 0 or volume[i] < vol_avg_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0