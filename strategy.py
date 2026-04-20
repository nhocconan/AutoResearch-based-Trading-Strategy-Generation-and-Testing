#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_TrendFollowing
# Hypothesis: Use 1d weekly pivot points (calculated from prior week's OHLC) to identify trend direction on 6h timeframe.
# In bull markets: price above weekly pivot = long bias, buy pullbacks to support.
# In bear markets: price below weekly pivot = short bias, sell rallies to resistance.
# Uses volume confirmation to filter breakouts and avoid false signals.
# Weekly pivots are more significant than daily pivots as they reflect multi-day structure.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

name = "6h_1d_WeeklyPivot_TrendFollowing"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data
    # Group by week (Monday to Friday) and calculate pivot from weekly OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Create DataFrame for weekly grouping
    df_weekly = pd.DataFrame({
        'high': high_1d,
        'low': low_1d,
        'close': close_1d
    }, index=pd.to_datetime(df_1d.index))
    
    # Resample to weekly (Monday start)
    weekly = df_weekly.resample('W-MON').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (weekly['high'] + weekly['low'] + weekly['close']) / 3
    
    # Forward fill weekly pivot to daily frequency
    weekly_pivot_daily = weekly_pivot.reindex(df_1d.index, method='ffill')
    weekly_pivot_values = weekly_pivot_daily.values
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot with volume surge
            if (close[i] > pivot_aligned[i] * 1.002 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot with volume surge
            elif (close[i] < pivot_aligned[i] * 0.998 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot
            if close[i] < pivot_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot
            if close[i] > pivot_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals