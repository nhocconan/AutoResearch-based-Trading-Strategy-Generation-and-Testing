#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_weekly_pivot_breakout_v1
# Uses weekly pivot points (from 1d data aggregated to weekly) to identify key support/resistance.
# In bull markets: buy breakouts above weekly R1 with volume confirmation.
# In bear markets: sell breakdowns below weekly S1 with volume confirmation.
# Uses 6h timeframe for entries, weekly pivot for direction, volume filter to avoid false breakouts.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years).
name = "6h_1d_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data to calculate weekly pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data (simplified: use Friday's close as weekly close)
    # For pivot calculation, we need weekly high, low, close
    # Approximate weekly high as max of last 5 days, weekly low as min of last 5 days
    # Weekly close as Friday's close (assuming 5-day week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).shift(4).values  # Friday's close (4 days ago from current day)
    
    # Weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe
    r1_level = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_level = align_htf_to_ltf(prices, df_1d, weekly_s1)
    pivot_level = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(r1_level[i]) or np.isnan(s1_level[i]) or np.isnan(pivot_level[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly R1
        if close[i] > r1_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly S1
        elif close[i] < s1_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly pivot
        elif position == 1 and close[i] < pivot_level[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot_level[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals