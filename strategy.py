#!/usr/bin/env python3
"""
6h_SupportResistance_WeeklyTrend_Filter
Hypothesis: Trade breakouts of 6h support/resistance levels (swing high/low) with volume confirmation, filtered by weekly trend direction (EMA100). 
Long when price breaks above resistance with volume spike and weekly uptrend; short when breaks below support with volume spike and weekly downtrend.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""

name = "6h_SupportResistance_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA100 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema100_weekly = ema(close_weekly, 100)
    ema100_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema100_weekly)
    
    # Calculate swing points for support/resistance (using 5-period lookback)
    # Resistance: local high where high[i] is highest in window
    # Support: local low where low[i] is lowest in window
    window = 5
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    
    for i in range(window, n - window):
        # Check if current high is the maximum in the window
        if high[i] == np.max(high[i-window:i+window+1]):
            resistance[i] = high[i]
        # Check if current low is the minimum in the window
        if low[i] == np.min(low[i-window:i+window+1]):
            support[i] = low[i]
    
    # Forward fill the last valid support/resistance levels
    for i in range(1, n):
        if np.isnan(resistance[i]):
            resistance[i] = resistance[i-1]
        if np.isnan(support[i]):
            support[i] = support[i-1]
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema100_weekly_aligned[i]) or np.isnan(resistance[i]) or np.isnan(support[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above resistance with volume spike AND weekly uptrend (close > EMA100)
            if close[i] > resistance[i] and volume_spike[i] and close[i] > ema100_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below support with volume spike AND weekly downtrend (close < EMA100)
            elif close[i] < support[i] and volume_spike[i] and close[i] < ema100_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below support OR weekly trend turns down
            if close[i] < support[i] or close[i] < ema100_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above resistance OR weekly trend turns up
            if close[i] > resistance[i] or close[i] > ema100_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals