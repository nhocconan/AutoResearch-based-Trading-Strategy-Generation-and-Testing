#!/usr/bin/env python3
"""
6h_ElderRay_1d_Trend_Filter
Hypothesis: Trade Elder Ray power signals (bull power > 0 for long, bear power < 0 for short)
filtered by 1d EMA50 trend. Works in bull/bear markets: uses 1d trend to avoid counter-trend trades,
Elder Ray captures momentum shifts. Target: 60-120 total trades over 4 years (15-30/year)
with position size 0.25. Uses EMA13 for power calculation to balance responsiveness and noise.
"""

name = "6h_ElderRay_1d_Trend_Filter"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (using 13-period EMA of close)
    ema13 = ema(close, 13)
    
    # Calculate Elder Ray power
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bull power > 0 AND 1d uptrend (close > EMA50)
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power < 0 AND 1d downtrend (close < EMA50)
            elif bear_power[i] < 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power <= 0 OR 1d trend turns down
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power >= 0 OR 1d trend turns up
            if bear_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals