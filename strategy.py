#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_With_Regime_Filter
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) identifies institutional buying/selling pressure.
Combined with 1-day trend filter (price > EMA50 for long, < EMA50 for short) to avoid counter-trend trades.
Volume confirmation ensures institutional participation. Designed for 6h timeframe to target 50-150 total trades over 4 years.
Works in bull/bear via trend filter - only takes longs in uptrend, shorts in downtrend.
"""

name = "6h_ElderRay_BullBearPower_With_Regime_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    def ema(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        multiplier = 2.0 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
            else:
                result[i] = np.nan
        return result
    
    # EMA50 for 1-day trend filter
    def ema_long(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        multiplier = 2.0 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
            else:
                result[i] = np.nan
        return result
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1-day close for trend filter
    ema50_1d = ema_long(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema13 = ema(close, 13)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only long in uptrend (price > EMA50), only short in downtrend (price < EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + volume + uptrend
            if (bull_power[i] > 0 and 
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + volume + downtrend
            elif (bear_power[i] < 0 and 
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bull Power turns negative or trend changes to downtrend
            if (bull_power[i] <= 0) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bear Power turns positive or trend changes to uptrend
            if (bear_power[i] >= 0) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals