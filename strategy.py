#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_Filter
Hypothesis: Trade daily Donchian(20) breakouts with weekly trend filter and volume confirmation.
Long when price breaks above upper band with volume spike and weekly uptrend; short when breaks below lower band with volume spike and weekly downtrend.
Uses weekly EMA50 for trend filter and volume > 1.8x 20-day average to reduce false signals.
Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trends.
Works in bull/bear: weekly trend filter avoids counter-trend trades, high volume threshold filters false breakouts.
"""

name = "1d_Donchian_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_weekly = ema(close_weekly, 50)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    
    # Calculate volume spike (volume > 1.8x 20-period average for confirmation)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume spike AND weekly uptrend (price > EMA50)
            if close[i] > upper[i] and volume_spike[i] and close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike AND weekly downtrend (price < EMA50)
            elif close[i] < lower[i] and volume_spike[i] and close[i] < ema50_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band OR weekly trend turns down
            if close[i] < lower[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band OR weekly trend turns up
            if close[i] > upper[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals