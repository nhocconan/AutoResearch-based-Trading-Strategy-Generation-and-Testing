#/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_V5
Hypothesis: Trade Donchian(20) breakouts on 4h with volume confirmation and 1d EMA trend filter.
Long when price breaks above upper band with volume spike and 1d uptrend; short when breaks below lower band with volume spike and 1d downtrend.
Uses volume > 2.0x 20-period average for strong breakout confirmation to reduce trade frequency.
Target: 50-120 total trades over 4 years (12-30/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, high volume threshold filters false breakouts.
This version adds a minimum holding period of 4 bars to reduce churn and trades below 400 total.
"""

name = "4h_Donchian_Breakout_Volume_Trend_Filter_V5"
timeframe = "4h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
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
    
    # Calculate volume spike (volume > 2.0x 20-period average for strict confirmation)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume spike AND 1d uptrend (price > EMA50)
            if close[i] > upper[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below lower band with volume spike AND 1d downtrend (price < EMA50)
            elif close[i] < lower[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Long exit: price breaks below lower band OR 1d trend turns down OR min 4 bars held
            if bars_since_entry >= 4 and (close[i] < lower[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            bars_since_entry += 1
            # Short exit: price breaks above upper band OR 1d trend turns up OR min 4 bars held
            if bars_since_entry >= 4 and (close[i] > upper[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals