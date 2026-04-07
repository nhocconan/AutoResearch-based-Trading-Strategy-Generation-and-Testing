#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Hypothesis: Donchian(20) breakouts in direction of 1d EMA(50) trend with volume > 1.5x average
# capture high-probability momentum moves. Works in bull (breakouts continue) and bear
# (breakdowns continue) markets. Target: 20-40 trades/year (80-160 total over 4 years)
# to minimize fee drag while maintaining edge.

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(20) channels on 4h
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    high_20 = rolling_max(high, 20)
    low_20 = rolling_min(low, 20)
    
    # Volume average (20-period)
    def rolling_mean(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < window:
            return res
        # Calculate first value
        res[window-1] = np.mean(arr[0:window])
        # Rolling update
        for i in range(window, len(arr)):
            res[i] = res[i-1] + (arr[i] - arr[i-window]) / window
        return res
    
    vol_avg = rolling_mean(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if close[i] < low_20[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if close[i] > high_20[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: price breaks above Donchian high in uptrend
            if close[i] > high_20[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low in downtrend
            elif close[i] < low_20[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals