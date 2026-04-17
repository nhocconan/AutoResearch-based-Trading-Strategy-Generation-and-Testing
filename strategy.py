#!/usr/bin/env python3
"""
4h_4h1w_Donchian_Breakout_Volume_Trend_v1
Hypothesis: On 4h timeframe, buy when price breaks above 4h Donchian upper (20) with volume spike (>1.5x median) and 1w EMA trend alignment, sell when breaks below Donchian lower. Uses volume and trend filters to avoid false breakouts. Designed for low trade frequency (19-50/year) to minimize fee drag and work in both bull and bear markets via strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    return smooth_wilder(tr, period)

def calculate_ema(arr, period):
    """Calculate EMA with proper handling of initial values"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    ema = np.full_like(arr, np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        ema[i] = (arr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Indicators ===
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # 4h volume median (50-period)
    vol_median_4h = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # === 1w Trend Filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = calculate_ema(close_1w, 200)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_median_4h[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current volume > 1.5x median volume
        vol_spike = volume[i] > 1.5 * vol_median_4h[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and above weekly EMA200
            if close[i] > donchian_upper[i] and vol_spike and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian lower with volume spike and below weekly EMA200
            elif close[i] < donchian_lower[i] and vol_spike and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below Donchian lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above Donchian upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4h1w_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0