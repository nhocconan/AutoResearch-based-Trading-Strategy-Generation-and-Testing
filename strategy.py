#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_ADX_Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat. 
Combined with 1-day ADX > 25 to confirm trending regime, this should capture trends while avoiding whipsaws in ranges.
Works in both bull (follows uptrends) and bear (follows downtrends) markets. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - adaptive to market noise
    def calculate_kama(close_array, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close_array)
        kama = np.full(n, np.nan)
        if n < er_length:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close_array, n=er_length))  # |close[i] - close[i-er_length]|
        volatility = np.sum(np.abs(np.diff(close_array)), axis=0)  # sum of |close[i] - close[i-1]| over er_length period
        
        # Handle first er_length elements
        for i in range(er_length, n):
            if i - er_length >= 0:
                change_val = np.abs(close_array[i] - close_array[i - er_length])
                # Calculate volatility from i-er_length+1 to i
                vol_sum = 0.0
                for j in range(i - er_length + 1, i + 1):
                    vol_sum += np.abs(close_array[j] - close_array[j-1])
                if vol_sum > 0:
                    er = change_val / vol_sum
                else:
                    er = 0
                sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
                if i == er_length:
                    kama[i] = close_array[i]
                else:
                    kama[i] = kama[i-1] + sc * (close_array[i] - kama[i-1])
            else:
                kama[i] = close_array[i]
        
        # Fill beginning with first close value
        for i in range(er_length):
            kama[i] = close_array[0] if len(close_array) > 0 else 0
            
        return kama
    
    # Calculate ADX (Average Directional Index) for trend strength
    def calculate_adx(high_array, low_array, close_array, period=14):
        n = len(close_array)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr1 = high_array[1:] - low_array[1:]
        tr2 = np.abs(high_array[1:] - close_array[:-1])
        tr3 = np.abs(low_array[1:] - close_array[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max(tr[:3]) if len(tr) >= 3 else np.mean(tr) if len(tr) > 0 else 0], tr])
        
        # Directional Movement
        dm_plus = np.where((high_array[1:] - high_array[:-1]) > (low_array[:-1] - low_array[1:]), 
                           np.maximum(high_array[1:] - high_array[:-1], 0), 0)
        dm_minus = np.where((low_array[:-1] - low_array[1:]) > (high_array[1:] - high_array[:-1]), 
                            np.maximum(low_array[:-1] - low_array[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        def smooth_values(arr, period):
            n = len(arr)
            smoothed = np.full(n, np.nan)
            if n < period:
                return smoothed
            # First value is simple average
            smoothed[period-1] = np.mean(arr[1:period+1]) if period+1 <= n else np.mean(arr[1:])
            # Subsequent values: Wilder smoothing
            for i in range(period, n):
                smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
            return smoothed
        
        atr = smooth_values(tr, period)
        dm_plus_smooth = smooth_values(dm_plus, period)
        dm_minus_smooth = smooth_values(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
        di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = smooth_values(dx, period)
        
        return adx
    
    # Calculate KAMA on close prices
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Get 1-day data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1-day data
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Using 1d as reference for alignment
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        adx_val = adx_1d_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long: price above KAMA AND ADX > 25 (strong trend)
            if close_val > kama_val and adx_val > 25:
                signals[i] = size
                position = 1
            # Short: price below KAMA AND ADX > 25 (strong trend)
            elif close_val < kama_val and adx_val > 25:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_1d_ADX_Filter"
timeframe = "4h"
leverage = 1.0