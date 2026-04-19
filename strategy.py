#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_ADX_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction, 
filtered by weekly ADX > 25 to ensure strong trending markets (avoiding chop/range).
Works in both bull and bear markets by following the trend direction.
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
"""

name = "1d_KAMA_Trend_With_ADX_Filter"
timeframe = "1d"
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
    
    # KAMA parameters
    kama_fast = 2
    kama_slow = 30
    
    # Calculate KAMA
    def calculate_kama(close, fast, slow):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, 10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if False else np.zeros_like(close)
        # Calculate volatility properly
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        # Vectorized volatility calculation
        volatility = np.zeros_like(close)
        volatility[1:] = np.cumsum(np.abs(np.diff(close)))
        # Subtract to get rolling sum
        volatility = np.zeros_like(close)
        for i in range(10, len(close)):
            volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
        er = np.zeros_like(close)
        for i in range(10, len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    kama_1d = calculate_kama(df_1d['close'].values, kama_fast, kama_slow)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Weekly ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + (1.0/period) * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 10)
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            if close[i] > kama_1d_aligned[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama_1d_aligned[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if close[i] < kama_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals