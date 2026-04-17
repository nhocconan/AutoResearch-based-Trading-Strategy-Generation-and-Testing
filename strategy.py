#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v1
Ichimoku Cloud (10,26,52) with price breakout above/below cloud on 6h timeframe.
Uses daily close to confirm trend direction (above/below daily Kumo) for alignment.
Designed to work in both bull and bear markets by trading cloud breakouts with
trend alignment, reducing false reversals.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 6h Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 51:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
    senkou_b = (period52_high + period52_low) / 2
    
    # === 1d Ichimoku Cloud (for trend alignment) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(high_1d)
    
    # Tenkan-sen 1d
    period9_high_1d = np.full(n_1d, np.nan)
    period9_low_1d = np.full(n_1d, np.nan)
    for i in range(n_1d):
        if i >= 8:
            period9_high_1d[i] = np.max(high_1d[i-8:i+1])
            period9_low_1d[i] = np.min(low_1d[i-8:i+1])
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d
    period26_high_1d = np.full(n_1d, np.nan)
    period26_low_1d = np.full(n_1d, np.nan)
    for i in range(n_1d):
        if i >= 25:
            period26_high_1d[i] = np.max(high_1d[i-25:i+1])
            period26_low_1d[i] = np.min(low_1d[i-25:i+1])
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d
    period52_high_1d = np.full(n_1d, np.nan)
    period52_low_1d = np.full(n_1d, np.nan)
    for i in range(n_1d):
        if i >= 51:
            period52_high_1d[i] = np.max(high_1d[i-51:i+1])
            period52_low_1d[i] = np.min(low_1d[i-51:i+1])
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo (Cloud) top and bottom for 1d
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Kumo to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 52
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud top and bottom for current 6h bar
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above cloud AND price above 1d Kumo (bullish alignment)
            if (close[i] > cloud_top and 
                close[i] > kumo_top_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below cloud AND price below 1d Kumo (bearish alignment)
            elif (close[i] < cloud_bottom and 
                  close[i] < kumo_bottom_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price falls below cloud OR Tenkan crosses below Kijun
            if (close[i] < cloud_bottom or 
                tenkan[i] < kijun[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud OR Tenkan crosses above Kijun
            if (close[i] > cloud_top or 
                tenkan[i] > kijun[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0