#!/usr/bin/env python3
# 6h_Ichimoku_TenkanKijun_Cross_1dTrend
# Hypothesis: Use Ichimoku Cloud on 6h for trend confirmation and entry signals,
# filtered by 1d Kumo (cloud) direction. Enter long when Tenkan-sen crosses above Kijun-sen
# and price is above 1d Kumo, short when Tenkan crosses below Kijun and price below 1d Kumo.
# Exit on opposite cross or Kumo failure. Designed for low frequency (15-35 trades/year)
# to avoid fee drag. Ichimoku works in trending markets (bull/bear) and avoids sideways chop
# via cloud filter, making it robust across regimes.

name = "6h_Ichimoku_TenkanKijun_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """
    Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B.
    Returns tenkan, kijun, senkou_a, senkou_b arrays.
    """
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9 = 9
    for i in range(n):
        if i >= period9 - 1:
            high9 = np.max(high[i - period9 + 1:i + 1])
            low9 = np.min(low[i - period9 + 1:i + 1])
            tenkan[i] = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26 = 26
    for i in range(n):
        if i >= period26 - 1:
            high26 = np.max(high[i - period26 + 1:i + 1])
            low26 = np.min(low[i - period26 + 1:i + 1])
            kijun[i] = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52 = 52
    for i in range(n):
        if i >= period52 - 1:
            high52 = np.max(high[i - period52 + 1:i + 1])
            low52 = np.min(low[i - period52 + 1:i + 1])
            senkou_b[i] = (high52 + low52) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Kumo (cloud) filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Calculate 1d Ichimoku components for Kumo filter
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, df_1d['close'].values)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align daily Kumo to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure Ichimoku is stable (needs 52 periods for Senkou B)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signals: Tenkan/Kijun cross
        tenkan_cross_above = tenkan[i] > kijun[i]
        tenkan_cross_below = tenkan[i] < kijun[i]
        
        # Kumo filter: price above/below cloud
        price_above_kumo = close[i] > kumo_top_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_aligned[i]
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above 1d Kumo
            if tenkan_cross_above and price_above_kumo:
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below 1d Kumo
            elif tenkan_cross_below and price_below_kumo:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price falls below Kumo
            if tenkan_cross_below or not price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above Kumo
            if tenkan_cross_above or not price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals