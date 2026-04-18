#!/usr/bin/env python3
"""
6h_Ichimoku_TKCross_1dTrendFilter
Hypothesis: Ichimoku Tenkan/Kijun cross on 6h acts as momentum trigger, but only when aligned with 1d trend (price above/below Kumo). Reduces whipsaw in sideways markets. Works in bull (TK cross up + price above cloud) and bear (TK cross down + price below cloud). Limits trades via trend filter to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Get 1d data for trend filter (price vs Kumo)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo (cloud) boundaries on 1d: Senkou Span A and B
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if any values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kumo_top_1d_aligned[i]) or 
            np.isnan(kumo_bottom_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # 1d trend filter: price above/below Kumo
        price_above_kumo = close[i] > kumo_top_1d_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_1d_aligned[i]
        
        if position == 0:
            # Long: TK cross bullish + price above 1d Kumo + volume
            if tenkan[i] > kijun[i] and price_above_kumo and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below 1d Kumo + volume
            elif tenkan[i] < kijun[i] and price_below_kumo and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below Kumo bottom
            if tenkan[i] < kijun[i] or close[i] < kumo_bottom_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above Kumo top
            if tenkan[i] > kijun[i] or close[i] > kumo_top_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_1dTrendFilter"
timeframe = "6h"
leverage = 1.0