#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud color filter from 1d timeframe. Uses 1d cloud color (green/red) as trend filter and 6h TK cross for entry timing. Designed for medium-frequency trading (15-35 trades/year) with built-in trend alignment to work in both bull and bear markets by only taking trades in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku for TK cross
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Cloud filter: price above cloud (bullish) or below cloud (bearish)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    price_above_cloud = close > cloud_top_1d
    price_below_cloud = close < cloud_bottom_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) and 6h TK calculation (26)
    start_idx = max(52, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross above AND price above 1d cloud (bullish alignment)
            long_signal = tk_cross_above[i] and price_above_cloud[i]
            
            # Short: TK cross below AND price below 1d cloud (bearish alignment)
            short_signal = tk_cross_below[i] and price_below_cloud[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long until TK cross below (exit signal)
            signals[i] = 0.25
            if tk_cross_below[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short until TK cross above (exit signal)
            signals[i] = -0.25
            if tk_cross_above[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0