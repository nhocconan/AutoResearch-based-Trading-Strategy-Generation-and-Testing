#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter and volume confirmation.
Enters long when Tenkan crosses above Kijun, price is above 1d cloud, and volume spikes.
Enters short when Tenkan crosses below Kijun, price is below 1d cloud, and volume spikes.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Ichimoku cloud acts as dynamic support/resistance, filtering false crosses in sideways markets.
Works in both bull and bear markets by requiring alignment with higher timeframe trend (cloud position).
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
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Load 1d data for cloud filter (higher timeframe trend)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku cloud (Senkou Span A and B)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen 1d (9-period)
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d (26-period)
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d (52-period)
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d cloud components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine 1d cloud boundaries (top and bottom of cloud)
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_cross_bull = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_cross_bear = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
        
        # Long logic: bullish TK cross + price above 1d cloud + volume spike
        if tk_cross_bull and (close[i] > cloud_top_1d[i]) and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish TK cross + price below 1d cloud + volume spike
        elif tk_cross_bear and (close[i] < cloud_bottom_1d[i]) and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses opposite cloud boundary
        elif position == 1 and close[i] < cloud_bottom_1d[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > cloud_top_1d[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0