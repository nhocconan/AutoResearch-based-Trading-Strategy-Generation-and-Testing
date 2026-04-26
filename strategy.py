#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm
Hypothesis: Ichimoku Kumo twist (Tenkan/Kijun cross) with 1w trend filter (price vs Senkou Span B) and volume confirmation.
Works in bull/bear markets: Kumo twist captures momentum shifts, 1w Senkou Span B filters trend direction, volume confirms conviction.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25) to minimize fee churn.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Load 1w data for trend filter (Senkou Span B)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Senkou Span B on 1w
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (period52_high_1w + period52_low_1w) / 2
    
    # Align 1w Senkou Span B to LTF (1w values available after the 1w bar closes)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of 52, 20)
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(senkou_b_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Kumo twist: Tenkan crosses Kijun
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        tenkan_cross_above = tenkan[i] > kijun[i] and tenkan_prev <= kijun_prev
        tenkan_cross_below = tenkan[i] < kijun[i] and tenkan_prev >= kijun_prev
        
        # Long logic: Tenkan crosses above Kijun + price > 1w Senkou Span B (uptrend) + volume spike
        if tenkan_cross_above and close[i] > senkou_b_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Tenkan crosses below Kijun + price < 1w Senkou Span B (downtrend) + volume spike
        elif tenkan_cross_below and close[i] < senkou_b_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit logic: Tenkan/Kijun cross in opposite direction
        elif position == 1 and tenkan_cross_below:
            signals[i] = 0.0
            position = 0
        elif position == -1 and tenkan_cross_above:
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

name = "6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0