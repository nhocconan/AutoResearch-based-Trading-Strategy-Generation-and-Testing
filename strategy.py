#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v3
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation (1.5x). 
The Ichimoku system provides built-in trend, momentum, and support/resistance. 
Using 1d cloud as higher-timeframe trend filter ensures alignment with major trend, reducing false signals in choppy markets.
Volume confirmation ensures breakouts have conviction. 
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear via 1d cloud filter and volume confirmation for breakout validity.
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
    
    # Load 1d data ONCE before loop for cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h data
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
    
    # Load 1d OHLC for cloud calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku cloud (Senkou Span A and B)
    # Tenkan-sen 1d: (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2
    # Kijun-sen 1d: (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    # Senkou Span A 1d: (Tenkan + Kijun)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    # Senkou Span B 1d: (52-period high + 52-period low)/2
    senkou_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                   pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align 1d cloud to 6h (Senkou Span A and B)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # The cloud is between Senkou Span A and B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 9 for Tenkan, 20 for volume MA)
    start_idx = max(52, 26, 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku TK cross conditions with volume and cloud filter
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud AND volume spike
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and close[i] > cloud_top[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud AND volume spike
            elif tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and close[i] < cloud_bottom[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun OR price falls below cloud
            if tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v3"
timeframe = "6h"
leverage = 1.0