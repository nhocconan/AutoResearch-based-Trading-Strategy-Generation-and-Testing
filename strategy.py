#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter
Hypothesis: Ichimoku Tenkan-Kijun cross combined with 1d Kumo cloud filter captures medium-term momentum with institutional filtering. 
Tenkan (9) and Kijun (26) lines cross signals momentum shifts. 
Only take longs when price above Kumo cloud (bullish bias) and shorts when below (bearish bias).
Works in bull markets via continuation signals and bear markets via counter-trend reversals at cloud boundaries.
Target: 15-30 trades/year per symbol.
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
    
    # Get 1d data for Kumo cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
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
    
    # Get 1d Kumo cloud data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku components for cloud
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 1d cloud to 6h timeframe (Kumo cloud is plotted 26 periods ahead)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - need 52 for Senkou B calculation
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud (bullish bias)
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and 
                close[i] > cloud_top[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud (bearish bias)
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and 
                  close[i] < cloud_bottom[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0