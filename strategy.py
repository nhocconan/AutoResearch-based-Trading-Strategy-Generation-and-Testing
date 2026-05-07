#!/usr/bin/env python3
"""
6H_Ichimoku_Kijun_Tenkan_Cross_1D_Cloud_Filter
Hypothesis: Use Ichimoku Cloud on 1D for trend direction and support/resistance, with TK cross on 6H for entry timing. 
The cloud acts as dynamic support/resistance and trend filter, reducing false signals in sideways markets. 
TK cross provides timely entries within the trend. Works in bull markets (buy when price above cloud + TK cross up) 
and bear markets (sell when price below cloud + TK cross down). Targets 12-37 trades/year on 6H timeframe.
"""
name = "6H_Ichimoku_Kijun_Tenkan_Cross_1D_Cloud_Filter"
timeframe = "6h"
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
    
    # Get 1D data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6H timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price above cloud + TK cross bullish (Tenkan crosses above Kijun)
            if (close[i] > upper_cloud and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bearish (Tenkan crosses below Kijun)
            elif (close[i] < lower_cloud and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below cloud OR TK cross bearish
            if (close[i] < lower_cloud or 
                (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above cloud OR TK cross bullish
            if (close[i] > upper_cloud or 
                (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals