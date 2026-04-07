#!/usr/bin/env python3
"""
6h_ichimoku_1d_filter_v1
Hypothesis: On 6-hour timeframe, use Ichimoku Cloud components with daily timeframe filter. 
Enter long when Tenkan-sen crosses above Kijun-sen AND price is above Kumo (cloud) from daily timeframe, 
short when Tenkan-sen crosses below Kijun-sen AND price is below Kumo from daily timeframe.
Exit when Tenkan-sen crosses back in opposite direction. 
Uses Ichimoku's built-in trend/filter properties to reduce whipsaws in both bull and bear markets.
Target: 50-150 trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate daily Ichimoku components
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(d_high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(d_low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(d_high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(d_low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(d_high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(d_low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For filtering, we need current cloud (already shifted in Ichimoku definition)
    # So we use Senkou Span A and B as is for current cloud
    kumoinfo_top = np.maximum(senkou_a, senkou_b)  # Upper cloud boundary
    kumoinfo_bottom = np.minimum(senkou_a, senkou_b)  # Lower cloud boundary
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    kumoinfo_top_aligned = align_htf_to_ltf(prices, df_1d, kumoinfo_top)
    kumoinfo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumoinfo_bottom)
    
    # Calculate 6h Ichimoku for entry signals (Tenkan/Kijun cross)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if daily data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumoinfo_top_aligned[i]) or np.isnan(kumoinfo_bottom_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if 6h Ichimoku not ready
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Tenkan crosses below Kijun
            if tenkan_6h[i] < kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Tenkan crosses above Kijun
            if tenkan_6h[i] > kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Tenkan crosses above Kijun AND price above daily cloud
            tenkan_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
            price_above_cloud = close[i] > kumoinfo_top_aligned[i]
            
            # Short entry: Tenkan crosses below Kijun AND price below daily cloud
            tenkan_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
            price_below_cloud = close[i] < kumoinfo_bottom_aligned[i]
            
            if tenkan_cross_up and price_above_cloud:
                position = 1
                signals[i] = 0.25
            elif tenkan_cross_down and price_below_cloud:
                position = -1
                signals[i] = -0.25
    
    return signals