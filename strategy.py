# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1d_ichimoku_cloud_filter
Hypothesis: Use Ichimoku Cloud on daily timeframe to identify trend direction and support/resistance zones.
Enter on 6h when Tenkan-sen crosses Kijun-sen in direction of cloud color (bullish/bearish).
Exit when price exits the cloud or Tenkan/Kijun cross reverses.
Ichimoku components: Tenkan (9-period), Kijun (26-period), Senkou A/B (26/52-period).
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in trending markets by following cloud direction and avoids false signals in ranging markets.
"""

name = "6h_1d_ichimoku_cloud_filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily timeframe
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Shift Senkou spans forward by 26 periods (as per Ichimoku definition)
    senkou_a_shifted = np.roll(senkou_a_1d, 26)
    senkou_b_shifted = np.roll(senkou_b_1d, 26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and boundaries
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        green_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        red_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long entry: Tenkan crosses above Kijun in bullish cloud
        if (tenkan_aligned[i] > kijun_aligned[i] and 
            tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # crossed above
            green_cloud and 
            close[i] > cloud_bottom and  # price above cloud bottom
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: Tenkan crosses below Kijun in bearish cloud
        elif (tenkan_aligned[i] < kijun_aligned[i] and 
              tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # crossed below
              red_cloud and 
              close[i] < cloud_top and  # price below cloud top
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price exits cloud or Tenkan/Kijun cross reverses
        elif position == 1 and (close[i] < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > cloud_top or tenkan_aligned[i] > kijun_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals