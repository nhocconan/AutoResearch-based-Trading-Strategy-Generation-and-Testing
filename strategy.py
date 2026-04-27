#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter
Hypothesis: Uses Ichimoku Tenkan/Kijun cross on 6h with 1d cloud filter to capture trend changes. TK cross provides timely entries, while 1d cloud (Senkou Span A/B) acts as a strong trend filter. Works in bull/bear by only taking trades in direction of higher timeframe cloud color. Targets 15-25 trades/year on 6h to minimize fee drag.
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
    
    # Get 1d data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate TK cross on 6d using actual 6h data (more responsive)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_fast = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_fast = (period26_high_6h + period26_low_6h) / 2
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red if Senkou A < Senkou B (bearish)
    cloud_green = senkou_a_6h > senkou_b_6h
    cloud_red = senkou_a_6h < senkou_b_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(52, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_6h_fast[i]) or np.isnan(kijun_6h_fast[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_6h_fast[i]
        kijun_val = kijun_6h_fast[i]
        tenkan_prev = tenkan_6h_fast[i-1]
        kijun_prev = kijun_6h_fast[i-1]
        
        # TK cross signals
        tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_val > kijun_val)
        tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_val < kijun_val)
        
        if position == 0:
            # Long: TK cross up in bullish cloud (Senkou A > Senkou B)
            if tk_cross_up and cloud_green[i]:
                signals[i] = size
                position = 1
            # Short: TK cross down in bearish cloud (Senkou A < Senkou B)
            elif tk_cross_down and cloud_red[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross down or cloud turns bearish
            if tk_cross_down or cloud_red[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up or cloud turns bullish
            if tk_cross_up or cloud_green[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0