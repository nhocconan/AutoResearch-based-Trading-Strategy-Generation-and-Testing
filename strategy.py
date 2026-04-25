#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_Confluence
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with 1-day cloud filter for trend direction.
In bull markets (price above 1d cloud), TK cross up = long signal.
In bear markets (price below 1d cloud), TK cross down = short signal.
Uses 1d cloud as regime filter to avoid counter-trend whipsaws.
Target: 15-30 trades/year (~60-120 total over 4 years) by requiring confluence of TK cross and 1d cloud position.
Position size: 0.25 discrete levels to minimize fee churn.
Works in both bull and bear by only taking trades aligned with higher timeframe trend.
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
    
    # 1d data for Ichimoku cloud (Senkou Span A/B) - loaded ONCE
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Tenkan and Kijun for TK cross
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d Ichimoku (52) + 6h TK (26)
    start_idx = max(52, 26) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine 6h TK cross
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 0:
            # Look for entry signals with cloud filter
            # Long: TK cross up + price above 1d cloud (bullish alignment)
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below 1d cloud (bearish alignment)
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on TK cross down or price drops below cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on TK cross up or price rises above cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_Confluence"
timeframe = "6h"
leverage = 1.0