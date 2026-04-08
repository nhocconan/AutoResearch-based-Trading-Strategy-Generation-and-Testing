#!/usr/bin/env python3
# 6h_12h_ichimoku_cloud_trend_v1
# Hypothesis: 6h Ichimoku cloud with TK cross and 12h cloud filter for trend following.
# Long: price > 12h cloud AND Tenkan > Kijun (bullish TK cross)
# Short: price < 12h cloud AND Tenkan < Kijun (bearish TK cross)
# Exit: TK cross reverses or price touches opposite cloud boundary
# Uses 6h primary timeframe with 12h HTF for cloud to reduce whipsaw.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(9, n):
        period9_high[i] = np.max(high[i-9:i+1])
        period9_low[i] = np.min(low[i-9:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(26, n):
        period26_high[i] = np.max(high[i-26:i+1])
        period26_low[i] = np.min(low[i-26:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(52, n):
        period52_high[i] = np.max(high[i-52:i+1])
        period52_low[i] = np.min(low[i-52:i+1])
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 12h data for cloud filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate 12h Ichimoku cloud
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Tenkan-sen (9-period)
    tenkan_12h = np.full(len(df_12h), np.nan)
    period9_high_12h = np.full(len(df_12h), np.nan)
    period9_low_12h = np.full(len(df_12h), np.nan)
    for i in range(9, len(df_12h)):
        period9_high_12h[i] = np.max(high_12h[i-9:i+1])
        period9_low_12h[i] = np.min(low_12h[i-9:i+1])
    tenkan_12h = (period9_high_12h + period9_low_12h) / 2
    
    # Kijun-sen (26-period)
    kijun_12h = np.full(len(df_12h), np.nan)
    period26_high_12h = np.full(len(df_12h), np.nan)
    period26_low_12h = np.full(len(df_12h), np.nan)
    for i in range(26, len(df_12h)):
        period26_high_12h[i] = np.max(high_12h[i-26:i+1])
        period26_low_12h[i] = np.min(low_12h[i-26:i+1])
    kijun_12h = (period26_high_12h + period26_low_12h) / 2
    
    # Senkou Span A
    senkou_a_12h = (tenkan_12h + kijun_12h) / 2
    
    # Senkou Span B (52-period)
    senkou_b_12h = np.full(len(df_12h), np.nan)
    period52_high_12h = np.full(len(df_12h), np.nan)
    period52_low_12h = np.full(len(df_12h), np.nan)
    for i in range(52, len(df_12h)):
        period52_high_12h[i] = np.max(high_12h[i-52:i+1])
        period52_low_12h[i] = np.min(low_12h[i-52:i+1])
    senkou_b_12h = (period52_high_12h + period52_low_12h) / 2
    
    # Align 12h cloud to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a_12h)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b_12h)
    
    # Determine cloud boundaries (max/min of Senkou A and B)
    upper_cloud_12h = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud_12h = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any values are NaN
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(upper_cloud_12h[i]) or np.isnan(lower_cloud_12h[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        upper_cloud = upper_cloud_12h[i]
        lower_cloud = lower_cloud_12h[i]
        
        if position == 1:  # Long position
            # Exit if TK cross turns bearish or price drops below cloud
            if tenkan_val < kijun_val or price < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if TK cross turns bullish or price rises above cloud
            if tenkan_val > kijun_val or price > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long if price above cloud and bullish TK cross
            if price > upper_cloud and tenkan_val > kijun_val:
                position = 1
                signals[i] = 0.25
            # Enter short if price below cloud and bearish TK cross
            elif price < lower_cloud and tenkan_val < kijun_val:
                position = -1
                signals[i] = -0.25
    
    return signals