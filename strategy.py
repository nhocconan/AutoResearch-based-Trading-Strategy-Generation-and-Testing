#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: 6h Ichimoku TK cross filtered by 1d cloud direction (price above/below Kumo).
In bullish 1d trend (price > Senkou Span A&B): long on TK cross (Tenkan > Kijun).
In bearish 1d trend (price < Senkou Span A&B): short on TK cross (Tenkan < Kijun).
Requires cloud thickness > 0 to avoid whipsaw in sideways markets.
Uses discrete position sizing (0.25) to limit fee drag and drawdown.
Designed to work in both bull and bear markets by aligning with 1d Ichimoku trend.
Timeframe: 6h, uses 1d HTF for trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === 1d Ichimoku calculations ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === 6h price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) 
            or np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_1d_aligned[i]
        kijun = kijun_1d_aligned[i]
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        
        # Cloud thickness filter: avoid whipsaw in sideways markets
        cloud_thickness = abs(senkou_a - senkou_b)
        if cloud_thickness < 1e-8:  # Essentially flat cloud
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend direction from cloud
        bullish_cloud = price > senkou_a and price > senkou_b
        bearish_cloud = price < senkou_a and price < senkou_b
        
        if position == 0:
            # Bullish TK cross in bullish cloud -> long
            if bullish_cloud and tenkan > kijun:
                signals[i] = 0.25
                position = 1
            # Bearish TK cross in bearish cloud -> short
            elif bearish_cloud and tenkan < kijun:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions for long
            # TK cross reverse (Tenkan < Kijun) OR price breaks below cloud
            if tenkan < kijun or price < senkou_a or price < senkou_b:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            # TK cross reverse (Tenkan > Kijun) OR price breaks above cloud
            if tenkan > kijun or price > senkou_a or price > senkou_b:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0