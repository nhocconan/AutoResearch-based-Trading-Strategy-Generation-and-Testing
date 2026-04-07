#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_v1
Hypothesis: Ichimoku Cloud provides a comprehensive trend/momentum/volatility system. 
On 6h timeframe, we use daily timeframe Ichimoku for trend direction and momentum,
while using 6h price action for entry timing. 
Long when: price above daily Kumo (cloud), Tenkan > Kijun (bullish TK cross), and price > Senkou Span A.
Short when: price below daily Kumo, Tenkan < Kijun (bearish TK cross), and price < Senkou Span B.
Exit when TK cross reverses or price re-enters the cloud.
Uses daily trend filter to avoid counter-trend trades and reduce whipsaw.
Works in both bull and bear markets by following the daily Ichimoku trend.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_1d_aligned[i]
        kijun = kijun_1d_aligned[i]
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if position == 1:  # Long position
            # Exit: TK cross turns bearish OR price re-enters cloud (below cloud top)
            if tenkan < kijun or close[i] < cloud_top:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price re-enters cloud (above cloud bottom)
            if tenkan > kijun or close[i] > cloud_bottom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above cloud, bullish TK cross, and above Senkou Span A
            if close[i] > cloud_top and tenkan > kijun and close[i] > senkou_a:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud, bearish TK cross, and below Senkou Span B
            elif close[i] < cloud_bottom and tenkan < kijun and close[i] < senkou_b:
                position = -1
                signals[i] = -0.25
    
    return signals