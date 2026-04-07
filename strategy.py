#!/usr/bin/env python3
"""
6h_ichimoku_cloud_trend_v1
Hypothesis: Ichimoku Cloud provides reliable trend direction and support/resistance levels. 
In trending markets (price above/below cloud with TK cross), price tends to continue in the direction of the cloud.
Using 1d Ichimoku for trend filter and 6s for entry timing reduces whipsaw. Works in both bull and bear markets 
by following the higher timeframe trend and using cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): Close shifted -22 periods
    chikou_span = pd.Series(close).shift(-kijun + senkou)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close, 9, 26, 52)
    
    # Get 1d Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d, 9, 26, 52)
    
    # Align 1d Ichimoku to 6s timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        cloud_green = senkou_a[i] > senkou_b[i]
        cloud_red = senkou_a[i] < senkou_b[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        # TK Cross
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] if i > 0 else False
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] if i > 0 else False
        
        # 1d trend filter (price relative to 1d cloud)
        price_above_1d_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_1d_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price falls below cloud or TK cross down
            if price_below_cloud or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud or TK cross up
            if price_above_cloud or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above green cloud, TK cross up, and 1d trend bullish
            if price_above_cloud and cloud_green and tk_cross_up and price_above_1d_cloud:
                position = 1
                signals[i] = 0.25
            # Short: price below red cloud, TK cross down, and 1d trend bearish
            elif price_below_cloud and cloud_red and tk_cross_down and price_below_1d_cloud:
                position = -1
                signals[i] = -0.25
    
    return signals