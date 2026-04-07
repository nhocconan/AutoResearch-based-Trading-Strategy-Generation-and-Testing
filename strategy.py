#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1w_trend_v1
Hypothesis: Ichimoku Cloud system with Tenkan/Kijun cross and cloud color filter from weekly timeframe.
In bull markets (price above cloud), take longs on TK cross up; in bear markets (price below cloud), 
take shorts on TK cross down. Uses weekly trend filter to avoid counter-trend trades.
Works in both bull and bear markets by adapting to trend via cloud position.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods ago
    # Not used for signals but needed for cloud calculation
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Determine cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Determine if price is above or below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # TK cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 1:  # Long position
            # Exit: price drops below cloud or TK cross down
            if close[i] < cloud_bottom[i] or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price rises above cloud or TK cross up
            if close[i] > cloud_top[i] or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TK cross up AND price above cloud (bullish alignment)
            if tk_cross_up and price_above_cloud[i]:
                position = 1
                signals[i] = 0.25
            # Short: TK cross down AND price below cloud (bearish alignment)
            elif tk_cross_down and price_below_cloud[i]:
                position = -1
                signals[i] = -0.25
    
    return signals