#!/usr/bin/env python3
"""
6h_12h_1d_ichimoku_cloud_trend
Hypothesis: 6-hour strategy using Ichimoku cloud from daily timeframe for trend direction and entry signals.
Converts Tenkan/Kijun cross and price position relative to cloud into signals. Works in bull/bear markets by only taking trades aligned with higher timeframe trend.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (with 26-period look-ahead displacement)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Current cloud boundaries (Senkou Span A and B)
    upper_cloud = np.maximum(senkou_a_6h, senkou_b_6h)
    lower_cloud = np.minimum(senkou_a_6h, senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i]  # Bullish TK cross
        tk_cross_down = tenkan_6h[i] < kijun_6h[i]  # Bearish TK cross
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        
        # Entry conditions
        if tk_cross_up and price_above_cloud and position != 1:
            # Strong bullish signal: TK cross above cloud
            position = 1
            signals[i] = 0.25
        elif tk_cross_down and price_below_cloud and position != -1:
            # Strong bearish signal: TK cross below cloud
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TK cross or price re-enters cloud
        elif position == 1 and (tk_cross_down or not price_above_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_up or not price_below_cloud):
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

name = "6h_12h_1d_ichimoku_cloud_trend"
timeframe = "6h"
leverage = 1.0