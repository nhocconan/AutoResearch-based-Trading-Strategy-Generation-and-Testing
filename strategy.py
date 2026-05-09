#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend
Hypothesis: Use Ichimoku Cloud from 1d timeframe as the primary trend filter, with TK cross on 6h for entry timing.
In bull markets: price above cloud + TK cross up = long.
In bear markets: price below cloud + TK cross down = short.
The Ichimoku Cloud provides strong support/resistance levels and future projection, reducing false breaks.
TK cross (Tenkan/Kijun) provides timely entry signals within the trend.
Designed for low trade frequency (~15-30/year) with high win rate by requiring multiple confluence factors.
"""

name = "6h_Ichimoku_Cloud_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku Cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate TK cross on 6h (using actual 6h price for Tenkan/Kijun calculation)
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_9_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_9_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h_raw = (max_high_9_6h + min_low_9_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h_raw = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK cross signals
    tk_cross_up = tenkan_6h_raw > kijun_6h_raw
    tk_cross_down = tenkan_6h_raw < kijun_6h_raw
    
    # Price relative to cloud
    price_above_cloud = (close > span_a_6h) & (close > span_b_6h)
    price_below_cloud = (close < span_a_6h) & (close < span_b_6h)
    
    # Cloud color (bullish/bearish)
    cloud_bullish = span_a_6h > span_b_6h
    cloud_bearish = span_a_6h < span_b_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(span_a_6h[i]) or 
            np.isnan(span_b_6h[i]) or np.isnan(tenkan_6h_raw[i]) or np.isnan(kijun_6h_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud + TK cross up + bullish cloud
            if price_above_cloud[i] and tk_cross_up[i] and cloud_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + bearish cloud
            elif price_below_cloud[i] and tk_cross_down[i] and cloud_bearish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below cloud OR TK cross down
            if price_below_cloud[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above cloud OR TK cross up
            if price_above_cloud[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals