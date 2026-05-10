#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filter_1d
# Hypothesis: Use Ichimoku cloud (Senkou Span A/B) from 1d as trend filter. 
# Long when price > cloud and Tenkan > Kijun on 6h, short when price < cloud and Tenkan < Kijun.
# Exit when price crosses back into cloud or Tenkan/Kijun cross reverses.
# Ichimoku provides multi-line trend confirmation that works in both bull and bear markets.
# Tenkan/Kijun cross gives timely entries/exits while cloud filter avoids counter-trend trades.
# Target: 20-50 trades/year to minimize fee drag.

name = "6h_Ichimoku_Cloud_Trend_Filter_1d"
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
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Get Ichimoku components from 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen on 1d
    max_high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_9_1d + min_low_9_1d) / 2
    
    # Kijun-sen on 1d
    max_high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_26_1d + min_low_26_1d) / 2
    
    # Senkou Span A on 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B on 1d
    max_high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_52_1d + min_low_52_1d) / 2
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom from 1d
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud AND Tenkan > Kijun (bullish momentum)
            if close[i] > cloud_top_1d[i] and tenkan[i] > kijun[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND Tenkan < Kijun (bearish momentum)
            elif close[i] < cloud_bottom_1d[i] and tenkan[i] < kijun[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below cloud OR Tenkan crosses below Kijun
            if close[i] < cloud_top_1d[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above cloud OR Tenkan crosses above Kijun
            if close[i] > cloud_bottom_1d[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals