#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud strategy with 1d timeframe alignment.
# Uses Tenkan-sen (9-period) and Kijun-sen (26-period) crosses as entry signals,
# filtered by price position relative to Kumo (cloud) from Senkou Span A/B.
# Cloud color (Senkou Span A > B for bullish, A < B for bearish) provides trend filter.
# Trades only when price is outside the cloud to avoid false signals in consolidation.
# Designed for low trade frequency (~20-40/year) to minimize fee drag on 6h timeframe.
# Works in bull markets via cloud breakouts and in bear markets via cloud rejections.

name = "6h_Ichimoku_Cloud_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d Ichimoku components (Senkou Span A/B require 52 periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align all 1d indicators to 6s timeframe with proper delay
    tenkan_1d = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_1d = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after sufficient data for Senkou Span B
        if (np.isnan(tenkan_1d[i]) or np.isnan(kijun_1d[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries: Senkou Span A and B form the Kumo (cloud)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Bullish cloud: Senkou A > Senkou B
        # Bearish cloud: Senkou A < Senkou B
        
        if position == 0:
            # LONG: Price above cloud AND Tenkan crosses above Kijun
            if (close[i] > upper_cloud and 
                tenkan_1d[i] > kijun_1d[i] and 
                tenkan_1d[i-1] <= kijun_1d[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud AND Tenkan crosses below Kijun
            elif (close[i] < lower_cloud and 
                  tenkan_1d[i] < kijun_1d[i] and 
                  tenkan_1d[i-1] >= kijun_1d[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud or Tenkan crosses below Kijun
            if (close[i] < upper_cloud or 
                tenkan_1d[i] < kijun_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud or Tenkan crosses above Kijun
            if (close[i] > lower_cloud or 
                tenkan_1d[i] > kijun_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals