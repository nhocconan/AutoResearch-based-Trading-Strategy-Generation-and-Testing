#!/usr/bin/env python3
"""
6h_1d_Ichimoku_TK_Cross_Cloud_Filter_v1
Hypothesis: Use 1d Ichimoku cloud for long-term trend direction and 6h Tenkan/Kijun cross for entry timing.
Long when price breaks above Kumo cloud (from 1d) and Tenkan > Kijun on 6h; short when price breaks below Kumo and Tenkan < Kijun.
Exit when Tenkan/Kijun cross reverses or price re-enters cloud.
Ichimoku works well in trending markets and avoids whipsaws in ranging markets via cloud filter.
Designed for 6h timeframe to limit trade frequency while capturing multi-day trends.
"""

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Tenkan-sen (9-period) and Kijun-sen (26-period)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (9), Kijun-sen (26), Senkou Span A & B (52)
    high_9_1d = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2)
    high_52_1d = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52_1d + low_52_1d) / 2
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo cloud boundaries (Senkou Span A and B shifted forward 26 periods)
    # For cloud at time t, we use Senkou values from 26 periods ago
    senkou_span_a_shifted = np.roll(senkou_span_a_aligned, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b_aligned, 26)
    # First 26 values are invalid due to shift
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above cloud AND Tenkan > Kijun
            if (close[i] > cloud_top[i] and tenkan[i] > kijun[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud AND Tenkan < Kijun
            elif (close[i] < cloud_bottom[i] and tenkan[i] < kijun[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan < Kijun OR price re-enters cloud
            if tenkan[i] < kijun[i] or (close[i] > cloud_bottom[i] and close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan > Kijun OR price re-enters cloud
            if tenkan[i] > kijun[i] or (close[i] > cloud_bottom[i] and close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals