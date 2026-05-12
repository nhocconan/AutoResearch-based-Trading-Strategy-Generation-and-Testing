#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Tenkan_Cross_1dCloud_Filter
Hypothesis: Ichimoku Tenkan/Kijun cross on 6h timeframe, filtered by 1d Kumo cloud (bullish/bearish), captures momentum with reduced whipsaws. Works in bull markets via long signals above cloud and in bear markets via short signals below cloud. The cloud acts as dynamic support/resistance, improving signal quality.
"""

name = "6h_Ichimoku_Kijun_Tenkan_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Calculate 1d Ichimoku components for cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # For 1d: Tenkan1d = (9-period high + low)/2, Kijun1d = (26-period high + low)/2
    high_1d_tenkan = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_1d_tenkan = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_1d_tenkan + low_1d_tenkan) / 2
    
    high_1d_kijun = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_1d_kijun = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_1d_kijun + low_1d_kijun) / 2
    
    senkou_a = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_1d_senkou = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_1d_senkou = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_1d_senkou + low_1d_senkou) / 2)
    
    # Shift Senkou spans by 26 periods (cloud is plotted 26 periods ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Note: using df_1d as reference for alignment, but values are 6h
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # Determine cloud boundaries (top and bottom of cloud)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Kijun (26) and Senkou (52) warmup
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above cloud (bullish)
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # crossed just now
                close[i] > cloud_top[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below cloud (bearish)
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # crossed just now
                  close[i] < cloud_bottom[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price falls below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] and 
                tenkan_aligned[i-1] >= kijun_aligned[i-1]) or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1]) or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals