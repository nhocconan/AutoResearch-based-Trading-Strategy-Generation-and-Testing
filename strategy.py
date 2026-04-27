#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter
Hypothesis: Use Ichimoku TK cross (Tenkan/Kijun) on 6h timeframe with 1d cloud filter (Senkou Span A/B). 
Long when TK crosses above AND price > cloud (bullish), short when TK crosses below AND price < cloud (bearish).
Ichimoku provides built-in trend/momentum/cloud support/resistance, effective in both trending and ranging markets.
Cloud from higher timeframe (1d) adds institutional-level support/resistance filtering.
Target: 50-150 total trades over 4 years (~12-37/year) with size 0.25.
"""

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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Get 1d data for cloud filter (higher timeframe support/resistance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    max_high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_tenkan_1d + min_low_tenkan_1d) / 2
    
    # 1d Kijun-sen (26-period)
    max_high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_kijun_1d + min_low_kijun_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    max_high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((max_high_senkou_b_1d + min_low_senkou_b_1d) / 2)
    
    # Align 1d cloud components to 6h timeframe (wait for previous day's close)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud boundaries: max/min of Senkou A/B (cloud top/bottom)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # TK cross signals
    tk_cross_up = (tenkan > kijun) & (tenkan[:-1] <= kijun[:-1])  # crossed above
    tk_cross_down = (tenkan < kijun) & (tenkan[:-1] >= kijun[:-1])  # crossed below
    
    # Prepend False for first element comparison
    tk_cross_up = np.concatenate(([False], tk_cross_up[:-1]))
    tk_cross_down = np.concatenate(([False], tk_cross_down[:-1]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all calculations
    start_idx = max(52, 26)  # Senkou B needs 52 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross up AND price above cloud (bullish)
            if tk_cross_up[i] and close[i] > cloud_top[i]:
                signals[i] = size
                position = 1
            # Short: TK cross down AND price below cloud (bearish)
            elif tk_cross_down[i] and close[i] < cloud_bottom[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK cross down OR price drops below cloud
            if tk_cross_down[i] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up OR price rises above cloud
            if tk_cross_up[i] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0