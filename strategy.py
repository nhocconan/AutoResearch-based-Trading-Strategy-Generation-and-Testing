#!/usr/bin/env python3
# 6h_ichimoku_kijun_sen_cross_v2
# Hypothesis: 6h strategy using Ichimoku Kijun-sen/Tenkan-sen cross with daily cloud filter.
# Long when Tenkan > Kijun and price above daily cloud (Senou Span A/B max).
# Short when Tenkan < Kijun and price below daily cloud (Senou Span A/B min).
# Uses daily HTF for cloud to avoid look-ahead and ensure completed-bar alignment.
# Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear by following
# institutional trend alignment via Ichimoku cloud as dynamic support/resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kijun_sen_cross_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components (9, 26, 52 periods) on 6h
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senou Span A: (Tenkan + Kijun) / 2
    senou_a = (tenkan + kijun) / 2
    
    # Senou Span B: (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senou_b = (high_52 + low_52) / 2
    
    # Daily HTF for cloud alignment
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Ichimoku components (9, 26, 52)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senou_a_1d = (tenkan_1d + kijun_1d) / 2
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Daily cloud: max/min of Senou Span A/B
    daily_cloud_top = np.maximum(senou_a_1d, senou_b_1d)
    daily_cloud_bottom = np.minimum(senou_a_1d, senou_b_1d)
    
    # Align daily cloud to 6h
    daily_cloud_top_aligned = align_htf_to_ltf(prices, df_1d, daily_cloud_top)
    daily_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, daily_cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(close[i]) or
            np.isnan(daily_cloud_top_aligned[i]) or np.isnan(daily_cloud_bottom_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Tenkan < Kijun OR price below daily cloud
            if tenkan[i] < kijun[i] or close[i] < daily_cloud_bottom_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan > Kijun OR price above daily cloud
            if tenkan[i] > kijun[i] or close[i] > daily_cloud_top_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter: Tenkan/Kijun cross with price vs daily cloud
            if tenkan[i] > kijun[i] and close[i] > daily_cloud_top_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif tenkan[i] < kijun[i] and close[i] < daily_cloud_bottom_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals