#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1D
Hypothesis: Use Ichimoku Tenkan/Kijun cross on 6h with 1d cloud filter for trend direction.
In bull markets: price above cloud + TK cross up = long.
In bear markets: price below cloud + TK cross down = short.
Ichimoku provides built-in trend, support/resistance, and momentum.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Ichimoku on 6h: Tenkan (9), Kijun (26), Senkou A/B (26, 52)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # 1d cloud filter: get daily Ichimoku cloud for trend context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan and Kijun (9, 26)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Daily Senkou A and B
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Current cloud bounds (senkou A and B)
    span_a = senkou_a
    span_b = senkou_b
    
    # For cloud, we need the values from 26 periods ago (since senkou is plotted ahead)
    # But for price vs cloud comparison, we use current senkou values
    # Actually, senkou span is plotted 26 periods ahead, so to get current cloud,
    # we need senkou values from 26 periods ago
    span_a_lagged = np.roll(span_a, 26)
    span_b_lagged = np.roll(span_b, 26)
    # Fill first 26 with NaN
    span_a_lagged[:26] = np.nan
    span_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(span_a_lagged, span_b_lagged)
    cloud_bottom = np.minimum(span_a_lagged, span_b_lagged)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(52, 26)  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_1d[i]) or np.isnan(kijun_1d[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        
        # 1d trend: price vs 1d cloud
        price_vs_1d_cloud_top = senkou_a_1d_aligned[i]
        price_vs_1d_cloud_bottom = senkou_b_1d_aligned[i]
        # Actually, we need to compare price to 1d cloud
        # Get 1d cloud values aligned
        span_a_1d_aligned = senkou_a_1d_aligned[i]
        span_b_1d_aligned = senkou_b_1d_aligned[i]
        cloud_top_1d = max(span_a_1d_aligned, span_b_1d_aligned)
        cloud_bottom_1d = min(span_a_1d_aligned, span_b_1d_aligned)
        
        if position == 0:
            # Long: price above cloud + TK cross up + 1d bullish (price above 1d cloud)
            if (price > cloud_top_val and 
                tenkan_val > kijun_val and 
                tenkan[i-1] <= kijun[i-1] and  # crossed up this bar
                price > cloud_top_1d):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + 1d bearish (price below 1d cloud)
            elif (price < cloud_bottom_val and 
                  tenkan_val < kijun_val and 
                  tenkan[i-1] >= kijun[i-1] and  # crossed down this bar
                  price < cloud_bottom_1d):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price drops below cloud OR TK cross down
            if price < cloud_bottom_val or (tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK cross up
            if price > cloud_top_val or (tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1D"
timeframe = "6h"
leverage = 1.0