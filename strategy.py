#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter for trend alignment.
- Long when TK crosses above AND price > 1d cloud (Senou Span A/B max)
- Short when TK crosses below AND price < 1d cloud (Senou Span A/B min)
- Uses completed 6h bars for TK cross to avoid look-ahead
- 1d cloud acts as dynamic support/resistance filter ensuring trades align with higher timeframe trend
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite TK cross or price re-entering cloud
- Novelty: Combines Ichimoku momentum with HTF cloud filter for BTC/ETH edge in trending markets while avoiding ranging conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 6h data ONCE before loop for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Ichimoku components on completed 6h bars
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan = (pd.Series(df_6h['high'].values).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
              pd.Series(df_6h['low'].values).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun = (pd.Series(df_6h['high'].values).rolling(window=period_kijun, min_periods=period_kijun).max() +
             pd.Series(df_6h['low'].values).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senou_span_a = (tenkan + kijun) / 2
    
    # Senou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senou_b = 52
    senou_span_b = (pd.Series(df_6h['high'].values).rolling(window=period_senou_b, min_periods=period_senou_b).max() +
                    pd.Series(df_6h['low'].values).rolling(window=period_senou_b, min_periods=period_senou_b).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun.values)
    senou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senou_span_a.values)
    senou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senou_span_b.values)
    
    # Load daily data ONCE before loop for cloud filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku cloud (Senou Span A/B) for trend filter
    # Tenkan-sen 1d
    tenkan_1d = (pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max() +
                 pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen 1d
    kijun_1d = (pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max() +
                pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min()) / 2
    # Senou Span A 1d
    senou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    # Senou Span B 1d
    senou_span_b_1d = (pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max() +
                       pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min()) / 2
    
    # Align daily cloud to 6h timeframe
    senou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senou_span_a_1d.values)
    senou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senou_span_b_1d.values)
    
    # Cloud boundaries: max/min of Senou Span A/B
    cloud_top_1d = np.maximum(senou_span_a_1d_aligned, senou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senou_span_a_1d_aligned, senou_span_b_1d_aligned)
    
    # TK cross signals (using completed 6h bars only)
    tk_cross_above = (tenkan_aligned > kijun_aligned) & (tenkan_aligned.shift(1) <= kijun_aligned.shift(1))
    tk_cross_below = (tenkan_aligned < kijun_aligned) & (tenkan_aligned.shift(1) >= kijun_aligned.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku TK cross with 1d cloud filter
        if position == 0:
            # Long: TK cross above AND price above 1d cloud
            if tk_cross_above[i] and close[i] > cloud_top_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below 1d cloud
            elif tk_cross_below[i] and close[i] < cloud_bottom_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross below OR price falls below 1d cloud
            if tk_cross_below[i] or close[i] < cloud_bottom_1d[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above OR price rises above 1d cloud
            if tk_cross_above[i] or close[i] > cloud_top_1d[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_v1"
timeframe = "6h"
leverage = 1.0