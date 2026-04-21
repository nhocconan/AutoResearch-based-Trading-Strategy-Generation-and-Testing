#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Trend_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 1d as trend filter, combined with TK cross on 6h for entries.
Cloud twist indicates trend acceleration; TK cross provides timing. Works in bull/bear via trend filter.
Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Ichimoku components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === 6h TK Cross (Tenkan/Kijun) for entry timing ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (9-period) on 6h
    max_high_9_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    min_low_9_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (max_high_9_6h + min_low_9_6h) / 2
    
    # Kijun-sen (26-period) on 6h
    max_high_26_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    min_low_26_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK Cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) 
            or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to Kumo (cloud)
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        kum_top = max(senkou_a, senkou_b)
        kum_bottom = min(senkou_a, senkou_b)
        price = close_6h[i]
        
        # Determine trend: price above cloud = bullish, below cloud = bearish
        price_above_kumo = price > kum_top
        price_below_kumo = price < kum_bottom
        
        # Cloud twist detection: Senkou Span A/B cross indicating trend change
        senkou_a_prev = senkou_a_1d_aligned[i-1] if i > 0 else senkou_a
        senkou_b_prev = senkou_b_1d_aligned[i-1] if i > 0 else senkou_b
        twist_bullish = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
        twist_bearish = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
        
        # Entry conditions
        long_entry = price_above_kumo & tk_cross_above & twist_bullish
        short_entry = price_below_kumo & tk_cross_below & twist_bearish
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price re-enters cloud or TK cross down
            if price < kum_top or (tenkan_6h[i] < kijun_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price re-enters cloud or TK cross up
            if price > kum_bottom or (tenkan_6h[i] > kijun_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Trend_v1"
timeframe = "6h"
leverage = 1.0