#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend
Hypothesis: Ichimoku TK cross with cloud filter on 6h, aligned with 1w trend (price > weekly Kumo top for longs, < weekly Kumo bottom for shorts).
Works in bull/bear markets: In trending regimes (price above/below weekly cloud), TK cross captures momentum with cloud acting as dynamic support/resistance.
Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn. Targets 50-150 trades over 4 years on 6h timeframe.
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
    
    # Get 1w data for trend filter (Kumo)
    df_1w = get_htf_data(prices, '1w')
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 1w Kumo (cloud) for trend filter
    # Weekly Tenkan and Kijun
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_period9_high = pd.Series(wk_high).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(wk_low).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    wk_period26_high = pd.Series(wk_high).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(wk_low).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly Senkou Span A and B
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_period52_high = pd.Series(wk_high).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(wk_low).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2
    
    # Weekly Kumo top (max of Senkou A/B) and bottom (min of Senkou A/B)
    wk_kumo_top = np.maximum(wk_senkou_a, wk_senkou_b)
    wk_kumo_bottom = np.minimum(wk_senkou_a, wk_senkou_b)
    
    # Align Ichimoku and Kumo to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    wk_kumo_top_aligned = align_htf_to_ltf(prices, df_1w, wk_kumo_top)
    wk_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, wk_kumo_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position
    
    # Warmup: need 52-period calculations
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(wk_kumo_top_aligned[i]) or np.isnan(wk_kumo_bottom_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        wk_kumo_top_val = wk_kumo_top_aligned[i]
        wk_kumo_bottom_val = wk_kumo_bottom_aligned[i]
        
        # Kumo (cloud) boundaries
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Look for entry: TK cross with price above/below weekly Kumo
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_tk = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_tk = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            long_condition = bullish_tk and close_val > wk_kumo_top_val
            short_condition = bearish_tk and close_val < wk_kumo_bottom_val
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below Kumo OR TK cross turns bearish
            if close_val < lower_kumo or (tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Kumo OR TK cross turns bullish
            if close_val > upper_kumo or (tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend"
timeframe = "6h"
leverage = 1.0