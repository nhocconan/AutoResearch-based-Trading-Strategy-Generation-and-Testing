#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dTrend_Filter
Hypothesis: Ichimoku Tenkan/Kijun cross on 6h with 1d trend filter (price vs Kumo) captures momentum with controlled frequency. Weekly trend avoided to reduce noise in bear markets. Discrete sizing (0.25) balances return and fee drag. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Ichimoku (52 periods for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        # Determine if price is above/below cloud
        price_above_kumo = close_val > upper_kumo
        price_below_kumo = close_val < lower_kumo
        
        # Determine Tenkan/Kijun cross
        tenkan_prev = tenkan_aligned[i-1]
        kijun_prev = kijun_aligned[i-1]
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_prev <= kijun_prev)
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_prev >= kijun_prev)
        
        if position == 0:
            # Long: TK cross up + price above cloud
            if tk_cross_up and price_above_kumo:
                signals[i] = size
                position = 1
            # Short: TK cross down + price below cloud
            elif tk_cross_down and price_below_kumo:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK cross down OR price drops below cloud
            if tk_cross_down or price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up OR price rises above cloud
            if tk_cross_up or price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0