#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dTrend_Filter
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d trend filter (price vs Kumo) captures momentum shifts.
In bull markets: longs when TK cross bullish + price above Kumo. In bear markets: shorts when TK cross bearish + price below Kumo.
Uses discrete sizing (0.25) and exits on opposite TK cross or Kumo break. Designed for low turnover (~15-25 trades/year) 
and works in both regimes by requiring alignment with higher timeframe trend structure.
"""

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
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).mean() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).mean() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).mean()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).mean() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).mean()) / 2)
    # Kumo (cloud) top/bottom: Senkou Span A/B
    kumotop_1d = np.where(senkou_a_1d >= senkou_b_1d, senkou_a_1d, senkou_b_1d)
    kumobottom_1d = np.where(senkou_a_1d <= senkou_b_1d, senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    kumotop_1d_aligned = align_htf_to_ltf(prices, df_1d, kumotop_1d)
    kumobottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumobottom_1d)
    
    # 6h Ichimoku for entry signal (TK cross)
    # Tenkan-sen (6h)
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).mean() + 
                 pd.Series(low).rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (6h)
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).mean() + 
                pd.Series(low).rolling(window=26, min_periods=26).mean()) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 6h TK (26) and 1d aligned arrays
    start_idx = max(26, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(kumotop_1d_aligned[i]) or np.isnan(kumobottom_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 6h TK cross signals
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # 1d trend: price vs Kumo
        price_above_kumo = close[i] > kumotop_1d_aligned[i]
        price_below_kumo = close[i] < kumobottom_1d_aligned[i]
        
        if position == 0:
            # Long: bullish TK cross + price above 1d Kumo (bullish alignment)
            if tk_bullish and price_above_kumo:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below 1d Kumo (bearish alignment)
            elif tk_bearish and price_below_kumo:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold
            signals[i] = 0.25
            # Exit: bearish TK cross OR price breaks below 1d Kumo bottom
            if tk_bearish or price_below_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold
            signals[i] = -0.25
            # Exit: bullish TK cross OR price breaks above 1d Kumo top
            if tk_bullish or price_above_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0