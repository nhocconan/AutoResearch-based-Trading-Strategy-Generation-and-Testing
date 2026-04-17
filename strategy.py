#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Cloud_Trend_v1
6h strategy using Ichimoku Cloud from 1d and Tenkan/Kijun cross from 1w for trend alignment.
Enters long when price is above 1d Kumo cloud AND 1w Tenkan > Kijun.
Enters short when price is below 1d Kumo cloud AND 1w Tenkan < Kijun.
Exits when price crosses back into the cloud or Tenkan/Kijun cross reverses.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A, Senkou B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Get 1d Ichimoku data ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1d
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === Get 1w Tenkan/Kijun for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Ichimoku on 1w (only need Tenkan and Kijun)
    tenkan_1w, kijun_1w, _, _ = calculate_ichimoku(high_1w, low_1w, close_1d)  # close_1d is placeholder, will be replaced
    
    # Recalculate 1w Ichimoku with correct close
    tenkan_1w, kijun_1w, _, _ = calculate_ichimoku(high_1w, low_1w, df_1w['close'].values)
    
    # Align 1w Tenkan/Kijun to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or 
            np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Cloud boundaries (Senkou A and B shifted forward 26 periods)
        # For Ichimoku cloud, Senkou spans are plotted 26 periods ahead
        # So we need to look at current Senkou values that were calculated 26 periods ago
        if i >= 26:
            senkou_a_current = senkou_a_1d_aligned[i - 26]
            senkou_b_current = senkou_b_1d_aligned[i - 26]
        else:
            # Not enough data for cloud projection
            signals[i] = 0.0
            position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_current, senkou_b_current)
        cloud_bottom = min(senkou_a_current, senkou_b_current)
        
        # Tenkan/Kijun cross on 1w
        tk_cross_1w = tenkan_1w_aligned[i] - kijun_1w_aligned[i]
        tk_cross_1w_prev = tenkan_1w_aligned[i-1] - kijun_1w_aligned[i-1] if i > 0 else 0
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish = tk_cross_1w > 0 and tk_cross_1w_prev <= 0
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish = tk_cross_1w < 0 and tk_cross_1w_prev >= 0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above cloud AND bullish TK cross on 1w
            if close[i] > cloud_top and tk_bullish:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below cloud AND bearish TK cross on 1w
            elif close[i] < cloud_bottom and tk_bearish:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below cloud OR TK cross turns bearish
            if close[i] < cloud_top or tk_bearish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above cloud OR TK cross turns bullish
            if close[i] > cloud_bottom or tk_bullish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0