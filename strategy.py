#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 1d as regime filter, combined with Tenkan-Kijun cross on 6h for entry. Cloud twist indicates major trend change; TK cross in direction of new trend captures momentum. Works in bull/bear by only trading in direction of higher timeframe regime. Target: 60-120 total trades over 4 years (15-30/year).
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
    
    # Load 1d data ONCE before loop for Ichimoku (regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Need sufficient 1d data for Ichimoku calculation (52 periods max)
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed for TK cross)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate TK cross on 6h for entry signal
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # TK cross signals: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h[:-1] <= kijun_6h[:-1])  # previous bar condition
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h[:-1] >= kijun_6h[:-1])
    
    # Handle first bar for cross signals
    tk_cross_up = np.insert(tk_cross_up[1:], 0, False)
    tk_cross_down = np.insert(tk_cross_down[1:], 0, False)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(52, 26, 9)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d regime filter: Kumo twist (Senkou Span A/B cross)
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        senkou_a_above = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        senkou_a_above_prev = senkou_a_1d_aligned[i-1] > senkou_b_1d_aligned[i-1]
        kumo_twist_bullish = senkou_a_above and not senkou_a_above_prev
        kumo_twist_bearish = not senkou_a_above and senkou_a_above_prev
        
        # Current Kumo state (cloud color)
        bullish_kumo = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        bearish_kumo = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # Entry logic: TK cross in direction of Kumo twist or established Kumo
        long_entry = False
        short_entry = False
        
        # Aggressive entry: TK cross during Kumo twist (trend change)
        if tk_cross_up[i] and (kumo_twist_bullish or bullish_kumo):
            long_entry = True
        if tk_cross_down[i] and (kumo_twist_bearish or bearish_kumo):
            short_entry = True
        
        # Conservative entry: TK cross only when Kumo is established (after twist)
        # long_entry = tk_cross_up[i] and bullish_kumo
        # short_entry = tk_cross_down[i] and bearish_kumo
        
        # Exit logic: TK cross in opposite direction or price moves too far from cloud
        long_exit = tk_cross_down[i] and bearish_kumo
        short_exit = tk_cross_up[i] and bullish_kumo
        
        # Alternative exit: price closes outside cloud in opposite direction
        # long_exit = position == 1 and close[i] < senkou_b_1d_aligned[i]
        # short_exit = position == -1 and close[i] > senkou_a_1d_aligned[i]
        
        if long_entry and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_entry and position != -1:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_v1"
timeframe = "6h"
leverage = 1.0