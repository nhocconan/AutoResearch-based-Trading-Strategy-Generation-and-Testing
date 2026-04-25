#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Tenkan_Cross_1wTrend_Filter
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1w trend filter (price > 1w Kumo cloud top for long, < bottom for short).
Only trade in alignment with weekly trend to avoid counter-trend whipsaws. Uses Kumo twist (Senkou A/B cross) as additional trend strength filter.
Designed for ~15-30 trades/year by requiring weekly trend alignment and Kumo confirmation.
Works in bull/bear markets via weekly trend filter; avoids false signals in sideways markets via Kumo twist requirement.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # need at least 52 weeks for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Kumo twist detection: Senkou A crossing Senkou B (trend strength indicator)
    # Kumo twist bullish: Senkou A > Senkou B (after previously being below)
    # Kumo twist bearish: Senkou A < Senkou B (after previously being above)
    kumo_twist_bullish = senkou_a_aligned > senkou_b_aligned
    kumo_twist_bearish = senkou_a_aligned < senkou_b_aligned
    
    # Weekly trend filter: price relative to Kumo cloud
    # Weekly uptrend: price > Kumo top (max of Senkou A/B)
    # Weekly downtrend: price < Kumo bottom (min of Senkou A/B)
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    weekly_uptrend = close > kumo_top
    weekly_downtrend = close < kumo_bottom
    
    # Tenkan/Kijun cross signals
    # Bullish cross: Tenkan crosses above Kijun
    # Bearish cross: Tenkan crosses below Kijun
    tenkan_above_kijun = tenkan_aligned > kijun_aligned
    tenkan_below_kijun = tenkan_aligned < kijun_aligned
    
    # Detect crossovers (requires previous bar state)
    tenkan_above_kijun_prev = np.roll(tenkan_above_kijun, 1)
    tenkan_below_kijun_prev = np.roll(tenkan_below_kijun, 1)
    tenkan_above_kijun_prev[0] = False
    tenkan_below_kijun_prev[0] = False
    
    bullish_cross = tenkan_above_kijun & (~tenkan_above_kijun_prev)
    bearish_cross = tenkan_below_kijun & (~tenkan_below_kijun_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations (max period is 52)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entry signals aligned with weekly trend and Kumo twist
            long_signal = (bullish_cross[i] and 
                          weekly_uptrend[i] and 
                          kumo_twist_bullish[i])
            
            short_signal = (bearish_cross[i] and 
                           weekly_downtrend[i] and 
                           kumo_twist_bearish[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on bearish cross or weekly trend breakdown
            signals[i] = 0.25
            exit_signal = bearish_cross[i] or (~weekly_uptrend[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short position: exit on bullish cross or weekly trend breakdown
            signals[i] = -0.25
            exit_signal = bullish_cross[i] or (~weekly_downtrend[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kijun_Tenkan_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0