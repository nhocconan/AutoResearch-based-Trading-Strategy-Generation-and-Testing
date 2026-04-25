#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: 6h Ichimoku TK cross with Kumo twist (Senkou A/B cross) from 1d as trend filter.
Long when Tenkan > Kijun and price above Kumo with bullish Kumo twist (Senkou A > Senkou B) from 1d.
Short when Tenkan < Kijun and price below Kumo with bearish Kumo twist (Senkou A < Senkou B) from 1d.
Exit on opposite TK cross or Kumo trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
Ichimoku works in bull via trend-following crosses, in bear via mean reversion at Kumo edges.
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components for 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b_6h = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to original timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    
    # Get 1d data for Kumo twist (Senkou A/B cross) as trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components for 1d (same as 6h but daily)
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo twist: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
    kumo_twist_bullish = senkou_a_1d > senkou_b_1d
    kumo_twist_bearish = senkou_a_1d < senkou_b_1d
    
    # Align Kumo twist to original timeframe
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        lower_kumo = np.minimum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun, price above Kumo, bullish Kumo twist from 1d
            long_signal = (tenkan_6h_aligned[i] > kijun_6h_aligned[i]) and \
                          (close[i] > upper_kumo) and \
                          kumo_twist_bullish_aligned[i]
            # Short: Tenkan < Kijun, price below Kumo, bearish Kumo twist from 1d
            short_signal = (tenkan_6h_aligned[i] < kijun_6h_aligned[i]) and \
                           (close[i] < lower_kumo) and \
                           kumo_twist_bearish_aligned[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: Tenkan < Kijun or price below Kumo or Kumo twist turns bearish
            exit_signal = (tenkan_6h_aligned[i] < kijun_6h_aligned[i]) or \
                          (close[i] < lower_kumo) or \
                          kumo_twist_bearish_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: Tenkan > Kijun or price above Kumo or Kumo twist turns bullish
            exit_signal = (tenkan_6h_aligned[i] > kijun_6h_aligned[i]) or \
                          (close[i] > upper_kumo) or \
                          kumo_twist_bullish_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0