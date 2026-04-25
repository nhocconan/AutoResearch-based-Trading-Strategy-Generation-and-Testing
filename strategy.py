#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) on 6h with 1d cloud filter (price above/below cloud) for trend alignment. 
Only trades when TK cross occurs in direction of 1d cloud (bullish: price > cloud, bearish: price < cloud). 
Uses discrete position sizing (0.25) to minimize fee churn. Designed for medium frequency (50-150 trades/4 years) 
to work in both bull and bear markets via cloud-based trend filter that adapts to volatility.
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
    
    # Get 1d data for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align HTF Ichimoku components to 6h timeframe (no additional delay needed for cloud)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate TK cross on 6h
    # Tenkan-sen (6h)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (6h)
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals: Tenkan crossing above/below Kijun
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))
    
    # Determine cloud color and price position relative to cloud
    # Bullish cloud: Senkou A > Senkou B
    # Bearish cloud: Senkou A < Senkou B
    # Price above cloud: close > max(Senkou A, Senkou B)
    # Price below cloud: close < min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and 6h TK components (26)
    start_idx = max(52, 26)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for TK cross signals with cloud filter
            # Long: bullish TK cross (Tenkan > Kijun) when price is above bullish cloud
            # Short: bearish TK cross (Tenkan < Kijun) when price is below bearish cloud
            long_signal = tk_cross_above[i] and price_above_cloud[i] and (senkou_a_aligned[i] > senkou_b_aligned[i])
            short_signal = tk_cross_below[i] and price_below_cloud[i] and (senkou_a_aligned[i] < senkou_b_aligned[i])
            
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
            # Exit when Tenkan crosses back below Kijun (TK cross bearish)
            exit_signal = tk_cross_below[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Tenkan crosses back above Kijun (TK cross bullish)
            exit_signal = tk_cross_above[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter"
timeframe = "6h"
leverage = 1.0