#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Trend
Hypothesis: On 6h timeframe, use Ichimoku Kumo twist (Senkou Span A/B cross) from daily timeframe as trend filter, with Tenkan/Kijun cross on 6h for entry. Works in bull/bear as Kumo twist identifies major trend changes, while TK cross captures pullbacks in trend. Target: 15-25 trades/year.
"""

name = "6h_Ichimoku_Kumo_Twist_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: tenkan=9, kijun=26, senkou=52
    tenkan_period = 9
    kijun_period = 26
    senkou_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_period, min_periods=senkou_period).max().values + 
                     pd.Series(low_1d).rolling(window=senkou_period, min_periods=senkou_period).min().values) / 2
    
    # Kumo twist: Senkou Span A cross above/below Senkou Span B
    # Bullish twist: Senkou A > Senkou B (after previously being below)
    # Bearish twist: Senkou A < Senkou B (after previously being above)
    senkou_a_above_b = senkou_span_a > senkou_span_b
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_above_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_above_b.astype(float))
    
    # Get 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TK cross: Tenkan > Kijun for bullish, Tenkan < Kijun for bearish
    tk_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Senkou B (52 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_above_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Kumo bullish twist + TK bullish cross
            if senkou_a_above_b_aligned[i] == 1.0 and tk_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Kumo bearish twist + TK bearish cross
            elif senkou_a_above_b_aligned[i] == 0.0 and tk_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Kumo turns bearish OR TK turns bearish
            if senkou_a_above_b_aligned[i] == 0.0 or tk_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Kumo turns bullish OR TK turns bullish
            if senkou_a_above_b_aligned[i] == 1.0 or tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals