#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend
Hypothesis: Ichimoku cloud twist (TK cross) combined with weekly trend filter (price vs weekly EMA) captures trend reversals in both bull and bear markets. The cloud acts as dynamic support/resistance, reducing whipsaws. Weekly trend filter ensures alignment with higher timeframe momentum, improving win rate during strong trends while avoiding counter-trend trades in ranging markets. Target: 15-30 trades/year on 6H timeframe.
"""

name = "6h_Ichimoku_Kumo_Twist_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate weekly EMA26 for trend filter
    close_1w = df_1w['close'].values
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52 periods) and weekly EMA (26 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema26_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross (Tenkan-sen crosses Kijun-sen)
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema26_1w_aligned[i]
        weekly_downtrend = close[i] < ema26_1w_aligned[i]
        
        if position == 0:
            # Long entry: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and price_below_cloud and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price falls below cloud bottom
            if tk_cross_down or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price rises above cloud top
            if tk_cross_up or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals