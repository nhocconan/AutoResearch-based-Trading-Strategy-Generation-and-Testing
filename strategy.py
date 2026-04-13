#!/usr/bin/env python3
"""
6h_1d_1w_IchimokuCloudTrend
Ichimoku-based strategy: TK cross + cloud filter from 1d, with weekly trend confirmation.
Targets 15-35 trades/year (60-140 total over 4 years) on 6h timeframe.
Uses Ichimoku cloud as dynamic support/resistance and TK cross for momentum.
Weekly trend filter prevents counter-trade entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly 50 EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: price above EMA = uptrend, below EMA = downtrend
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align weekly trend to 6h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(52, n):  # Start after Ichimoku calculation period
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross signals
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Entry conditions: TK cross + price outside cloud + weekly trend alignment
        long_entry = (tk_cross_up and 
                      price_above_cloud and 
                      uptrend_1w_aligned[i] > 0.5)
        short_entry = (tk_cross_down and 
                       price_below_cloud and 
                       downtrend_1w_aligned[i] > 0.5)
        
        # Exit when TK cross reverses or price re-enters cloud
        exit_long = position == 1 and (not tk_cross_up or not price_above_cloud)
        exit_short = position == -1 and (not tk_cross_down or not price_below_cloud)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_IchimokuCloudTrend"
timeframe = "6h"
leverage = 1.0