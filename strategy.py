#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) combined with Kumo breakout and daily EMA trend filter.
Works in bull/bear by trading in direction of daily trend (EMA50). Cloud twist signals momentum shift,
while price above/below cloud confirms trend continuation. Uses 6h for entry timing and 1d for trend/filter.
Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag.
"""

name = "6h_Ichimoku_Kumo_Twist_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = np.full(len(high_1d), np.nan)
    if len(high_1d) >= tenkan_period:
        for i in range(tenkan_period - 1, len(high_1d)):
            tenkan_sen_1d[i] = (np.max(high_1d[i - tenkan_period + 1:i + 1]) + 
                               np.min(low_1d[i - tenkan_period + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = np.full(len(high_1d), np.nan)
    if len(high_1d) >= kijun_period:
        for i in range(kijun_period - 1, len(high_1d)):
            kijun_sen_1d[i] = (np.max(high_1d[i - kijun_period + 1:i + 1]) + 
                              np.min(low_1d[i - kijun_period + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1d = np.full(len(high_1d), np.nan)
    if len(high_1d) >= kijun_period:
        for i in range(len(high_1d)):
            idx = i + kijun_period
            if idx < len(high_1d) and not np.isnan(tenkan_sen_1d[i]) and not np.isnan(kijun_sen_1d[i]):
                senkou_span_a_1d[idx] = (tenkan_sen_1d[i] + kijun_sen_1d[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = np.full(len(high_1d), np.nan)
    if len(high_1d) >= senkou_span_b_period:
        for i in range(senkou_span_b_period - 1, len(high_1d)):
            senkou_span_b_1d[i] = (np.max(high_1d[i - senkou_span_b_period + 1:i + 1]) + 
                                  np.min(low_1d[i - senkou_span_b_period + 1:i + 1])) / 2
        # Shift forward by kijun_period (26)
        senkou_span_b_shifted = np.full(len(high_1d), np.nan)
        for i in range(len(high_1d) - kijun_period):
            if not np.isnan(senkou_span_b_1d[i]):
                senkou_span_b_shifted[i + kijun_period] = senkou_span_b_1d[i]
        senkou_span_b_1d = senkou_span_b_shifted
    
    # Kumo (cloud) top and bottom
    kumo_top_1d = np.full(len(high_1d), np.nan)
    kumo_bottom_1d = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(senkou_span_a_1d[i]) and not np.isnan(senkou_span_b_1d[i]):
            kumo_top_1d[i] = max(senkou_span_a_1d[i], senkou_span_b_1d[i])
            kumo_bottom_1d[i] = min(senkou_span_a_1d[i], senkou_span_b_1d[i])
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d indicators to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for Ichimoku and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Kumo twist: Tenkan-sen/Kijun-sen cross above/below cloud
        tk_cross_above = tenkan_sen_1d_aligned[i] > kijun_sen_1d_aligned[i]
        tk_cross_below = tenkan_sen_1d_aligned[i] < kijun_sen_1d_aligned[i]
        
        # Price relative to cloud
        price_above_kumo = close[i] > kumo_top_1d_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_1d_aligned[i]
        
        # Trend filter
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: TK cross above + price above cloud + uptrend
            if tk_cross_above and price_above_kumo and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + downtrend
            elif tk_cross_below and price_below_kumo and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross below OR price drops below cloud OR trend turns down
            if not tk_cross_above or not price_above_kumo or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross above OR price rises above cloud OR trend turns up
            if not tk_cross_below or not price_below_kumo or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals