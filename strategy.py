#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Ichimoku cloud breakout with daily trend filter works in both bull and bear markets.
The 1d Ichimoku cloud provides clear support/resistance zones, and breakouts above/below the cloud
with TK cross confirmation capture strong trends. Using 6h timeframe for execution reduces noise
while maintaining sufficient trade frequency. Daily trend filter (price vs Kumo) avoids counter-trend trades.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 8:
            period9_high[i] = np.max(high_1d[i-8:i+1])
            period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 25:
            period26_high[i] = np.max(high_1d[i-25:i+1])
            period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 51:
            period52_high[i] = np.max(high_1d[i-51:i+1])
            period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward by 26 periods
    # For trend filter, we use current cloud (Senkou Span A/B)
    # Kumo top = max(Senkou Span A, Senkou Span B)
    # Kumo bottom = min(Senkou Span A, Senkou Span B)
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_up = np.zeros(len(tenkan_sen_aligned), dtype=bool)
    tk_cross_down = np.zeros(len(tenkan_sen_aligned), dtype=bool)
    for i in range(1, len(tenkan_sen_aligned)):
        if (not np.isnan(tenkan_sen_aligned[i-1]) and not np.isnan(kijun_sen_aligned[i-1]) and
            not np.isnan(tenkan_sen_aligned[i]) and not np.isnan(kijun_sen_aligned[i])):
            if tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                tk_cross_up[i] = True
            if tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                tk_cross_down[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure Ichimoku is fully calculated
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Kumo AND TK cross up AND price > Kumo (uptrend)
            if (close[i] > kumo_top[i] and 
                tk_cross_up[i] and 
                close[i] > kumo_bottom[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Kumo AND TK cross down AND price < Kumo (downtrend)
            elif (close[i] < kumo_bottom[i] and 
                  tk_cross_down[i] and 
                  close[i] < kumo_top[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Kumo bottom OR TK cross down
            if close[i] < kumo_bottom[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Kumo top OR TK cross up
            if close[i] > kumo_top[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals