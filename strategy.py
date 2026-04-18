#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1DTrend_Filter
Hypothesis: Use Ichimoku cloud on 6h for entry timing and 1d trend filter (price above/below Kumo) to capture major trends.
In bull markets: price above daily Kumo + Tenkan-Kijun cross up → long.
In bear markets: price below daily Kumo + Tenkan-Kijun cross down → short.
Kumo acts as dynamic support/resistance, reducing whipsaw. Low trade frequency expected.
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
    
    # Ichimoku parameters
    tenkan = 9
    kijun = 26
    senkou = 52
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full_like(high, np.nan)
    for i in range(tenkan - 1, len(high)):
        tenkan_sen[i] = (np.max(high[i - tenkan + 1:i + 1]) + np.min(low[i - tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full_like(high, np.nan)
    for i in range(kijun - 1, len(high)):
        kijun_sen[i] = (np.max(high[i - kijun + 1:i + 1]) + np.min(low[i - kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(high, np.nan)
    for i in range(len(high)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + kijun
            if idx < len(high):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full_like(high, np.nan)
    for i in range(senkou - 1, len(high)):
        senkou_span_b[i + kijun] = (np.max(high[i - senkou + 1:i + 1]) + np.min(low[i - senkou + 1:i + 1])) / 2
    
    # Get daily trend filter: price relative to daily Kumo
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Kumo (Ichimoku on daily)
    tenkan_1d = 9
    kijun_1d = 26
    senkou_1d = 52
    
    tenkan_sen_1d = np.full_like(high_1d, np.nan)
    for i in range(tenkan_1d - 1, len(high_1d)):
        tenkan_sen_1d[i] = (np.max(high_1d[i - tenkan_1d + 1:i + 1]) + np.min(low_1d[i - tenkan_1d + 1:i + 1])) / 2
    
    kijun_sen_1d = np.full_like(high_1d, np.nan)
    for i in range(kijun_1d - 1, len(high_1d)):
        kijun_sen_1d[i] = (np.max(high_1d[i - kijun_1d + 1:i + 1]) + np.min(low_1d[i - kijun_1d + 1:i + 1])) / 2
    
    senkou_span_a_1d = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tenkan_sen_1d[i]) and not np.isnan(kijun_sen_1d[i]):
            idx = i + kijun_1d
            if idx < len(high_1d):
                senkou_span_a_1d[idx] = (tenkan_sen_1d[i] + kijun_sen_1d[i]) / 2
    
    senkou_span_b_1d = np.full_like(high_1d, np.nan)
    for i in range(senkou_1d - 1, len(high_1d)):
        senkou_span_b_1d[i + kijun_1d] = (np.max(high_1d[i - senkou_1d + 1:i + 1]) + np.min(low_1d[i - senkou_1d + 1:i + 1])) / 2
    
    # Align daily Kumo to 6h timeframe
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    kumo_bottom_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    kumo_top_6h = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_6h = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Align Ichimoku lines to 6h (they are already on 6h, but ensure alignment)
    # For safety, we'll align them too (though they should be fine)
    tenkan_sen_6h = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}, index=prices.index), tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}, index=prices.index), kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}, index=prices.index), senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}, index=prices.index), senkou_span_b)
    
    # Kumo on 6h
    kumo_top_6h_self = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    kumo_bottom_6h_self = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan, kijun, senkou) + kijun + 10  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(kumo_top_6h[i]) or np.isnan(kumo_bottom_6h[i])):
            signals[i] = 0.0
            continue
        
        # Kumo twist: Kumo is bullish when Senkou A > Senkou B
        kumo_bullish = senkou_span_a_6h[i] > senkou_span_b_6h[i]
        kumo_bearish = senkou_span_a_6h[i] < senkou_span_b_6h[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumo_top_6h_self[i]
        price_below_kumo = close[i] < kumo_bottom_6h_self[i]
        
        # Tenkan-Kijun cross
        tk_cross_up = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_down = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        # Daily trend filter: price relative to daily Kumo
        price_above_daily_kumo = close[i] > kumo_top_6h[i]
        price_below_daily_kumo = close[i] < kumo_bottom_6h[i]
        
        if position == 0:
            # Long: bullish Kumo + TK cross up + price above daily Kumo
            if kumo_bullish and tk_cross_up and price_above_daily_kumo:
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo + TK cross down + price below daily Kumo
            elif kumo_bearish and tk_cross_down and price_below_daily_kumo:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Kumo OR TK cross down
            if close[i] < kumo_bottom_6h_self[i] or tk_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Kumo OR TK cross up
            if close[i] > kumo_top_6h_self[i] or tk_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1DTrend_Filter"
timeframe = "6h"
leverage = 1.0