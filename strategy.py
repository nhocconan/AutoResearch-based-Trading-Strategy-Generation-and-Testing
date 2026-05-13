#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Sen_Bounce_Trend_Filter
Hypothesis: In 6h timeframe, price bouncing off Kijun-sen (26-period) with Tenkan-sen/Kijun-sen cross alignment and 1d trend filter provides institutional-grade support/resistance. Works in bull/bear as Ichimoku adapts to volatility. Target: 15-35 trades/year.
"""

name = "6h_Ichimoku_Kijun_Sen_Bounce_Trend_Filter"
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
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = np.zeros(n)
    period9_low = np.zeros(n)
    for i in range(n):
        if i < 8:
            period9_high[i] = np.nan
            period9_low[i] = np.nan
        else:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = np.zeros(n)
    period26_low = np.zeros(n)
    for i in range(n):
        if i < 25:
            period26_high[i] = np.nan
            period26_low[i] = np.nan
        else:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = np.zeros(n)
    period52_low = np.zeros(n)
    for i in range(n):
        if i < 51:
            period52_high[i] = np.nan
            period52_low[i] = np.nan
        else:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A/B shifted 26 periods ahead
    # For cloud at time t, we need values from t-26
    senkou_span_a_lagged = np.full(n, np.nan)
    senkou_span_b_lagged = np.full(n, np.nan)
    for i in range(26, n):
        senkou_span_a_lagged[i] = senkou_span_a[i-26]
        senkou_span_b_lagged[i] = senkou_span_b[i-26]
    
    # Kumo top and bottom
    kumo_top = np.where(senkou_span_a_lagged > senkou_span_b_lagged, senkou_span_a_lagged, senkou_span_b_lagged)
    kumo_bottom = np.where(senkou_span_a_lagged < senkou_span_b_lagged, senkou_span_a_lagged, senkou_span_b_lagged)
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen and Kijun-sen
    period9_high_1d = np.zeros(len(df_1d))
    period9_low_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 8:
            period9_high_1d[i] = np.nan
            period9_low_1d[i] = np.nan
        else:
            period9_high_1d[i] = np.max(high_1d[i-8:i+1])
            period9_low_1d[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = np.zeros(len(df_1d))
    period26_low_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 25:
            period26_high_1d[i] = np.nan
            period26_low_1d[i] = np.nan
        else:
            period26_high_1d[i] = np.max(high_1d[i-25:i+1])
            period26_low_1d[i] = np.min(low_1d[i-25:i+1])
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d trend: price above/below Kijun-sen
    uptrend_1d = close_1d > kijun_sen_1d
    downtrend_1d = close_1d < kijun_sen_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if any values are NaN
        if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        tk_cross = tenkan > kijun  # Tenkan above Kijun
        price_above_kumo = price > kumo_top_val
        price_below_kumo = price < kumo_bottom_val
        price_in_kumo = (price >= kumo_bottom_val) and (price <= kumo_top_val)
        
        if position == 0:
            # LONG: price bounces off Kijun-sen from below, TK cross bullish, 1d uptrend
            if (abs(price - kijun) < 0.001 * price and price > kijun and  # bounce off Kijun
                tk_cross and 
                uptrend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price bounces off Kijun-sen from above, TK cross bearish, 1d downtrend
            elif (abs(price - kijun) < 0.001 * price and price < kijun and  # bounce off Kijun
                  not tk_cross and 
                  downtrend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Kijun-sen or TK cross turns bearish
            if price < kijun or not tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Kijun-sen or TK cross turns bullish
            if price > kijun or tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals