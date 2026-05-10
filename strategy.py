#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Filter_1dTrend
# Hypothesis: Ichimoku cloud on 1d provides strong trend filter (price above/below cloud).
# TK cross on 6h gives entry timing with trend alignment. Cloud acts as dynamic support/resistance.
# Works in bull via buying dips in uptrend (price > cloud + TK cross up).
# Works in bear via selling rallies in downtrend (price < cloud + TK cross down).
# Low trade frequency expected due to dual timeframe confirmation.

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
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
    
    # Get daily data for Ichimoku cloud (base on previous day to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = low_1d = df_1d['low'].values
    
    # Ichimoku components (using standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    # Tenkan-sen (9-period)
    tenkan_sen = (rolling_max(high_1d, 9) + rolling_min(low_1d, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (rolling_max(high_1d, 26) + rolling_min(low_1d, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (rolling_max(high_1d, 52) + rolling_min(low_1d, 52)) / 2
    
    # Align Ichimoku components to 6h timeframe (no shift needed as align_htf_to_ltf handles it)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate Kumo (cloud) top and bottom
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    # Using previous values to avoid look-ahead (already handled by align)
    kumO_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    kumO_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # TK Cross on 6h (using current 6h data)
    # Tenkan-sen on 6h
    tenkan_sen_6h_internal = (rolling_max(high, 9) + rolling_min(low, 9)) / 2
    # Kijun-sen on 6h
    kijun_sen_6h_internal = (rolling_max(high, 26) + rolling_min(low, 26)) / 2
    
    # TK Cross signals: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_sen_6h_internal > kijun_sen_6h_internal) & \
                  (np.roll(tenkan_sen_6h_internal, 1) <= np.roll(kijun_sen_6h_internal, 1))
    tk_cross_down = (tenkan_sen_6h_internal < kijun_sen_6h_internal) & \
                    (np.roll(tenkan_sen_6h_internal, 1) >= np.roll(kijun_sen_6h_internal, 1))
    
    # Handle first element for roll
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26) + 1  # need enough history for Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(kumO_top[i]) or np.isnan(kumO_bottom[i]) or \
           np.isnan(tenkan_sen_6h_internal[i]) or np.isnan(kijun_sen_6h_internal[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d cloud: price above cloud = uptrend, below = downtrend
        price_above_cloud = close[i] > kumO_top[i]
        price_below_cloud = close[i] < kumO_bottom[i]
        
        if position == 0:
            # Long: price above cloud (uptrend) + TK cross up
            if price_above_cloud and tk_cross_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud (downtrend) + TK cross down
            elif price_below_cloud and tk_cross_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud bottom OR TK cross down
            if close[i] < kumO_bottom[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud top OR TK cross up
            if close[i] > kumO_top[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals