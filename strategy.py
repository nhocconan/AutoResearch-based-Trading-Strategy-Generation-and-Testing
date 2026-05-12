#!/usr/bin/env python3
# 6h_IchimokuCloud_1dTrend_TenkanKijunCross
# Hypothesis: Use Ichimoku cloud from 1d for trend filter and support/resistance, with Tenkan/Kijun cross on 6h for entry.
# In bullish trend (price above 1d cloud), go long on Tenkan-Kijun cross above; in bearish trend (price below 1d cloud), go short on cross below.
# Exit when price crosses opposite Kumo edge or Tenkan/Kijun reverses.
# Designed for low frequency (12-30 trades/year) by using 1d for trend and 6h for precise entry timing.

name = "6h_IchimokuCloud_1dTrend_TenkanKijunCross"
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
    
    # === 1d data for Ichimoku cloud (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                     pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() +
                    pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Kumo (Cloud) edges: Senkou Span A and B shifted forward by 26 periods
    # For trend filter, we use current cloud (Senkou Span A/B from 26 periods ago)
    senkou_span_a_lagged = np.roll(senkou_span_a_1d, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b_1d, 26)
    # Fill NaN from roll
    senkou_span_a_lagged[:26] = senkou_span_a_1d[0]
    senkou_span_b_lagged[:26] = senkou_span_b_1d[0]
    
    # Kumo top and bottom
    kumo_top_1d = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    kumo_bottom_1d = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    # Align 1d Ichimoku components to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # === 6d data for Tenkan/Kijun cross (entry signal) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < kijun_period:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen and Kijun-sen on 6h
    tenkan_sen_6h = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                     pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_6h = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() +
                    pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, kijun_period)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d cloud
        price_above_kumo = close[i] > kumo_top_1d_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_1d_aligned[i]
        
        # Tenkan-Kijun cross on 6h
        tk_cross_above = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_below = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        if position == 0:
            # LONG: Price above cloud and Tenkan crosses above Kijun
            if price_above_kumo and tk_cross_above:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud and Tenkan crosses below Kijun
            elif price_below_kumo and tk_cross_below:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below Kumo bottom or Tenkan crosses below Kijun
            if close[i] < kumo_bottom_1d_aligned[i] or tk_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Kumo top or Tenkan crosses above Kijun
            if close[i] > kumo_top_1d_aligned[i] or tk_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals