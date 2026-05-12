#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_1dCloud_Filter
# Hypothesis: Use Ichimoku Tenkan-Kijun cross for entry signals on 6h timeframe,
# filtered by the 1d Ichimoku cloud to ensure trades align with higher timeframe trend.
# Cloud acts as dynamic support/resistance: only go long when price above cloud,
# short when price below cloud. This reduces false signals in choppy markets.
# TK cross provides timely entries while cloud filter adds trend context.
# Designed for 15-35 trades/year per symbol with discrete sizing to minimize fee drag.

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
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

    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)

    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)

    # Calculate 6h Tenkan and Kijun for entry signals
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_fast = (high_9_6h + low_9_6h) / 2

    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_fast = (high_26_6h + low_26_6h) / 2

    # TK cross signals: bullish when Tenkan crosses above Kijun
    tk_cross_up = (tenkan_6h_fast > kijun_6h_fast) & (tenkan_6h_fast[1] <= kijun_6h_fast[1]) if n > 1 else False
    tk_cross_down = (tenkan_6h_fast < kijun_6h_fast) & (tenkan_6h_fast[1] >= kijun_6h_fast[1]) if n > 1 else False
    # Vectorized version for loop
    tk_cross_up_arr = np.zeros(n, dtype=bool)
    tk_cross_down_arr = np.zeros(n, dtype=bool)
    for i in range(1, n):
        tk_cross_up_arr[i] = (tenkan_6h_fast[i] > kijun_6h_fast[i]) and (tenkan_6h_fast[i-1] <= kijun_6h_fast[i-1])
        tk_cross_down_arr[i] = (tenkan_6h_fast[i] < kijun_6h_fast[i]) and (tenkan_6h_fast[i-1] >= kijun_6h_fast[i-1])

    # Cloud boundaries: Senkou Span A and B form the cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross_up_arr[i]) or np.isnan(tk_cross_down_arr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Cloud filter: only long when price above cloud, short when price below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]

        if position == 0:
            # LONG: TK cross up AND price above cloud
            if tk_cross_up_arr[i] and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross down AND price below cloud
            elif tk_cross_down_arr[i] and price_below_cloud:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross down OR price drops below cloud
            if tk_cross_down_arr[i] or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross up OR price rises above cloud
            if tk_cross_up_arr[i] or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals