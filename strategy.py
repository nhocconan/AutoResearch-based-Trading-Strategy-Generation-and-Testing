#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_Filter
Hypothesis: Price breaking above/below the Ichimoku Cloud (Senkou Span A/B) on 6h timeframe, with weekly trend filter (price above/below weekly Kumo) and volume confirmation (1.5x average), captures strong momentum moves while avoiding false breakouts. Ichimoku provides dynamic support/resistance and trend direction, making it effective in both bull and bear markets when combined with higher timeframe trend filter.
"""

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_Filter"
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
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2

    # Get weekly Ichimoku for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Weekly Tenkan-sen and Kijun-sen
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1w = (high_9_1w + low_9_1w) / 2

    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen_1w = (high_26_1w + low_26_1w) / 2

    # Weekly Senkou Span A and B
    senkou_span_a_1w = (tenkan_sen_1w + kijun_sen_1w) / 2
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1w = (high_52_1w + low_52_1w) / 2

    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, prices, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, prices, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)

    # Align weekly Ichimoku components to 6h timeframe
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)

    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Senkou Span B warmup
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(senkou_span_a_1w_aligned[i]) or np.isnan(senkou_span_b_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries (future cloud)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])

        # Determine weekly cloud boundaries
        weekly_upper_cloud = np.maximum(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])
        weekly_lower_cloud = np.minimum(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])

        if position == 0:
            # LONG: Price breaks above cloud + price above weekly cloud + TK cross bullish + volume spike
            if (close[i] > upper_cloud and 
                close[i] > weekly_upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + price below weekly cloud + TK cross bearish + volume spike
            elif (close[i] < lower_cloud and 
                  close[i] < weekly_lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud
            if close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud
            if close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals