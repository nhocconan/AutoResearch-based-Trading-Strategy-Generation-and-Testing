#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Breakout_1dTrend
Hypothesis: On 6h timeframe, buy when price breaks above Kumo (cloud) from 1d Ichimoku with Tenkan/Kijun cross bullish; sell when price breaks below Kumo with Tenkan/Kijun cross bearish. Uses 1d Ichimoku for trend direction and cloud support/resistance, reducing whipsaws in both bull and bear markets. Targets 15-35 trades per year to minimize fee drag.
"""

name = "6h_Ichimoku_Kumo_Breakout_1dTrend"
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

    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2

    # Align Ichimoku components to 6h timeframe (with proper delay for forward-looking spans)
    # Senkou spans are plotted 26 periods ahead, so we need to shift them back for alignment
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)

    # Kumo (cloud) boundaries: Senkou Span A and B
    # Kumo top is the higher of Senkou A and B, Kumo bottom is the lower
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after sufficient warmup for Ichimoku
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Kumo AND Tenkan > Kijun (bullish cross)
            if (close[i] > kumo_top[i] and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Kumo AND Tenkan < Kijun (bearish cross)
            elif (close[i] < kumo_bottom[i] and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Kumo OR Tenkan/Kijun cross turns bearish
            if close[i] < kumo_bottom[i] or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Kumo OR Tenkan/Kijun cross turns bullish
            if close[i] > kumo_top[i] or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals