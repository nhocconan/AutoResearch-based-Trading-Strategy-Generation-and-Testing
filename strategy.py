#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filtered_TK_Cross
Hypothesis: Use daily Ichimoku cloud (conversion line, base line, leading spans) as trend filter on 6h chart. Enter long when Tenkan-sen crosses above Kijun-sen AND price is above cloud (bullish), enter short when Tenkan-sen crosses below Kijun-sen AND price is below cloud (bearish). Add volume confirmation to avoid false signals. Designed for 15-35 trades/year on 6h timeframe to work in both bull and bear markets by using cloud as dynamic support/resistance and TK cross as momentum signal.
"""

name = "6h_Ichimoku_Cloud_Filtered_TK_Cross"
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

    # Get daily data for Ichimoku (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Ichimoku parameters: 9, 26, 52
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2

    # Align Ichimoku components to 6h timeframe (values from previous day's close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)

    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a_val = senkou_span_a_aligned[i]
        span_b_val = senkou_span_b_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(span_a_val) or np.isnan(span_b_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Cloud top and bottom
        cloud_top = max(span_a_val, span_b_val)
        cloud_bottom = min(span_a_val, span_b_val)

        if position == 0:
            # LONG: TK cross bullish + price above cloud + volume confirmation
            if (tenkan_val > kijun_val and 
                close[i] > cloud_top and 
                volume[i] > vol_avg_val * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish + price below cloud + volume confirmation
            elif (tenkan_val < kijun_val and 
                  close[i] < cloud_bottom and 
                  volume[i] > vol_avg_val * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish OR price drops below cloud
            if tenkan_val < kijun_val or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish OR price rises above cloud
            if tenkan_val > kijun_val or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals