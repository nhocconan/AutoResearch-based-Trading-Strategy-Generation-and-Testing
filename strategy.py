#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
# Hypothesis: Ichimoku Tenkan/Kijun cross with 1d cloud filter on 6h timeframe.
# Uses 1d Tenkan-sen/Kijun-sen for cloud calculation to filter trades by higher timeframe trend.
# Long when price > cloud and TK cross bullish; short when price < cloud and TK cross bearish.
# Ichimoku cloud acts as dynamic support/resistance, reducing whipsaw in ranging markets.
# Works in bull/bear markets by following 1d trend via cloud position.
# Targets low trade frequency with high-conviction signals.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
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

    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)

    # Calculate 6h Tenkan/Kijun for crossover signal
    max_high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (max_high_9_6h + min_low_9_6h) / 2

    max_high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    min_low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (max_high_26_6h + min_low_26_6h) / 2

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Senkou B calculation window
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])

        if position == 0:
            # LONG: Price above cloud and bullish TK cross
            if (close[i] > cloud_top and 
                tenkan_6h[i] > kijun_6h[i] and 
                tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud and bearish TK cross
            elif (close[i] < cloud_bottom and 
                  tenkan_6h[i] < kijun_6h[i] and 
                  tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud or bearish TK cross
            if close[i] < cloud_bottom or (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud or bullish TK cross
            if close[i] > cloud_top or (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals