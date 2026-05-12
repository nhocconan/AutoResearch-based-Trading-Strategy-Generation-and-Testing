#!/usr/bin/env python3

# 6h_1W_Ichimoku_TK_Cross_Cloud_Trend
# Hypothesis: Use Ichimoku cloud from weekly timeframe for trend filter, with Tenkan-Kijun cross on 6h for entry.
# Weekly cloud provides strong trend context (price above/below cloud), reducing whipsaws in sideways markets.
# Tenkan-Kijun cross acts as momentum signal within the trend. Designed for low frequency (15-30 trades/year) to minimize fee drift.
# Works in bull markets (trend + momentum alignment) and bear markets (avoids counter-trend signals via cloud filter).

name = "6h_1W_Ichimoku_TK_Cross_Cloud_Trend"
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

    # Get weekly data for Ichimoku cloud
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen = tenkan_sen.values

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen = kijun_sen.values

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                     pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = senkou_span_b.values

    # Chikou Span (Lagging Span): close plotted 26 periods back (not used for filtering)

    # Align Ichimoku components to 6h timeframe (already weekly, so aligns to weekly close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)

    # Calculate 6h Tenkan-Kijun cross for entry signal
    tenkan_6h = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                 pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() +
                pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    tenkan_6h = tenkan_6h.values
    kijun_6h = kijun_6h.values

    # Bullish cross: Tenkan crosses above Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h[:-1] <= kijun_6h[:-1])
    # Bearish cross: Tenkan crosses below Kijun
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h[:-1] >= kijun_6h[:-1])
    # Prepend False for first element
    tk_cross_up = np.concatenate([[False], tk_cross_up])
    tk_cross_down = np.concatenate([[False], tk_cross_down])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Wait for Senkou Span B to be valid
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        bullish_trend = close[i] > upper_cloud
        bearish_trend = close[i] < lower_cloud

        if position == 0:
            # LONG: Price above cloud + bullish TK cross
            if bullish_trend and tk_cross_up[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + bearish TK cross
            elif bearish_trend and tk_cross_down[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below cloud or bearish TK cross
            if close[i] < upper_cloud or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above cloud or bullish TK cross
            if close[i] > lower_cloud or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals