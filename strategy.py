#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Twist_Filter
# Hypothesis: Use Ichimoku cloud (TK cross + cloud color) from 1d timeframe filtered by weekly trend (price above/below weekly Kumo). Enter long when TK crosses above cloud in bullish weekly regime, short when TK crosses below cloud in bearish weekly regime. Exit when TK reverses back into cloud. Weekly trend filter reduces false signals in chop. Targets 20-40 trades/year on 6h with disciplined entries to avoid fee drag. Works in bull/bear via adaptive regime filter.

name = "6h_Ichimoku_Cloud_Twist_Filter"
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

    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2.0
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2.0
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2.0
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2.0

    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)

    # Weekly Kumo (cloud) for trend filter
    # Senkou Span A and B from 1w data (shifted forward 26 periods)
    span_a_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2.0
    span_b_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2.0
    # Align weekly cloud to 6h
    span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, span_a_1w.values)
    span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, span_b_1w.values)

    # Determine cloud color (green if Span A > Span B, red otherwise)
    # For entry: price above cloud = bullish, below cloud = bearish
    # We'll use weekly cloud color as regime filter

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # start after Ichimoku lookback
        # Skip if any required value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(span_a_1w_aligned[i]) or np.isnan(span_b_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # TK cross signals
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]

        # Price relative to cloud
        price_above_cloud = close[i] > max(span_a_aligned[i], span_b_aligned[i])
        price_below_cloud = close[i] < min(span_a_aligned[i], span_b_aligned[i])
        price_in_cloud = not (price_above_cloud or price_below_cloud)

        # Weekly regime: bullish if price above weekly cloud, bearish if below
        weekly_bullish = close[i] > max(span_a_1w_aligned[i], span_b_1w_aligned[i])
        weekly_bearish = close[i] < min(span_a_1w_aligned[i], span_b_1w_aligned[i])

        if position == 0:
            # LONG: TK cross above + price above cloud + weekly bullish regime
            if tk_cross_above and price_above_cloud and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross below + price below cloud + weekly bearish regime
            elif tk_cross_below and price_below_cloud and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross back below OR price re-enters cloud
            if tk_cross_below or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross back above OR price re-enters cloud
            if tk_cross_above or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals