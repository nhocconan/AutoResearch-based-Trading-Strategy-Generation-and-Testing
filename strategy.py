#!/usr/bin/env python3
# 6h_Ichimoku_Kumo_Twist_Trend_1d
# Hypothesis: Use Ichimoku Cloud on daily timeframe for trend direction (price above/below cloud) combined with Tenkan/Kijun cross on 6h for entry timing. 
# In bull markets: price above daily cloud + TK cross up = long. In bear markets: price below daily cloud + TK cross down = short.
# The daily cloud acts as a strong trend filter that works in both bull/bear regimes, while TK cross provides timely entries.
# Targets 15-30 trades/year to minimize fee drag.

name = "6h_Ichimoku_Kumo_Twist_Trend_1d"
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

    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Ichimoku components on daily
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)

    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)

    # Calculate TK cross on 6h for entry timing
    # Tenkan-sen (Conversion Line) on 6h
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2

    # Kijun-sen (Base Line) on 6h
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2

    # TK cross signals
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(26, n):
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and trend
        senkou_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        senkou_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom

        if position == 0:
            # LONG: Price above daily cloud + TK cross up on 6h
            if price_above_cloud and tk_cross_up[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below daily cloud + TK cross down on 6h
            elif price_below_cloud and tk_cross_down[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below daily cloud OR TK cross down
            if price_below_cloud or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above daily cloud OR TK cross up
            if price_above_cloud or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals