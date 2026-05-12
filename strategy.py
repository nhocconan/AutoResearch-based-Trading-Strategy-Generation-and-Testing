#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend
Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen AND price is above Kumo (cloud) with 1d EMA50 trending up; enter short when Tenkan-sen crosses below Kijun-sen AND price is below Kumo with 1d EMA50 trending down. Uses Ichimoku cloud as dynamic support/resistance and trend filter to capture momentum with controlled risk. Targets 20-50 trades per year to reduce fee drift.
"""

name = "6h_Ichimoku_Kumo_Twist_1dTrend"
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

    # Get 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou Span B
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0

    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)

    # Kumo (cloud) boundaries: Senkou Span A and B shifted forward by 26 periods
    # For entry signals, we use current cloud (not shifted)
    # Upper cloud boundary: max(Senkou Span A, Senkou Span B)
    # Lower cloud boundary: min(Senkou Span A, Senkou Span B)
    kumomax = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumomin = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumomax[i]) or np.isnan(kumomin[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Tenkan-sen crosses above Kijun-sen AND price above Kumo AND 1d uptrend
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # Cross just happened
                close[i] > kumomax[i] and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan-sen crosses below Kijun-sen AND price below Kumo AND 1d downtrend
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # Cross just happened
                  close[i] < kumomin[i] and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan-sen crosses below Kijun-sen OR price falls below Kumo
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               close[i] < kumomax[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan-sen crosses above Kijun-sen OR price rises above Kumo
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               close[i] > kumomin[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals