#!/usr/bin/env python3
"""
6h_IchiCloud_1dTrend_WeeklyTrend
Hypothesis: Ichimoku cloud on 1d (Tenkan/Kijun + Cloud) filters 6h entries aligned with 1d and 1w trend.
Long when price above cloud + Tenkan > Kijun (bullish TK cross) + 1d/1w uptrend.
Short when price below cloud + Tenkan < Kijun (bearish TK cross) + 1d/1w downtrend.
Uses Ichimoku as multi-condition filter to reduce whipsaws in both bull and bear markets.
"""

name = "6h_IchiCloud_1dTrend_WeeklyTrend"
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

    # Get 1d data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Ichimoku components on 1d: Tenkan (9), Kijun (26), Senkou A/B (52)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2

    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2

    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B: (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2

    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Need 52 for Senkou B
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])

        if position == 0:
            # LONG: Price above cloud + bullish TK cross + 1w uptrend
            if close[i] > cloud_top and tenkan_6h[i] > kijun_6h[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + bearish TK cross + 1w downtrend
            elif close[i] < cloud_bottom and tenkan_6h[i] < kijun_6h[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below cloud OR bearish TK cross OR 1w downtrend
            if close[i] < cloud_top or tenkan_6h[i] < kijun_6h[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above cloud OR bullish TK cross OR 1w uptrend
            if close[i] > cloud_bottom or tenkan_6h[i] > kijun_6h[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals