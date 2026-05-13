#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Follow_With_1d_Trend_Filter
Hypothesis: Ichimoku system on 6h provides robust trend signals, with 1d trend filter ensuring alignment with higher timeframe momentum. Tenkan/Kijun cross with price above/below cloud acts as entry, while Senkou Span B acts as dynamic stop. Works in both bull and bear markets by following 1d trend direction.
"""

name = "6h_Ichimoku_Cloud_Trend_Follow_With_1d_Trend_Filter"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)

    # Senkou Span B (Leading Span B): (52-period high + low) / 2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if any required value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])

        if position == 0:
            # LONG: Price above cloud, Tenkan > Kijun, and 1d uptrend
            if close[i] > cloud_top and tenkan[i] > kijun[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud, Tenkan < Kijun, and 1d downtrend
            elif close[i] < cloud_bottom and tenkan[i] < kijun[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud or Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud or Tenkan > Kijun
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals