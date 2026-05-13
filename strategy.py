#!/usr/bin/env python3
# 6h_IchimokuCloud_Trend_1dFilter
# Hypothesis: Ichimoku cloud (Tenkan/Kijun/Senkou Span A/B) from 1d timeframe provides robust trend direction.
# Price above/below cloud + TK cross confirms momentum in direction of higher timeframe trend.
# Works in bull/bear markets: cloud acts as dynamic support/resistance, TK cross captures momentum shifts.
# Low frequency: trades only when price interacts with cloud + TK cross, targeting ~20-50 trades/year.

name = "6h_IchimokuCloud_Trend_1dFilter"
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

    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')

    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2

    # Align Ichimoku components to 6h timeframe (wait for 1d close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)

    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = np.zeros(n, dtype=bool)
    tk_cross_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if (tenkan_6h[i-1] <= kijun_6h[i-1] and tenkan_6h[i] > kijun_6h[i]):
            tk_cross_up[i] = True
        elif (tenkan_6h[i-1] >= kijun_6h[i-1] and tenkan_6h[i] < kijun_6h[i]):
            tk_cross_down[i] = True

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Senkou B lookback
        # Skip if any required value is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
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
            # LONG: Price above cloud + TK cross up
            if close[i] > cloud_top and tk_cross_up[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + TK cross down
            elif close[i] < cloud_bottom and tk_cross_down[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud or TK cross down
            if close[i] < cloud_top or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud or TK cross up
            if close[i] > cloud_bottom or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals