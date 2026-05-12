# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# 6h_1D_Ichimoku_TK_Cross_Cloud_Filter
# Hypothesis: Use daily Ichimoku (conversion, base, spanA/B) as filter for 6h TK crosses.
# Long: TK crosses above base AND price above cloud (bullish). Short: TK crosses below base AND price below cloud (bearish).
# Ichimoku cloud from daily timeframe provides robust trend/filter to avoid whipsaws in sideways markets.
# Designed for low frequency (12-30 trades/year) to minimize fee drag on 6h timeframe.
# Works in both bull/bear via cloud filter: only trade in direction of higher timeframe trend.

name = "6h_1D_Ichimoku_TK_Cross_Cloud_Filter"
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
    volume = prices['volume'].values

    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Conversion Line (tenkan-sen): (9-period high + low)/2
    period_conv = 9
    high_conv = pd.Series(high_1d).rolling(window=period_conv, min_periods=period_conv).max().values
    low_conv = pd.Series(low_1d).rolling(window=period_conv, min_periods=period_conv).min().values
    tenkan = (high_conv + low_conv) / 2

    # Base Line (kijun-sen): (26-period high + low)/2
    period_base = 26
    high_base = pd.Series(high_1d).rolling(window=period_base, min_periods=period_base).max().values
    low_base = pd.Series(low_1d).rolling(window=period_base, min_periods=period_base).min().values
    kijun = (high_base + low_base) / 2

    # Leading Span A (senkou span A): (Conversion + Base)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Leading Span B (senkou span B): (52-period high + low)/2 shifted 26 periods ahead
    period_span_b = 52
    high_span_b = pd.Series(high_1d).rolling(window=period_span_b, min_periods=period_span_b).max().values
    low_span_b = pd.Series(low_1d).rolling(window=period_span_b, min_periods=period_span_b).min().values
    senkou_b = ((high_span_b + low_span_b) / 2)

    # Align Ichimoku components to 6h timeframe (with proper look-ahead prevention)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)

    # Calculate TK cross on 6d data
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    # Handle first element
    tk_cross_above[0] = False
    tk_cross_below[0] = False

    # Cloud: price above both spans = bullish, below both = bearish
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        bullish_setup = tk_cross_above[i] and price_above_cloud[i]
        bearish_setup = tk_cross_below[i] and price_below_cloud[i]

        if position == 0:
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross below base OR price drops below cloud
            if tk_cross_below[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross above base OR price rises above cloud
            if tk_cross_above[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals