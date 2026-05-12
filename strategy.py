#!/usr/bin/env python3
# 6h_1D_Ichimoku_TK_Cross_CloudFilter_Trend
# Hypothesis: Ichimoku Tenkan-Kijun cross on 1d with cloud filter and 60-period 6h EMA trend.
# Long when Tenkan > Kijun and price > Kumo cloud (bullish) and 6h EMA60 uptrend.
# Short when Tenkan < Kijun and price < Kumo cloud (bearish) and 6h EMA60 downtrend.
# Cloud filter from 1d ensures we trade in strong trends; TK cross provides timely entry.
# Works in bull/bear: uses both long and short conditions with trend alignment.
# Target: 50-150 trades over 4 years (~12-37/year) with size 0.25.

name = "6h_1D_Ichimoku_TK_Cross_CloudFilter_Trend"
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

    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Ichimoku components (9, 26, 52 periods)
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2

    # Senkou Span A and B
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)

    # Kumo cloud edges (top and bottom)
    kumomax = np.maximum(senkou_a, senkou_b)
    kumomin = np.minimum(senkou_a, senkou_b)

    # Align Ichimoku to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    kumomax_aligned = align_htf_to_ltf(prices, df_1d, kumomax.values)
    kumomin_aligned = align_htf_to_ltf(prices, df_1d, kumomin.values)

    # 6h EMA60 trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(kumomax_aligned[i]) or np.isnan(kumomin_aligned[i]) or
            np.isnan(ema_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Ichimoku conditions
        bullish_cross = tenkan_aligned[i] > kijun_aligned[i]
        bearish_cross = tenkan_aligned[i] < kijun_aligned[i]
        price_above_cloud = close[i] > kumomax_aligned[i]
        price_below_cloud = close[i] < kumomin_aligned[i]
        uptrend = close[i] > ema_60[i]
        downtrend = close[i] < ema_60[i]

        if position == 0:
            # LONG: Bullish TK cross + price above cloud + uptrend
            if bullish_cross and price_above_cloud and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish TK cross + price below cloud + downtrend
            elif bearish_cross and price_below_cloud and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish TK cross or price below cloud
            if bearish_cross or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish TK cross or price above cloud
            if bullish_cross or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals