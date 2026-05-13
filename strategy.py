#!/usr/bin/env python3
# 6h_IKI_Pulse: Ichimoku Kinetic Index Pulse
# Hypothesis: Use 12h Ichimoku Cloud (Tenkan/Kijun cross) as trend filter with 1d Kumo twist for bias.
# Enter long when price breaks above Kumo (cloud) with TK cross bullish and Kumo twist bullish.
# Enter short when price breaks below Kumo with TK cross bearish and Kumo twist bearish.
# Exit when price re-enters Kumo or TK cross reverses.
# Uses Ichimoku's multi-line structure to filter whipsaws in both bull and bear markets.
# Target: 15-25 trades/year on 6h to avoid fee drag while capturing strong trends.

name = "6h_IKI_Pulse"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0

    # Kumo twist: Senkou A > Senkou B = bullish twist, else bearish
    kumotwist_bullish = senkou_a > senkou_b

    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    kumotwist_aligned = align_htf_to_ltf(prices, df_12h, kumotwist_bullish.astype(float))

    # Kumo cloud boundaries (top and bottom of cloud)
    kumotop = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumobottom = np.minimum(senkou_a_aligned, senkou_b_aligned)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):
        # Skip if any required value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumotop[i]) or np.isnan(kumobottom[i]) or 
            np.isnan(kumotwist_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above cloud + TK cross bullish + Kumo twist bullish
            if (close[i] > kumotop[i] and 
                tenkan_aligned[i] > kijun_aligned[i] and
                kumotwist_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + TK cross bearish + Kumo twist bearish
            elif (close[i] < kumobottom[i] and 
                  tenkan_aligned[i] < kijun_aligned[i] and
                  kumotwist_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud OR TK cross turns bearish
            if (close[i] <= kumotop[i] and close[i] >= kumobottom[i]) or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud OR TK cross turns bullish
            if (close[i] <= kumotop[i] and close[i] >= kumobottom[i]) or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals