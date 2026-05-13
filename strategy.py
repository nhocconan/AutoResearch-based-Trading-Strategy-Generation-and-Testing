#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Twist_Filter
# Hypothesis: Ichimoku Cloud twist (Tenkan/Kijun cross) combined with daily Cloud color filter
# provides high-probability trend entries. Green cloud (SenkouA > SenkouB) favors longs,
# red cloud favors shorts. TK cross acts as momentum trigger. Works in both bull/bear
# markets by following higher timeframe trend via cloud color. Designed for low-frequency,
# high-conviction trades to minimize fee drag on 6H timeframe.

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
    volume = prices['volume'].values

    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0

    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0

    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0

    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but required for cloud calculation integrity

    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)

    # Cloud color: Green (bullish) when SenkouA > SenkouB, Red (bearish) when SenkouA < SenkouB
    cloud_green = senkou_a_aligned > senkou_b_aligned
    cloud_red = senkou_a_aligned < senkou_b_aligned

    # Volume confirmation: volume > 1.8 * 20-period average (~10 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after 52-period for Senkou B
        # Skip if any required value is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Green cloud + TK cross bullish (Tenkan > Kijun) + volume spike
            if cloud_green[i] and tenkan_aligned[i] > kijun_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Red cloud + TK cross bearish (Tenkan < Kijun) + volume spike
            elif cloud_red[i] and tenkan_aligned[i] < kijun_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish or cloud turns red
            if tenkan_aligned[i] < kijun_aligned[i] or cloud_red[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish or cloud turns green
            if tenkan_aligned[i] > kijun_aligned[i] or cloud_green[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals