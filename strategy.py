#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike
# Hypothesis: 6s Ichimoku TK cross with 1d cloud filter and volume spike confirmation.
# Uses 1d Ichimoku cloud (Senkou A/B) to determine trend direction (price above/below cloud)
# and TK cross (Tenkan/Kijun) on 6s for entry timing. Volume spike confirms momentum.
# Works in bull/bear by following 1d Ichimoku trend, avoiding counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 6s data for TK cross calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 26:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Calculate Ichimoku components on 6s
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2

    # Get 1d data for cloud filter (using same Ichimoku but on daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Ichimoku components on 1d
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2

    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2

    senkou_a_1d = (tenkan_1d + kijun_1d) / 2

    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2

    # Align Ichimoku components to 6s timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)

    # Align 1d cloud components to 6s timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)

    # Calculate 6s volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine 1d cloud position: price above/below cloud
        # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top_1d
        price_below_cloud = close[i] < cloud_bottom_1d

        if position == 0:
            # LONG: TK cross bullish (Tenkan > Kijun) AND price above 1d cloud AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and price_above_cloud and
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish (Tenkan < Kijun) AND price below 1d cloud AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and price_below_cloud and
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish OR price breaks below 1d cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_bottom_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish OR price breaks above 1d cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_top_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals