#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike
# Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d Kumo cloud filter and volume spike confirmation.
# In bull markets: price above cloud + bullish TK cross + volume spike = long.
# In bear markets: price below cloud + bearish TK cross + volume spike = short.
# The 1d cloud acts as a strong trend filter, reducing whipsaws. Volume spikes confirm momentum.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

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

    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get 1d data for Kumo cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2

    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2

    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)

    # Calculate 1d Kumo cloud (Senkou Span A and B)
    max_high_1d_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_1d_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    senkou_a_1d = (max_high_1d_9 + min_low_1d_9) / 2

    max_high_1d_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_1d_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_1d_52 + min_low_1d_52) / 2

    # Align 1d cloud to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud color and position
        upper_band = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_band = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > upper_band
        price_below_cloud = close[i] < lower_band

        # TK cross signals
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]

        if position == 0:
            # LONG: Price above cloud + bullish TK cross + volume spike
            if price_above_cloud and tk_bullish and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + bearish TK cross + volume spike
            elif price_below_cloud and tk_bearish and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud or bearish TK cross
            if price_below_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud or bullish TK cross
            if price_above_cloud or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals