#!/usr/bin/env python3
# 4h_Pivot_Trend_Follow
# Hypothesis: Use daily pivot points (PP) from 1d timeframe as trend filters, combined with 4h price action above/below PP for entry.
# In bull markets (price > 1d PP), go long on 4h breakouts above resistance; in bear markets (price < 1d PP), go short on breakdowns below support.
# Uses volume confirmation to avoid false breakouts. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in both bull/bear by following 1d trend direction via pivot point location.

name = "4h_Pivot_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot point: (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate support and resistance levels
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    s2_1d = pp_1d - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (wait for daily close)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)

    # Calculate 4h SMA20 for trend confirmation and noise filter
    close_series = pd.Series(close)
    sma20_4h = close_series.rolling(window=20, min_periods=20).mean().values

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(sma20_4h[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above PP and above SMA20 (bullish bias) with breakout above R1 and volume spike
            if close[i] > pp_1d_aligned[i] and close[i] > sma20_4h[i] and \
               high[i] > r1_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below PP and below SMA20 (bearish bias) with breakdown below S1 and volume spike
            elif close[i] < pp_1d_aligned[i] and close[i] < sma20_4h[i] and \
                 low[i] < s1_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below PP (trend change) or breaks below S1 with volume
            if close[i] < pp_1d_aligned[i] or (low[i] < s1_1d_aligned[i] and volume[i] > volume_sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above PP (trend change) or breaks above R1 with volume
            if close[i] > pp_1d_aligned[i] or (high[i] > r1_1d_aligned[i] and volume[i] > volume_sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals