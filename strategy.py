#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_TrendVolume_v2
# Hypothesis: Combine weekly pivot levels (from 1w) as trend filter with 6h Donchian breakout and volume confirmation.
# Weekly pivot provides long-term trend bias; Donchian breakout captures momentum; volume confirms institutional interest.
# Works in bull/bear markets by aligning with weekly trend direction. Targets 20-50 trades/year to minimize fee drag.

name = "6h_WeeklyPivot_DonchianBreakout_TrendVolume_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)

    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)

    # Align weekly pivot levels to 6h timeframe (delayed by 1 week for completed bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot, additional_delay_bars=0)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1, additional_delay_bars=0)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1, additional_delay_bars=0)

    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values

    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values

    # Align Donchian channels to 6t timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_channel)

    # Volume confirmation: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above weekly pivot, breaks above Donchian upper channel with volume spike
            if (close[i] > pivot_aligned[i] and 
                close[i] > upper_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly pivot, breaks below Donchian lower channel with volume spike
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < lower_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower channel or below weekly pivot
            if close[i] < lower_aligned[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper channel or above weekly pivot
            if close[i] > upper_aligned[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals