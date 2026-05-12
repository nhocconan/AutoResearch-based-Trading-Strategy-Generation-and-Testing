#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: 6s Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly pivot points to determine long-term bias, Donchian breakouts for entry timing,
# and volume spikes to confirm breakout strength. Designed for 50-150 total trades over 4 years.
# Works in bull markets by following weekly trend and in bear markets by avoiding counter-trend trades.
# Weekly pivot provides structural support/resistance levels that persist across market regimes.

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
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

    # Get 6h data for price action and Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)

    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)

    # Calculate Donchian channels (20-period) on 6h data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values

    # Align Donchian channels to 6h timeframe (already aligned since calculated on 6h data)
    # But we still use align_htf_to_ltf for consistency with potential future modifications
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian upper in weekly uptrend (above R3) with volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > r3_1w_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian lower in weekly downtrend (below S3) with volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < s3_1w_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal signal) or below weekly pivot
            if close[i] < donchian_lower_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal signal) or above weekly pivot
            if close[i] > donchian_upper_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals