#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBound_MeanReversion
Hypothesis: Weekly pivot points (from prior week) define key support/resistance levels.
Price tends to mean-revert within the weekly range (S1-R1) in ranging markets, with
breakouts above R1 or below S1 signaling trend continuation. Uses 1d ADX < 25 to identify
ranging conditions and volume confirmation to avoid false signals. Works in bull via
breakout continuation and bear via mean-reversion at extremes with volatility filter.
Target: 15-35 trades/year (60-140 total over 4 years) with low turnover to minimize fee drag.
"""

name = "6h_WeeklyPivot_RangeBound_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    # We use R1/S1 as primary levels
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    # Use previous week's levels to avoid look-ahead
    pp_prev = np.roll(pp, 1)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    # Align to 6m timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_prev)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_prev)

    # Get 1d data for ADX and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values

    # Calculate 1d ADX(14) for regime detection (trending vs ranging)
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, high - prev_close, prev_close - low)
    # +DM smoothed, -DM smoothed, TR smoothed
    # ADX = 100 * smoothed(|+DM - -DM|/(+DM + -DM))
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)

    plus_dm = np.where((high_1d - high_prev) > (low_prev - low_1d), np.maximum(high_1d - high_prev, 0), 0)
    minus_dm = np.where((low_prev - low_1d) > (high_1d - high_prev), np.maximum(low_prev - low_1d, 0), 0)
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_prev), np.abs(low_1d - close_prev)))

    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result

    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)

    # Avoid division by zero
    dx = np.zeros_like(tr_smooth)
    denom = plus_dm_smooth + minus_dm_smooth
    dx = np.where(denom != 0, 100 * np.abs(plus_dm_smooth - minus_dm_smooth) / denom, 0)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: 1.2x 24-period average (more sensitive for 6h)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_avg_val = vol_avg_24[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or 
            np.isnan(adx_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range filter: only trade when ADX < 25 (ranging market)
        if adx_val >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price approaches S1 support with volume confirmation (mean reversion)
            if (close[i] <= s1_level * 1.005 and  # Allow small buffer
                volume[i] > vol_avg_val * 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price approaches R1 resistance with volume confirmation (mean reversion)
            elif (close[i] >= r1_level * 0.995 and  # Allow small buffer
                  volume[i] > vol_avg_val * 1.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or shows weakness
            midpoint = (r1_level + s1_level) / 2
            if (close[i] >= midpoint or 
                volume[i] < vol_avg_val * 0.8):  # Loss of momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or shows weakness
            midpoint = (r1_level + s1_level) / 2
            if (close[i] <= midpoint or 
                volume[i] < vol_avg_val * 0.8):  # Loss of momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals