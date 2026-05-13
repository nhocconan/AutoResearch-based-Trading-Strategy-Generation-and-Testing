#!/usr/bin/env python3
# 6h_DonchianBreakout_WeeklyTrend
# Hypothesis: Breakouts from 4-week Donchian channels in the direction of weekly trend,
# filtered by volume spikes, capture explosive moves in both bull and bear markets.
# Weekly trend filter prevents counter-trend trades during corrections.
# Target: 15-25 trades/year per symbol.

name = "6h_DonchianBreakout_WeeklyTrend"
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

    # Get weekly data for Donchian channel and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # 4-week Donchian channel (20 periods)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values

    # Weekly trend: 50-period EMA
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > weekly_ema50
    weekly_downtrend = close_1w < weekly_ema50

    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    # Align weekly data to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned weekly values
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        up_trend = weekly_uptrend_aligned[i]
        down_trend = weekly_downtrend_aligned[i]
        vol_spike = volume_spike[i]

        # Skip if any value is NaN
        if np.isnan(upper) or np.isnan(lower) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + price breaks above Donchian high + volume spike
            if up_trend and close[i] > upper and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below Donchian low + volume spike
            elif down_trend and close[i] < lower and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low or trend changes
            if close[i] < lower or not up_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high or trend changes
            if close[i] > upper or not down_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals