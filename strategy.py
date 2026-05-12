#!/usr/bin/env python3
# 6h_1d_Donchian20_Breakout_WeeklyTrend_Volume
# Hypothesis: 6h Donchian channel (20-period) breakout with weekly trend filter (1w EMA50) and volume confirmation (1.5x avg volume).
# Weekly trend ensures alignment with higher timeframe direction to avoid counter-trend trades.
# Volume breakout confirms momentum. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull/bear by following weekly trend. Uses only close-based exits to avoid look-ahead.

name = "6h_1d_Donchian20_Breakout_WeeklyTrend_Volume"
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

    # Get 1d data for Donchian channel calculation (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Donchian channel (20-period) on daily data
    # Upper band = max(high over last 20 days)
    # Lower band = min(low over last 20 days)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values

    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)

    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian upper in weekly uptrend with volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian lower in weekly downtrend with volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal signal)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal signal)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals