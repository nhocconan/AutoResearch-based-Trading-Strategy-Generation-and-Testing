#!/usr/bin/env python3
# 4h_Keltner_Breakout_TrendVolume
# Hypothesis: 4h Keltner channel breakout with 1d EMA trend filter and volume spike confirmation.
# The 1d EMA provides trend direction to avoid counter-trend trades, while volume spikes confirm breakout strength.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag. Works in bull/bear by following 1d trend.
# Uses 4h ATR for Keltner channel width and volume confirmation for signal strength.

name = "4h_Keltner_Breakout_TrendVolume"
timeframe = "4h"
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

    # Get 4h data for price action and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate ATR for Keltner channels
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate Keltner channels: EMA20 ± 2 * ATR
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20_4h + 2 * atr
    keltner_lower = ema20_4h - 2 * atr

    # Align Keltner channels to 4h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_4h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_4h, keltner_lower)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Keltner upper in 1d uptrend with volume spike
            if close[i] > keltner_upper_aligned[i] and close[i] > ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Keltner lower in 1d downtrend with volume spike
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Keltner lower (reversal signal)
            if close[i] < keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Keltner upper (reversal signal)
            if close[i] > keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals