#!/usr/bin/env python3
# 4h_TripleBarrier_Breakout_1dTrend_Volume
# Hypothesis: Combines Donchian(20) breakout with volume confirmation and 1d EMA trend filter.
# Uses triple-barrier logic: enter on breakout of 20-period high/low, exit on reversal or time.
# Designed to work in both bull and bear markets by following higher timeframe trend and avoiding false breakouts.
# Target: ~25-35 trades/year to stay within optimal range and minimize fee drag.

name = "4h_TripleBarrier_Breakout_1dTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 2x 20-period SMA (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema34_aligned = ema34_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(ema34_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike (2x) + 1d uptrend
            if (close[i] > donchian_high_val and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike (2x) + 1d downtrend
            elif (close[i] < donchian_low_val and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR 1d trend turns down
            if close[i] < donchian_low_val or close[i] < ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR 1d trend turns up
            if close[i] > donchian_high_val or close[i] > ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals