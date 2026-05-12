#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_With_Volume_Trend_Filter
# Hypothesis: 4-hour Donchian(20) breakouts with volume confirmation and 1-day EMA50 trend filter.
# Uses actual Donchian channels for price structure, volume spike (2.0x SMA20) for confirmation,
# and daily EMA50 to filter trend direction. Designed for low trade frequency (<50/year) to minimize
# fee drag while capturing trends in both bull and bear markets.

name = "4h_Donchian_Breakout_20_With_Volume_Trend_Filter"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        donchian_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), donchian_high)[i]
        donchian_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), donchian_low)[i]
        ema50_aligned = ema50_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned) or np.isnan(donchian_low_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above Donchian high + volume spike + daily uptrend
            if (close[i] > donchian_high_aligned and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below Donchian low + volume spike + daily downtrend
            elif (close[i] < donchian_low_aligned and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low OR daily trend turns down
            if close[i] < donchian_low_aligned or close[i] < ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high OR daily trend turns up
            if close[i] > donchian_high_aligned or close[i] > ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals