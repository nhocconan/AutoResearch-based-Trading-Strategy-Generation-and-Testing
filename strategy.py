#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_1dTrend_Volume
# Hypothesis: 12h breakout of 20-period Donchian channel with 1d trend filter and volume confirmation.
# Uses price channel breakouts (Donchian) as robust support/resistance, confirmed by 1d EMA trend and volume spike.
# Designed to work in both bull and bear markets by following higher timeframe trend and avoiding false breakouts.
# Target: 10-30 trades per year to minimize fee drag and improve generalization.

name = "12h_PriceChannel_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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

    # Calculate 12h Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        donchian_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), donchian_high)[i]
        donchian_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), donchian_low)[i]
        ema34_aligned = ema34_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned) or np.isnan(donchian_low_aligned) or 
            np.isnan(ema34_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + 1d uptrend
            if (close[i] > donchian_high_aligned and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + 1d downtrend
            elif (close[i] < donchian_low_aligned and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR 1d trend turns down
            if close[i] < donchian_low_aligned or close[i] < ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR 1d trend turns up
            if close[i] > donchian_high_aligned or close[i] > ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals