#!/usr/bin/env python3
# 4h_Donchian_20_breakout_Volume_Trend
# Hypothesis: Donchian breakout with volume confirmation and trend filter on 4h timeframe.
# Uses 20-period Donchian channels for breakout detection, volume spike confirmation (>2x 20-period average),
# and 4h EMA50 trend filter. Designed to work in both bull and bear markets by requiring
# trend alignment and volume confirmation to filter false breakouts.
# Target: 20-50 trades per year on 4h timeframe to stay under fee drag threshold.

name = "4h_Donchian_20_breakout_Volume_Trend"
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

    # Calculate Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Calculate 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate volume spike threshold (2x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high with uptrend and volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50[i] and 
                volume[i] > volume_spike_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low with downtrend and volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50[i] and 
                  volume[i] > volume_spike_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals