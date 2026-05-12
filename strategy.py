#!/usr/bin/env python3
# 1h_4h_Donchian_Breakout_VolumeTrend
# Hypothesis: Donchian breakout on 4h for trend direction, volume spike for confirmation, and 1h for precise entry timing.
# Uses 4h Donchian channels (20-period) to establish trend, volume spike > 2x 20-period SMA for momentum confirmation.
# Enters long when price breaks above 4h upper band with volume confirmation, short when breaks below lower band.
# Includes session filter (08-20 UTC) to avoid low-liquidity hours. Targets 15-35 trades/year to minimize fee drag.

name = "1h_4h_Donchian_Breakout_VolumeTrend"
timeframe = "1h"
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

    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_band = align_htf_to_ltf(prices, df_4h, high_max_20)
    lower_band = align_htf_to_ltf(prices, df_4h, low_min_20)

    # Volume spike: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_sma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 4h upper band with volume spike
            if close[i] > upper_band[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h lower band with volume spike
            elif close[i] < lower_band[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h lower band (reverse signal)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h upper band (reverse signal)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals