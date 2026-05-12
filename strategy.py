#!/usr/bin/env python3
# 4h_DonchianBreakout_VolumeTrend
# Hypothesis: 4h Donchian(20) breakout with volume (>2x 20-period average) and 1d EMA50 trend filter.
# Long when price breaks above upper band + volume spike + price > 1d EMA50.
# Short when price breaks below lower band + volume spike + price < 1d EMA50.
# Exit when price crosses back through the opposite band or trend fails.
# Designed for 20-50 trades/year to minimize fee drag and work in bull/bear via trend filter.

name = "4h_DonchianBreakout_VolumeTrend"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian band + volume spike + price > 1d EMA50
            if (close[i] > highest_high[i] and
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band + volume spike + price < 1d EMA50
            elif (close[i] < lowest_low[i] and
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below lower Donchian band OR trend fails (price < 1d EMA50)
            if (close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above upper Donchian band OR trend fails (price > 1d EMA50)
            if (close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals