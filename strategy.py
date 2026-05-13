#!/usr/bin/env python3
# 4h_Donchian20_Breakout_VolumeTrend_1dTrendFilter
# Hypothesis: Trade Donchian(20) breakouts with volume confirmation and 1d EMA trend filter.
# Long when price breaks above 20-period high with volume > 1.5x 20-period average and price > 1d EMA50.
# Short when price breaks below 20-period low with volume > 1.5x 20-period average and price < 1d EMA50.
# Exit when price crosses 10-period EMA in opposite direction or trend flips.
# Designed for 4h timeframe to capture medium-term trends with minimal trades (~25-40/year).
# Works in bull markets via trend-following breakouts and in bear markets via short breakdowns.
# Uses 1d trend filter to avoid counter-trend whipsaws, targeting 20-50 trades/year per symbol.

name = "4h_Donchian20_Breakout_VolumeTrend_1dTrendFilter"
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
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i]) or np.isnan(ema_10[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + volume spike + 1d uptrend
            if high[i] > donchian_high[i-1] and volume[i] > vol_avg_20[i] * 1.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + volume spike + 1d downtrend
            elif low[i] < donchian_low[i-1] and volume[i] > vol_avg_20[i] * 1.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 10 EMA or trend turns down
            if close[i] < ema_10[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 10 EMA or trend turns up
            if close[i] > ema_10[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals