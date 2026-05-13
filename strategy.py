#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_1dTrend_Trend
# Hypothesis: Use 12h Donchian breakout with volume confirmation and 1d EMA trend filter.
# In bull markets, go long on breakout above upper band with volume spike and price above 1d EMA.
# In bear markets, go short on breakout below lower band with volume spike and price below 1d EMA.
# The 1d EMA filter ensures alignment with daily trend, reducing false signals.
# Volume confirmation adds conviction to breakout moves.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Donchian_Breakout_Volume_1dTrend_Trend"
timeframe = "12h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Donchian Channel (20) on 12h
    donchian_length = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_length, min_periods=donchian_length).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_length, min_periods=donchian_length).min().values

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian upper band + volume spike + price above 1d EMA50 (bullish trend)
            if (close[i] > donchian_upper[i] and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band + volume spike + price below 1d EMA50 (bearish trend)
            elif (close[i] < donchian_lower[i] and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian lower band or price below 1d EMA50
            if (close[i] < donchian_lower[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian upper band or price above 1d EMA50
            if (close[i] > donchian_upper[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals