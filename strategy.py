#!/usr/bin/env python3
# 4H_Donchian_Breakout_1dTrend_Volume_Scaled
# Hypothesis: Use 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above Donchian high with 1d EMA up and volume spike.
# Enter short when price breaks below Donchian low with 1d EMA down and volume spike.
# Exit when price crosses Donchian midline to avoid reversals.
# Uses tight position sizing (0.25) to limit drawdown and reduce trade frequency.
# Designed to work in both bull and bear markets via trend filter and volatility-based channels.

name = "4H_Donchian_Breakout_1dTrend_Volume_Scaled"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Donchian(20) on 4h data
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donch_mid = (donch_high + donch_low) / 2.0

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(lookback, n):
        # Skip if any required value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + 1d EMA up + volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + 1d EMA down + volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Cross below Donchian midline (mean reversion)
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Cross above Donchian midline
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals