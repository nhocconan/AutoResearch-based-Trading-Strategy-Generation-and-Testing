#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian channel breakouts with volume confirmation and 1-day EMA50 trend filter capture directional momentum while avoiding false breakouts in choppy markets. The strategy targets 20-40 trades per year per symbol, suitable for 4H timeframe, with emphasis on BTC and ETH performance.
"""

name = "4h_Donchian_Breakout_Volume_Trend"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Donchian High + 1d uptrend + volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = 0.30
                position = 1
            # SHORT: Close breaks below Donchian Low + 1d downtrend + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Donchian Low or 1d trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close crosses above Donchian High or 1d trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals