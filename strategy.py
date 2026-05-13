#!/usr/bin/env python3
# 6h_LowVol_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low volatility periods (using ATR contraction) with 1d EMA trend filter and volume spike.
# Low volatility precedes explosive moves. Works in both bull and bear as breakouts capture momentum in trend direction.
# Uses 1d EMA50 for trend filter and ATR contraction for volatility regime. Target: 20-40 trades/year on 6h.

name = "6h_LowVol_Breakout_1dTrend_Volume"
timeframe = "6h"
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

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # ATR(20) for volatility measurement on 6s
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # ATR ratio: current ATR / 50-period average ATR (volatility contraction/expansion)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50  # < 1 = low volatility, > 1 = high volatility

    # Donchian breakout levels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + low volatility (contraction) + uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                atr_ratio[i] < 0.8 and  # volatility contraction
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + low volatility (contraction) + downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_ratio[i] < 0.8 and  # volatility contraction
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian mean or volatility expansion
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] < donchian_mid or atr_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian mean or volatility expansion
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] > donchian_mid or atr_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals