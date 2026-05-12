#!/usr/bin/env python3

# 6h_12h_1d_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX momentum on 6h combined with 12h trend filter and volume confirmation.
# TRIX filters out insignificant cycles, showing smoothed momentum. Works in both bull and bear markets
# by requiring alignment with higher timeframe trend and volume spikes to confirm breakout strength.
# Targets 15-30 trades/year (~60-120 total over 4 years).

name = "6h_12h_1d_TRIX_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Calculate TRIX on 6h (15-period EMA of EMA of EMA, then ROC)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change(periods=1))
    trix_values = trix.values

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(trix_values[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 12h
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]

        if position == 0:
            # LONG: TRIX crosses above zero with bullish trend and volume confirmation
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with bearish trend and volume confirmation
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend turns bearish
            if trix_values[i] < 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend turns bullish
            if trix_values[i] > 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals