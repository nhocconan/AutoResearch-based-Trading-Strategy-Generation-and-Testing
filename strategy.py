#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout of Camarilla R3/S3 levels on 12h with 1d trend filter and volume confirmation captures momentum while avoiding false breaks. Works in bull/bear via 1d trend filter and volatility-based levels.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # ATR for context (not used in signal)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Camarilla levels from previous day (R3, S3)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First bar: use current values
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2

    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + 1d EMA34 uptrend + volume spike
            if (close[i] > r3[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + 1d EMA34 downtrend + volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below previous day's close or volatility drop
            if close[i] < prev_close[i] or volume[i] < vol_avg_30[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above previous day's close or volatility drop
            if close[i] > prev_close[i] or volume[i] < vol_avg_30[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals