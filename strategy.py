#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla pivot breakouts with 1-week trend filter and volume confirmation
capture institutional supply/demand levels. R1/S1 are primary support/resistance.
Breakout above R1 with weekly uptrend = long; breakdown below S1 with weekly downtrend = short.
Works in both bull and bear markets by following weekly trend. Target: 10-20 trades/year.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Calculate daily Camarilla levels (based on previous day's range)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    
    # First day: use current values
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    range_prev = high_prev - low_prev
    
    # Camarilla levels: R1 = close + 1.1/12 * range, S1 = close - 1.1/12 * range
    camarilla_r1 = close_prev + (1.1 / 12) * range_prev
    camarilla_s1 = close_prev - (1.1 / 12) * range_prev

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R1 + weekly uptrend + volume spike
            if close[i] > camarilla_r1[i] and close[i] > ema21_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1 + weekly downtrend + volume spike
            elif close[i] < camarilla_s1[i] and close[i] < ema21_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla S1 or weekly trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla R1 or weekly trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals