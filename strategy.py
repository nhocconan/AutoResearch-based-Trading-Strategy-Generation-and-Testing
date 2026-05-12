#!/usr/bin/env python3
"""
4h_Pivot_HighLow_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts above the daily high or below the daily low on 4h timeframe when aligned with 1d EMA50 trend and confirmed by volume spike. This strategy uses the previous day's high/low as key support/resistance levels, which are more robust than pivot points in trending markets. Works in both bull and bear markets by following the daily trend with trend-following entries and exiting at the opposite daily level.
Timeframe: 4h
"""

name = "4h_Pivot_HighLow_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for high/low levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily high and low (using prior day's OHLC)
    prev_high = df_1d['high'].shift(1).values  # prior day high
    prev_low = df_1d['low'].shift(1).values    # prior day low
    # Align to 4h: daily high/low values are constant through the day
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)

    # Get daily data for EMA50 trend filter ONCE before loop
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume spike: current > 2.0x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(80, n):  # Start after EMA50 warmup
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > previous day's high + price > 1d EMA50 + volume spike
            if (close[i] > prev_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < previous day's low + price < 1d EMA50 + volume spike
            elif (close[i] < prev_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < previous day's low
            if close[i] < prev_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > previous day's high
            if close[i] > prev_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals