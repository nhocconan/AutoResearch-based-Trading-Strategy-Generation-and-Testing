#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_1dTrend_Volume
Hypothesis: Trade reversals at daily Camarilla pivot points (R1/S1) on 4h timeframe when aligned with 1d EMA50 trend and confirmed by volume spike. Uses limit orders at pivot levels for precision entries. Works in both bull and bear markets by fading extreme moves toward the daily value area. Target: 20-50 trades/year via tight confluence of price, trend, and volume.
Timeframe: 4h
"""

name = "4h_Camarilla_Pivot_Reversal_1dTrend_Volume"
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

    # Get daily data for Camarilla levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    # Pivot point (not used for entry but for exit reference)
    pp = (ph + pl + pc) / 3.0
    # Align to 4h: daily values are constant through the day
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)

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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price touches or crosses below S1 (support) + uptrend + volume spike
            if (low[i] <= s1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price touches or crosses above R1 (resistance) + downtrend + volume spike
            elif (high[i] >= r1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches or crosses above pivot point (mean reversion to value area)
            if high[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches or crosses below pivot point (mean reversion to value area)
            if low[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals