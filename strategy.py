#!/usr/bin/env python3
"""
1d_KAMA_Direction_WeeklyTrend_Volume
Hypothesis: On daily timeframe, go long when KAMA indicates up-trend, price is above weekly EMA50, and volume spike confirms; go short when KAMA indicates down-trend, price below weekly EMA50, and volume spike. Exit when KAMA reverses. Uses weekly trend filter to avoid counter-trend trades and volume spike to ensure commitment. Designed for low trade frequency (<25/year) to minimize fee drag in both bull and bear markets.
"""

name = "1d_KAMA_Direction_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
    volatility = np.concatenate([[0], volatility[1:]])  # avoid shift issues
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)  # 1=up, -1=down
    kama_dir[0] = 1  # initialize

    # Volume spike: current > 2.0x average of last 10 days
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up + price > weekly EMA50 + volume spike
            if (kama_dir[i] == 1 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down + price < weekly EMA50 + volume spike
            elif (kama_dir[i] == -1 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals