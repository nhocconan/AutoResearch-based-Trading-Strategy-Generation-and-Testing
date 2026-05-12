#!/usr/bin/env python3
"""
1d_Pivot_Bounce_With_Weekly_Trend_Filter
Hypothesis: Price bounces off weekly pivot levels (R1/S1) on daily timeframe when aligned with weekly trend (EMA50) and confirmed by volume spike. Works in both bull and bear markets by trading mean-reversion at key weekly levels with trend filter to avoid counter-trend traps. Targets 10-25 trades/year by requiring confluence of level touch, trend alignment, and volume confirmation.
Timeframe: 1d
"""

name = "1d_Pivot_Bounce_With_Weekly_Trend_Filter"
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

    # Get weekly data for pivot levels and trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate weekly pivot levels (using prior week's OHLC)
    ph = df_1w['high'].shift(1).values  # prior week high
    pl = df_1w['low'].shift(1).values   # prior week low
    pc = df_1w['close'].shift(1).values # prior week close
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    # Align to daily: weekly pivot values constant through the week
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Get weekly data for EMA50 trend filter ONCE before loop
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume spike: current > 2.0x average of last 5 days
    vol_ma = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price touches or crosses below S1 then reverses up + above weekly EMA50 + volume spike
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price touches or crosses above R1 then reverses down + below weekly EMA50 + volume spike
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches or crosses weekly R1 level
            if high[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches or crosses weekly S1 level
            if low[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals