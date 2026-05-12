#!/usr/bin/env python3
"""
12h_WeeklyKAMA_Trend_Volume
Hypothesis: Trade weekly KAMA direction on 12h timeframe with volume confirmation.
Weekly KAMA adapts to market efficiency - trending in strong moves, mean-reverting in chop.
Works in bull/bear by following weekly trend, avoids whipsaws via adaptive smoothing.
Volume spike confirms institutional participation. Targets 15-25 trades/year.
"""

name = "12h_WeeklyKAMA_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for KAMA trend ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Calculate weekly KAMA
    wk_close = df_1w['close'].values
    wk_kama = calculate_kama(wk_close, er_len=10, fast=2, slow=30)
    wk_kama_prev = np.roll(wk_kama, 1)
    wk_kama_prev[0] = wk_kama[0]
    
    # Align to 12h
    wk_kama_aligned = align_htf_to_ltf(prices, df_1w, wk_kama)
    wk_kama_prev_aligned = align_htf_to_ltf(prices, df_1w, wk_kama_prev)

    # 12h volume spike: current > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after weekly KAMA warmup
        if (np.isnan(wk_kama_aligned[i]) or np.isnan(wk_kama_prev_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: weekly price > KAMA + volume spike
            if (wk_kama_aligned[i] > wk_kama_prev_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: weekly price < KAMA + volume spike
            elif (wk_kama_aligned[i] < wk_kama_prev_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: weekly price < KAMA (trend change)
            if wk_kama_aligned[i] < wk_kama_prev_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: weekly price > KAMA (trend change)
            if wk_kama_aligned[i] > wk_kama_prev_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals