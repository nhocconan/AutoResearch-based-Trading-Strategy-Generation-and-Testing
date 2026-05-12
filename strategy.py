#!/usr/bin/env python3
"""
12h_WMA_Crossover_VolumeTrend
Hypothesis: 12h WMA crossover (fast/slow) with volume confirmation and 1w EMA trend filter captures medium-term momentum.
Works in bull/bear by requiring trend alignment: only long when price above 1w EMA, short when below.
Volume spike confirms breakout strength. Targets 20-50 trades/year to avoid fee drag.
"""

name = "12h_WMA_Crossover_VolumeTrend"
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

    # Get 1w data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate 12h WMA crossover
    wma_fast = pd.Series(close).ewm(span=9, adjust=False).mean().values  # WMA approximation via EMA
    wma_slow = pd.Series(close).ewm(span=21, adjust=False).mean().values
    wma_cross = wma_fast - wma_slow  # >0 bullish, <0 bearish

    # Volume confirmation: volume > 1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(wma_cross[i]) or np.isnan(vol_avg_30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: WMA bullish cross + price above 1w EMA50 + volume spike
            if wma_cross[i] > 0 and wma_cross[i-1] <= 0 and close[i] > ema50_1w_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: WMA bearish cross + price below 1w EMA50 + volume spike
            elif wma_cross[i] < 0 and wma_cross[i-1] >= 0 and close[i] < ema50_1w_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WMA bearish cross or price below 1w EMA50
            if wma_cross[i] < 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WMA bullish cross or price above 1w EMA50
            if wma_cross[i] > 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals