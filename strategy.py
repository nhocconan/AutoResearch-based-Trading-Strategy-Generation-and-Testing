#!/usr/bin/env python3
"""
6h_KAMA_Trend_With_Volume_Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat. 
Combined with volume confirmation (1.5x average) and 1d trend filter, this captures strong trending moves while avoiding false signals in chop. 
Works in bull/bear by following 1d trend direction and requiring volume confirmation.
"""

name = "6h_KAMA_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate KAMA (Kaufman Adaptive Moving Average) - 60 period
    # ER = Efficiency Ratio = |change| / sum(|changes|)
    # SSC = Smoothing Constant = [ER*(fastest-1) + 1]^2
    # KAMA = previous KAMA + SSC*(price - previous KAMA)
    close_series = pd.Series(close)
    change = abs(close_series.diff())
    volatility = change.rolling(window=30, min_periods=30).sum()
    direction = abs(close_series - close_series.shift(30))
    er = direction / volatility
    er = er.fillna(0)
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + 1d EMA34 uptrend + volume spike
            if (close[i] > kama[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + 1d EMA34 downtrend + volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals