#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_Follow
Hypothesis: The Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) identifies trends when lines are ordered and separated. 
In bull: Lips > Teeth > Jaw; in bear: Lips < Teeth < Jaw. 
Add 1w EMA50 trend filter and volume confirmation (>1.5x 20-day avg) to avoid whipsaws.
Works in both bull and bear by following higher timeframe trend.
"""

name = "1d_Williams_Alligator_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Williams Alligator on daily data
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMA

    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + price above 1w EMA50 + volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + price below 1w EMA50 + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth (trend weakness)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth (trend weakness)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals