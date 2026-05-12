#!/usr/bin/env python3
"""
4h_4H_EMA50_Slope_1dTrend_Filter_300Vol
Hypothesis: Price crossing above/below 4h EMA50 with 1d EMA34 trend filter and 300-period volume confirmation captures strong trending moves while avoiding false breakouts. EMA50 represents intermediate trend, EMA34 on 1d provides higher timeframe trend direction, and volume filter ensures momentum behind moves. Works in bull/bear by following 1d trend direction.
"""

name = "4h_4H_EMA50_Slope_1dTrend_Filter_300Vol"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 350:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')

    # 4h EMA50
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume spike: >2.0x 300-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=300, min_periods=300).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(300, n):  # Start after volume MA warmup
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above EMA50 + 1d EMA34 uptrend + volume spike
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i-1] <= ema_50_4h_aligned[i-1] and
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price crosses below EMA50 + 1d EMA34 downtrend + volume spike
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i-1] >= ema_50_4h_aligned[i-1] and
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA50 (trend reversal)
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above EMA50 (trend reversal)
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals