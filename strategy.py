#!/usr/bin/env python3
# 12h_KAMA_Direction_1wTrend_Volume
# Hypothesis: KAMA direction on 12h, filtered by 1-week trend and volume confirmation.
# KAMA adapts to market noise, providing smooth trend signals. The 1-week trend ensures alignment with higher timeframe momentum.
# Volume confirmation filters out low-conviction moves. Works in both bull and bear markets by trading only in the direction of the 1w trend.

name = "12h_KAMA_Direction_1wTrend_Volume"
timeframe = "12h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: volatility calculation needs to be over er_length window
    volatility = pd.Series(close).rolling(window=er_length).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)

    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2

    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # seed
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # 1-week EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(er_length + 1, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, above 1w EMA34, with volume confirmation
            if (kama[i] > kama[i-1] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, below 1w EMA34, with volume confirmation
            elif (kama[i] < kama[i-1] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falls or price crosses below 1w EMA34
            if kama[i] < kama[i-1] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rises or price crosses above 1w EMA34
            if kama[i] > kama[i-1] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals