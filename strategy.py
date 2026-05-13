#!/usr/bin/env python3
# 1d_KAMA_1wTrend_Volume
# Hypothesis: Use KAMA direction on daily timeframe for trend following with weekly trend filter and volume confirmation.
# Long when KAMA > previous KAMA (upward trend) and weekly trend is up and volume above average.
# Short when KAMA < previous KAMA (downward trend) and weekly trend is down and volume above average.
# Exit when KAMA reverses direction or weekly trend changes.
# Designed for low trade frequency (30-100 total over 4 years) to minimize fee drag.

name = "1d_KAMA_1wTrend_Volume"
timeframe = "1d"
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

    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) with ER=10, FC=2, SC=30
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # will fix below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start = max(0, i - 9)
        volatility_sum[i] = np.sum(volatility[start:i+1])
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no alignment needed as we're on 1d)
    kama_1d = kama

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (upward trend) + weekly trend up + volume spike
            if (i > 0 and kama_1d[i] > kama_1d[i-1] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (downward trend) + weekly trend down + volume spike
            elif (i > 0 and kama_1d[i] < kama_1d[i-1] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falls or weekly trend turns down
            if (i > 0 and kama_1d[i] < kama_1d[i-1]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rises or weekly trend turns up
            if (i > 0 and kama_1d[i] > kama_1d[i-1]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals