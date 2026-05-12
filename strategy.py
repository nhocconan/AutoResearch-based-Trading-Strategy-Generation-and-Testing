#!/usr/bin/env python3
"""
4h_Stochastic_BullBear_EqualWeight
Hypothesis: Buy when Stochastic %K crosses above %D from below 20 and 4h close > daily EMA50; sell when %K crosses below %D from above 80 and 4h close < daily EMA50. Uses equal-weight long/short to profit in both bull and bear markets. Confirm with volume spike (>1.5x 6-bar average) to reduce false signals. Target 20-50 trades/year.
Timeframe: 4h
"""

name = "4h_Stochastic_BullBear_EqualWeight"
timeframe = "4h"
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

    # Get daily data for EMA50 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Stochastic Oscillator (14,3,3) on 4h
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = np.where((highest_high - lowest_low) == 0, 50, k)  # avoid div by zero
    d = pd.Series(k).rolling(window=3, min_periods=3).mean().values

    # Volume spike: current > 1.5x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after Stochastic warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(k[i]) or np.isnan(d[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: %K crosses above %D from below 20, price > daily EMA50, volume spike
            if (k[i-1] <= d[i-1] and k[i] > d[i] and k[i] < 20 and
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: %K crosses below %D from above 80, price < daily EMA50, volume spike
            elif (k[i-1] >= d[i-1] and k[i] < d[i] and k[i] > 80 and
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: %K crosses below %D from above 80 (overbought)
            if k[i-1] >= d[i-1] and k[i] < d[i] and k[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: %K crosses above %D from below 20 (oversold)
            if k[i-1] <= d[i-1] and k[i] > d[i] and k[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals