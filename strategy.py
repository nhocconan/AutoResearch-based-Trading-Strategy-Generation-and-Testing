#!/usr/bin/env python3
# 4h_KAMA_Direction_1dRSI_Filter_Volume
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction,
# combined with 1d RSI for overbought/oversold filtering and volume confirmation.
# Enter long when KAMA turns up, RSI < 50 (not overbought), and volume spike.
# Enter short when KAMA turns down, RSI > 50 (not oversold), and volume spike.
# Exit when KAMA reverses direction. This adapts to market conditions and avoids
# chasing extremes, working in both bull and bear markets.

name = "4h_KAMA_Direction_1dRSI_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA on 4h close
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[:length] = close[:length]
        for i in range(length, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals

    kama_vals = kama(close, length=10, fast=2, slow=30)

    # Calculate 1d RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        # Pad beginning
        rsi_vals = np.concatenate([np.full(length, np.nan), rsi_vals[length:]])
        return rsi_vals

    rsi_1d = rsi(close_1d, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (current > previous), RSI < 50, volume spike
            if (kama_vals[i] > kama_vals[i-1] and 
                rsi_1d_aligned[i] < 50 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (current < previous), RSI > 50, volume spike
            elif (kama_vals[i] < kama_vals[i-1] and 
                  rsi_1d_aligned[i] > 50 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down
            if kama_vals[i] < kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up
            if kama_vals[i] > kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals