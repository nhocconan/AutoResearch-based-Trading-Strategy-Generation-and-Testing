#!/usr/bin/env python3
# 4h_KAMA_Direction_1dRSI_Strength_12hVolumeSpike
# Hypothesis: Use 4h KAMA to determine trend direction, combined with 1d RSI for overbought/oversold extremes and 12h volume spike for confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets. 1d RSI >70 or <30 identifies exhaustion points where reversals are likely.
# Volume spike on 12h confirms institutional participation. Designed for low turnover in both bull and bear markets by requiring trend alignment, extreme RSI, and volume confirmation.

name = "4h_KAMA_Direction_1dRSI_Strength_12hVolumeSpike"
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

    # Get daily data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    # Get 12h data ONCE before loop for volume spike
    df_12h = get_htf_data(prices, '12h')

    # Calculate 1d RSI(14)
    if len(df_1d) < 15:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Calculate 12h average volume for spike detection
    if len(df_12h) < 2:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=2, min_periods=2).mean().values  # Average of last 2 periods
    vol_spike_12h = vol_12h > (1.5 * vol_ma_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)

    # Calculate 4h KAMA
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # Needs correction
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.array([np.sum(np.abs(np.diff(close[max(0,i-9):i+1]))) for i in range(len(close))])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # Alternative simpler KAMA calculation using pandas (but we need values)
    # Using efficient computation: KAMA = previous KAMA + sc * (price - previous KAMA)
    # We'll compute it in a loop as above but fix volatility
    # Recalculate volatility correctly
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    # For first 10 values, use expanding sum
    for i in range(1, 10):
        volatility[i] = np.sum(np.abs(np.diff(close[0:i+1])))
    er = np.where(volatility > 0, np.abs(np.diff(close, k=10, prepend=close[:10])) / volatility, 0)
    # Avoid division by zero in ER calculation for first 10
    er[:10] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate 4h KAMA slope for direction
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(kama_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA slope up + RSI < 30 (oversold) + volume spike
            if (kama_slope[i] > 0 and 
                rsi_1d_aligned[i] < 30 and 
                vol_spike_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA slope down + RSI > 70 (overbought) + volume spike
            elif (kama_slope[i] < 0 and 
                  rsi_1d_aligned[i] > 70 and 
                  vol_spike_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA slope down OR RSI > 70 (overbought)
            if (kama_slope[i] < 0 or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA slope up OR RSI < 30 (oversold)
            if (kama_slope[i] > 0 or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals