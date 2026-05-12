#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Extreme
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a robust trend filter. Combined with RSI extremes (overbought/oversold) and volume confirmation, it captures mean-reversion within the trend. Works in bull/bear by only taking trades aligned with the daily KAMA trend. Uses 4h timeframe for balance of signal frequency and cost control.
"""

name = "4h_KAMA_Trend_RSI_Extreme"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d KAMA for trend filter
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    # Handle first 10 values where diff(10) is invalid
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])  # diff(1) loses 1 element
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama

    # Align 1d KAMA to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # 4h RSI (14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    # Subsequent averages (Wilder's smoothing)
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # 4h Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below KAMA (dip in uptrend) + RSI oversold + volume spike
            if (close[i] < kama_1d_aligned[i] and 
                rsi[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above KAMA (rally in downtrend) + RSI overbought + volume spike
            elif (close[i] > kama_1d_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above KAMA (trend resumption) or RSI overbought
            if close[i] > kama_1d_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below KAMA (trend resumption) or RSI oversold
            if close[i] < kama_1d_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals