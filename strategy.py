#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dRSI_OverboughtOversold
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h indicates trend direction, while 1d RSI identifies overbought/oversold conditions for mean-reversion entries in the direction of the 4h trend. This combines trend-following with mean-reversion to work in both bull and bear markets by following the 4h trend and using daily RSI for timing.
"""

name = "4h_KAMA_Direction_1dRSI_OverboughtOversold"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate KAMA on 4h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Initialize
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI on 1d data
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    # Shift by 1 to use previous day's RSI
    rsi_1d = np.roll(rsi_1d, 1)
    rsi_1d[0] = np.nan

    # Align KAMA and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # Already on 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after RSI warmup
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) and RSI oversold (<30)
            if (close[i] > kama_aligned[i] and rsi_1d_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) and RSI overbought (>70)
            elif (close[i] < kama_aligned[i] and rsi_1d_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI overbought (>70)
            if (close[i] < kama_aligned[i] or rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI oversold (<30)
            if (close[i] > kama_aligned[i] or rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals