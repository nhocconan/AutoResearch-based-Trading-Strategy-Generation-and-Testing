#!/usr/bin/env python3
# 1d_KAMA_Direction_1wTrend_RSI_Filter
# Hypothesis: Use KAMA on 1d to determine direction (trend-following), filter with 1w RSI to avoid counter-trend trades in strong trends, and enter on pullbacks when daily RSI is overextended. Exit when KAMA flips direction. This captures trend continuation while avoiding whipsaws in ranging markets, working in both bull and bear via adaptive trend filter.

name = "1d_KAMA_Direction_1wTrend_RSI_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate KAMA on 1d for trend direction
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2))**2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Align KAMA to 1d (already on 1d, but ensure alignment)
    kama_aligned = kama  # no alignment needed as already 1d

    # Calculate RSI on 1w for higher timeframe momentum filter
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    # Align RSI to 1d
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # Calculate RSI on 1d for entry timing (overbought/oversold)
    delta_d = np.diff(close)
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    avg_gain_d = pd.Series(gain_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_d = pd.Series(loss_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_d = np.where(avg_loss_d != 0, avg_gain_d / avg_loss_d, 0)
    rsi_d = 100 - (100 / (1 + rs_d))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # warmup for KAMA and RSI
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA trending up (close > KAMA) AND 1w RSI not overbought (<60) AND daily RSI oversold (<30)
            if (close[i] > kama_aligned[i] and 
                rsi_1w_aligned[i] < 60 and
                rsi_d[i] < 30):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down (close < KAMA) AND 1w RSI not oversold (>40) AND daily RSI overbought (>70)
            elif (close[i] < kama_aligned[i] and 
                  rsi_1w_aligned[i] > 40 and
                  rsi_d[i] > 70):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA flips down (close < KAMA)
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA flips up (close > KAMA)
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals