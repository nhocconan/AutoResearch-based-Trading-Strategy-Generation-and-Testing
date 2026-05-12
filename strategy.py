#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Reversal_4h1d
# Hypothesis: Use KAMA on 4h for trend direction and RSI on 1d for mean-reversion entries.
# In bull markets: go long when KAMA trends up and RSI pulls back from overbought.
# In bear markets: go short when KAMA trends down and RSI bounces from oversold.
# Uses 4h for entry timing and 1d RSI to avoid whipsaws. Designed for 15-35 trades/year.

name = "4h_KAMA_Trend_RSI_Reversal_4h1d"
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

    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 4h KAMA (trend filter)
    close_4h = close
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.abs(np.diff(close_4h))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    kama = kama  # already aligned to 4h

    # Calculate daily RSI (mean reversion signal)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi  # daily RSI values

    # Align daily RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from KAMA
        bullish_trend = close[i] > kama[i]
        bearish_trend = close[i] < kama[i]

        if position == 0:
            # LONG: KAMA uptrend + RSI pulls back from overbought (>70 to <70)
            if bullish_trend and rsi_aligned[i] < 70 and rsi_aligned[i-1] >= 70 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downtrend + RSI bounces from oversold (<30 to >30)
            elif bearish_trend and rsi_aligned[i] > 30 and rsi_aligned[i-1] <= 30 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI reaches overbought or trend turns bearish
            if rsi_aligned[i] >= 70 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI reaches oversold or trend turns bullish
            if rsi_aligned[i] <= 30 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals