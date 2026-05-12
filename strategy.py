#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Filter_And_Pullback
Hypothesis: KAMA adapts to market noise—efficient in trends, slow in chop. Use KAMA direction as trend filter, enter on pullbacks to KAMA with volume confirmation. Works in bull/bear by following trend. Avoids whipsaws in sideways markets by requiring pullback to adaptive average.
"""

name = "1d_KAMA_Trend_With_Filter_And_Pullback"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Calculate KAMA (adaptive moving average) on close
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # For first 10 periods, volatility is cumulative sum of abs changes
    volatility[:10] = np.cumsum(np.abs(np.diff(close, n=1))[:10])
    # Avoid division by zero
    volatility[volatility == 0] = 1e-10
    er = np.zeros_like(close)
    er[10:] = change[10:] / volatility[10:]
    # Smoothing constants: fastest = 2/(2+1)=0.667, slowest=2/(30+1)=0.0645
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Align weekly close to daily for trend filter
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        if (np.isnan(kama[i]) or np.isnan(weekly_close_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above weekly close (uptrend) AND pulls back to KAMA with volume
            if (close[i] > weekly_close_aligned[i] and 
                close[i] <= kama[i] * 1.01 and  # Within 1% above KAMA (pullback)
                low[i] >= kama[i] * 0.99 and   # Not below KAMA
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly close (downtrend) AND pulls back to KAMA with volume
            elif (close[i] < weekly_close_aligned[i] and 
                  close[i] >= kama[i] * 0.99 and  # Within 1% below KAMA (pullback)
                  high[i] <= kama[i] * 1.01 and   # Not above KAMA
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below KAMA (trend weakness)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above KAMA (trend weakness)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals