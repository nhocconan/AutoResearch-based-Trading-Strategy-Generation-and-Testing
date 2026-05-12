#!/usr/bin/env python3
# 4h_KAMA_Trend_With_200SMA_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction on 4h, confirmed by price position relative to 200-period SMA (trend filter). Enter long when KAMA > SMA200 and price > KAMA; short when KAMA < SMA200 and price < KAMA. Exit on opposite signal. This captures trending markets while avoiding whipsaws in sideways action. Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets via dual trend confirmation.

name = "4h_KAMA_Trend_With_200SMA_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Calculate KAMA (2-period ER, 30-period smoothing constant)
    change = np.abs(close - np.roll(close, 1))
    change[0] = 0
    dir = np.abs(close - np.roll(close, 9))  # 10-period direction
    dir[0] = 0
    vol = np.sum(change[1:10])  # 9-period volatility
    er = np.where(vol != 0, dir / vol, 0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 200-period SMA for trend filter
    sma200 = np.convolve(close, np.ones(200)/200, mode='same')
    # Handle edges: use expanding mean for first 200 periods
    for i in range(200):
        sma200[i] = np.mean(close[:i+1])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):  # Start after KAMA warmup
        if position == 0:
            # LONG: KAMA > SMA200 (bullish trend) and price > KAMA (momentum confirmation)
            if kama[i] > sma200[i] and close[i] > kama[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA < SMA200 (bearish trend) and price < KAMA (momentum confirmation)
            elif kama[i] < sma200[i] and close[i] < kama[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA < SMA200 (trend turns bearish)
            if kama[i] < sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA > SMA200 (trend turns bullish)
            if kama[i] > sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals