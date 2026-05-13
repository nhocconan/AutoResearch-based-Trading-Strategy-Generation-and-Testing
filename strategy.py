#!/usr/bin/env python3
# 6h_KAMA_Trend_With_Volume_Regime
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in ranging markets and capturing trends effectively.
# Combined with a volume regime filter (high volume = trending, low volume = ranging) to avoid false signals.
# Long when price crosses above KAMA with volume regime = trending; short when price crosses below KAMA with volume regime = trending.
# Works in bull markets by catching uptrends and in bear markets by catching downtrends, while avoiding range-bound whipsaw.
# Target: 15-35 trades/year per symbol to minimize fee drift.

name = "6h_KAMA_Trend_With_Volume_Regime"
timeframe = "6h"
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

    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30

    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    gap = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(gap != 0, change / gap, 0)
    er = np.concatenate([np.full(er_length, np.nan), er])

    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    sc = np.where(np.isnan(sc), 0, sc)

    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume regime: high volume = trending (volume > 1.5 * 20-period average)
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = np.nan
    vol_ma_20[-10:] = np.nan
    volume_trending = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if np.isnan(kama[i]) or np.isnan(volume_trending[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            if close[i] > kama[i] and volume_trending[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama[i] and volume_trending[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals