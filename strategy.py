#!/usr/bin/env python3
"""
6h_Laguerre_RSI_Trend_Filter_With_Volume_Confirmation
Hypothesis: Laguerre RSI (LRSI) provides smoother, less noisy RSI-like signals with faster
reaction to price changes. Combined with 1d trend filter (price above/below 100 EMA) and
volume confirmation, it aims to capture high-probability momentum entries in both bull
and bear markets. The LRSI reduces whipsaw compared to traditional RSI, improving
win rate while maintaining sufficient trade frequency.
"""

name = "6h_Laguerre_RSI_Trend_Filter_With_Volume_Confirmation"
timeframe = "6h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 100-period EMA on daily data for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)

    # Calculate Laguerre RSI (LRSI) on 6h close prices
    # LRSI formula: LRSI = (L0 + 2*L1 + 2*L2 + L3) / 3
    # where Ln = (1-g)*price + g*L(n-1), g = gamma (damping factor)
    gamma = 0.2  # typical value for Laguerre RSI
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)

    # Initialize first values
    L0[0] = (1 - gamma) * close[0] + gamma * close[0]
    L1[0] = (1 - gamma) * L0[0] + gamma * L0[0]
    L2[0] = (1 - gamma) * L1[0] + gamma * L1[0]
    L3[0] = (1 - gamma) * L2[0] + gamma * L2[0]

    # Calculate Laguerre values
    for i in range(1, n):
        L0[i] = (1 - gamma) * close[i] + gamma * L0[i-1]
        L1[i] = (1 - gamma) * L0[i] + gamma * L1[i-1]
        L2[i] = (1 - gamma) * L1[i] + gamma * L2[i-1]
        L3[i] = (1 - gamma) * L2[i] + gamma * L3[i-1]

    # Calculate LRSI: (L0 + 2*L1 + 2*L2 + L3) / 3
    lrs = (L0 + 2*L1 + 2*L2 + L3) / 3
    # Normalize to 0-100 scale: LRSI = LRS * 100
    lrs_i = lrs * 100

    # Volume spike: volume > 1.5 * 20-period average (~5 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(lrs_i[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (price > 100 EMA) + LRSI < 30 (oversold) + volume spike
            if close[i] > ema100_1d_aligned[i] and lrs_i[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < 100 EMA) + LRSI > 70 (overbought) + volume spike
            elif close[i] < ema100_1d_aligned[i] and lrs_i[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: LRSI > 70 (overbought) or trend turns bearish
            if lrs_i[i] > 70 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: LRSI < 30 (oversold) or trend turns bullish
            if lrs_i[i] < 30 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals