#!/usr/bin/env python3
"""
1d_KAMA_1wTrend_RSI_Momentum
Hypothesis: On daily timeframe, KAMA (Kaufman Adaptive Moving Average) tracks trend with low whipsaw.
Long when price > KAMA, RSI > 50, and 1w trend up (price > 1w EMA34).
Short when price < KAMA, RSI < 50, and 1w trend down (price < 1w EMA34).
Uses volume confirmation (volume > 1.5x 20-day average) to filter false breakouts.
Designed for low turnover: ~15-25 trades/year to minimize fee drag.
Works in bull via momentum continuation and bear via mean-reversion at extremes with trend filter.
"""

name = "1d_KAMA_1wTrend_RSI_Momentum"
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

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) = |net change| / sum(|changes|) over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = kama  # already aligned to daily

    # Calculate 1w EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # fill NaN with neutral

    # Volume confirmation: 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # start after warmup for KAMA/RSI
        # Get aligned values for current day
        ema34_1w = ema34_1w_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema34_1w) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA, RSI > 50, 1w trend up, volume surge
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema34_1w and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA, RSI < 50, 1w trend down, volume surge
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema34_1w and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA or RSI < 40
            if (close[i] < kama[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA or RSI > 60
            if (close[i] > kama[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals