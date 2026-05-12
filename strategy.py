#!/usr/bin/env python3
"""
4H_KAMA_TRIX_With_Volume_and_Trend_Filter
Hypothesis: KAMA identifies trend direction, TRIX identifies momentum, and volume confirms strength. Combined, they capture trends in both bull and bear markets with low trade frequency to avoid fee drag.
Designed for 20-40 trades/year on 4h timeframe.
"""

name = "4H_KAMA_TRIX_With_Volume_and_Trend_Filter"
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
    volume = volumes = prices['volume'].values

    # Get 4h data for KAMA (trend) and 1d for TRIX (momentum) - but we'll use 4h for both to stay on timeframe
    # Actually, let's use 4h for KAMA trend and 1d for TRIX momentum as per instruction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate KAMA on 4h for trend direction
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close_10|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes over 10 periods
    # Fix: volatility needs to be rolling sum of 1-period changes over 10 periods
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=10).sum().values
    # Prepend 9 NaNs to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Calculate TRIX on 1d for momentum
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = pd.Series(ema3).pct_change() * 100  # percentage

    # Align to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # same timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix.values)

    # Volume confirmation: 4h volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        kama_val = kama_aligned[i]
        trix_val = trix_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(kama_val) or np.isnan(trix_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA (uptrend) AND TRIX > 0 (bullish momentum) AND volume confirmation
            if close[i] > kama_val and trix_val > 0 and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (downtrend) AND TRIX < 0 (bearish momentum) AND volume confirmation
            elif close[i] < kama_val and trix_val < 0 and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA (trend change) OR TRIX < 0 (momentum loss)
            if close[i] < kama_val or trix_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA (trend change) OR TRIX > 0 (momentum loss)
            if close[i] > kama_val or trix_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals