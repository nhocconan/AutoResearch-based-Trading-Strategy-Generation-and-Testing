#!/usr/bin/env python3
# 4h_KAMA_Direction_1dTrend_ChopFilter
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in chop. Combined with 1d EMA trend filter and
# Choppiness Index regime filter, it captures trends while avoiding false signals in sideways markets.
# Works in bull (trend-following) and bear (trend-following short) by using 1d EMA for direction.
# Target: 20-30 trades/year per symbol.

name = "4h_KAMA_Direction_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        abs_change = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if np.sum(abs_change[i-period:i]) != 0:
                er[i] = change[i-1] / np.sum(abs_change[i-period:i])
            else:
                er[i] = 0
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama = calculate_kama(close, 10, 2, 30)

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Choppiness Index on 1d (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr
        # True Range sum over period
        tr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Max/min close over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # Chop
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop

    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data is not ready
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, 1d EMA uptrend, and chop < 61.8 (trending)
            if close[i] > kama[i] and close[i] > ema_1d_aligned[i] and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, 1d EMA downtrend, and chop < 61.8 (trending)
            elif close[i] < kama[i] and close[i] < ema_1d_aligned[i] and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR chop > 61.8 (choppy)
            if close[i] < kama[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR chop > 61.8 (choppy)
            if close[i] > kama[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals