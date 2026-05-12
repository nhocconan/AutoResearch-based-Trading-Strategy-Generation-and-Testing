#!/usr/bin/env python3
# 1d_WeeklyCryptoTrend
# Hypothesis: Weekly trend following with daily retracement entries. Uses 1w EMA for trend direction and 1d RSI for entry timing.
# In bull markets: buy dips in uptrend. In bear markets: sell rallies in downtrend.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Uses weekly trend filter to avoid counter-trend trades and daily RSI for precise entries.

name = "1d_WeeklyCryptoTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Get daily RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + daily RSI oversold (retracement buy)
            if close[i] > ema21_1w_aligned[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + daily RSI overbought (retracement sell)
            elif close[i] < ema21_1w_aligned[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR RSI overbought
            if close[i] < ema21_1w_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR RSI oversold
            if close[i] > ema21_1w_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals