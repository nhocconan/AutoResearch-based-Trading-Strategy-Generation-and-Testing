#!/usr/bin/env python3
# 6h_RSI_Divergence_1wTrend_Momentum
# Hypothesis: RSI(14) divergences on 6h with weekly trend filter capture reversals in both bull and bear markets. 
# Uses hidden bullish/bearish divergences: price makes lower low/higher high while RSI makes higher low/lower low.
# Weekly trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws.
# Volume confirmation adds confirmation of momentum strength.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing.

name = "6h_RSI_Divergence_1wTrend_Momentum"
timeframe = "6h"
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

    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Weekly trend filter: EMA50 on 1w close
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(rsi[i]) or np.isnan(rsi[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Detect hidden bullish divergence: price makes lower low, RSI makes higher low
        hidden_bull_div = False
        if i >= 2:
            if low[i] < low[i-1] and rsi[i] > rsi[i-1]:
                hidden_bull_div = True

        # Detect hidden bearish divergence: price makes higher high, RSI makes lower high
        hidden_bear_div = False
        if i >= 2:
            if high[i] > high[i-1] and rsi[i] < rsi[i-1]:
                hidden_bear_div = True

        if position == 0:
            # LONG: Hidden bullish divergence + weekly uptrend + volume spike
            if (hidden_bull_div and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Hidden bearish divergence + weekly downtrend + volume spike
            elif (hidden_bear_div and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Hidden bearish divergence or weekly trend turns down
            if hidden_bear_div or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Hidden bullish divergence or weekly trend turns up
            if hidden_bull_div or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals