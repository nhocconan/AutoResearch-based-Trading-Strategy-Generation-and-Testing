#!/usr/bin/env python3
# 12h_RSI_Extremes_With_1wTrend_Filter
# Hypothesis: RSI extremes on 12h (overbought >70, oversold <30) with weekly trend filter (price > weekly EMA200 for longs, < for shorts).
# Uses weekly EMA200 to filter trades in line with higher timeframe trend, reducing counter-trend trades.
# RSI calculated on 12h closes with 14-period lookback.
# Target: 12-37 trades/year per symbol to minimize fee decay while capturing mean-reversion in trending markets.
# Works in bull/bear: trend filter ensures trades align with major direction, RSI captures pullbacks/retracements.

name = "12h_RSI_Extremes_With_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # RSI on 12h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required value is NaN
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) and price above weekly EMA200 (uptrend)
            if rsi[i] < 30 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) and price below weekly EMA200 (downtrend)
            elif rsi[i] > 70 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (neutral) or price crosses below weekly EMA200
            if rsi[i] > 50 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (neutral) or price crosses above weekly EMA200
            if rsi[i] < 50 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals