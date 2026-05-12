#!/usr/bin/env python3
"""
1d_RSI_MeanReversion_WeeklyTrend
Hypothesis: Daily RSI mean reversion (buy oversold, sell overbought) filtered by weekly trend works in both bull and bear markets. Weekly EMA200 defines trend: long when RSI<30 and price>weekly EMA200, short when RSI>70 and price<weekly EMA200. Uses weekly trend filter to avoid counter-trend trades in strong trends, reducing false signals.
"""

name = "1d_RSI_MeanReversion_WeeklyTrend"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA200 for trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Daily RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) and price above weekly EMA200 (uptrend)
            if rsi_values[i] < 30 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price below weekly EMA200 (downtrend)
            elif rsi_values[i] > 70 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or price below weekly EMA200 (trend change)
            if rsi_values[i] > 70 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or price above weekly EMA200 (trend change)
            if rsi_values[i] < 30 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals