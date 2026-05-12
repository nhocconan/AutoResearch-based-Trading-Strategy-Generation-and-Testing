#!/usr/bin/env python3
# 1d_1W_KAMA_Trend_RSI_Reversal
# Hypothesis: Daily trend direction from weekly KAMA combined with RSI mean reversion on pullbacks.
# In bull markets: buy dips when weekly trend is up and RSI < 30.
# In bear markets: sell rallies when weekly trend is down and RSI > 70.
# Weekly trend filter avoids whipsaws; RSI extremes provide entry timing with mean-reversion edge.
# Designed for 10-25 trades/year with low turnover to minimize fee drag.

name = "1d_1W_KAMA_Trend_RSI_Reversal"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    # ER calculation requires cumulative sum over window
    er = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        start = max(0, i - 9)  # 10-period ER
        if i >= 9:
            num = np.abs(close_1w[i] - close_1w[start])
            den = np.sum(np.abs(np.diff(close_1w[start:i+1])))
            er[i] = num / den if den != 0 else 0
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)

    # Daily RSI for mean reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if required data is NaN
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_trend = close[i] > kama_1w_aligned[i]
        bearish_trend = close[i] < kama_1w_aligned[i]

        if position == 0:
            # LONG: Buy dip when weekly trend up and RSI oversold
            if bullish_trend and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Sell rally when weekly trend down and RSI overbought
            elif bearish_trend and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or trend turns bearish
            if rsi[i] > 70 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend turns bullish
            if rsi[i] < 30 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals