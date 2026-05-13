#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Trend
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter.
# Combine with RSI for momentum and Choppiness Index to filter ranging markets.
# Long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2).
# Short when KAMA turns down, RSI < 50, and market is trending (CHOP < 38.2).
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (~10-25 trades/year) to minimize fee drag on 1d timeframe.

name = "1d_KAMA_RSI_Chop_Trend"
timeframe = "1d"
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

    # Calculate KAMA (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0, keepdims=True)
    volatility = np.roll(np.cumsum(np.abs(np.diff(close), prepend=0)), 1)
    volatility = np.where(np.arange(len(close)) < 1, 0, volatility)
    volatility_diff = np.diff(volatility, prepend=0)
    er = np.where(volatility_diff != 0, change / volatility_diff, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) != 0, chop, 50)  # avoid division by zero

    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up, RSI > 50, trending market (CHOP < 38.2), above 1w EMA50
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 38.2 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down, RSI < 50, trending market (CHOP < 38.2), below 1w EMA50
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 38.2 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down or chop becomes too high (ranging) or below 1w EMA50
            if kama[i] < kama[i-1] or chop[i] > 61.8 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up or chop becomes too high (ranging) or above 1w EMA50
            if kama[i] > kama[i-1] or chop[i] > 61.8 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals