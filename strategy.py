#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_14_Range_Filter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for mean-reversion entries, and Choppiness Index to filter ranging vs trending regimes.
Long when KAMA rising, RSI < 30, and market is ranging (CHOP > 61.8).
Short when KAMA falling, RSI > 70, and market is ranging.
Avoids trend-following whipsaws in ranging markets and captures mean reversion with trend filter.
Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull/bear markets.
"""

name = "1d_KAMA_Direction_RSI_14_Range_Filter"
timeframe = "1d"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate KAMA (ER=10, fast=2, slow=30) on daily close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)

    # Weekly EMA(20) for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w.values)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema20_val = ema20_1w_aligned[i]

        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(ema20_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (bullish), RSI oversold, ranging market
            if kama[i] > kama[i-1] and rsi_val < 30 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish), RSI overbought, ranging market
            elif kama[i] < kama[i-1] and rsi_val > 70 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down or RSI overbought
            if kama[i] < kama[i-1] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up or RSI oversold
            if kama[i] > kama[i-1] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals