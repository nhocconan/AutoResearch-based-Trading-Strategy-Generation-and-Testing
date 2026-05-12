#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) trend direction combined with RSI momentum and Choppiness Index regime filter.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters for momentum strength.
# Choppiness Index (CHOP) > 61.8 indicates ranging markets (mean reversion), < 38.2 indicates trending markets (trend follow).
# Strategy: In trending markets (CHOP < 38.2), follow KAMA direction. In ranging markets (CHOP > 61.8), fade extreme RSI.
# Weekly trend filter ensures alignment with long-term direction. Designed for 15-25 trades/year per symbol.

name = "1d_KAMA_RSI_Chop_Filter"
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
    if len(df_1w) < 30:
        return np.zeros(n)

    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # KAMA (ER=10, fast=2, slow=30) - trend indicator
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14) - momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index (CHOP) - regime filter
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(np.sum(tr[-14:]) / (highest_high - lowest_low)) / np.log10(14),
                    50)
    # Handle edge cases
    chop = np.where(np.isnan(chop), 50, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filters
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8

        if position == 0:
            # Trending market: follow KAMA direction
            if is_trending:
                if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and close[i-1] >= kama[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging market: fade RSI extremes
            elif is_ranging:
                if rsi[i] < 30 and rsi[i-1] >= 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and rsi[i-1] <= 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: wait for clear signal
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA cross down OR RSI overbought in range OR trend change
            if (close[i] < kama[i]) or (is_ranging and rsi[i] > 70) or (not is_trending and close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA cross up OR RSI oversold in range OR trend change
            if (close[i] > kama[i]) or (is_ranging and rsi[i] < 30) or (not is_trending and close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals