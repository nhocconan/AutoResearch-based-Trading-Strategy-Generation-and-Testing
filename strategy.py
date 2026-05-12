#!/usr/bin/env python3
"""
1D_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA adapts to trend strength, filtering whipsaws in choppy markets. 
Long when KAMA rising and RSI > 50, short when KAMA falling and RSI < 50, only in trending regimes (ADX > 25).
Adds 1-week trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) to minimize fee drag.
"""

name = "1D_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # === 1-week trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # === Daily indicators ===
    # KAMA (adaptive moving average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, tr[0] if len(tr) > 0 else 0)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Align 1w trend to daily
    trend_up = close > ema50_1w_aligned
    trend_down = close < ema50_1w_aligned

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, RSI > 50, ADX > 25, and weekly uptrend
            if kama[i] > kama[i-1] and rsi[i] > 50 and adx[i] > 25 and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, ADX > 25, and weekly downtrend
            elif kama[i] < kama[i-1] and rsi[i] < 50 and adx[i] > 25 and trend_down[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI < 50 or weekly trend turns down
            if kama[i] < kama[i-1] or rsi[i] < 50 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI > 50 or weekly trend turns up
            if kama[i] > kama[i-1] or rsi[i] > 50 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals