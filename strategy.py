#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Confirmation_1wTrend
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing reliable trend direction. 
In trending markets (KAMA slope aligned with 1w trend), enter long when RSI crosses above 50 with volume confirmation, 
and short when RSI crosses below 50. Works in both bull (follows uptrend) and bear (follows downtrend) markets.
"""

name = "1d_KAMA_Direction_RSI_Confirmation_1wTrend"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # KAMA calculation (10-period ER, 2/30 SC)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (uptrend) + RSI crosses above 50 + volume spike + 1w uptrend
            if kama[i] > kama[i-1] and rsi[i] > 50 and rsi[i-1] <= 50 and volume[i] > vol_avg_20[i] * 1.5 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (downtrend) + RSI crosses below 50 + volume spike + 1w downtrend
            elif kama[i] < kama[i-1] and rsi[i] < 50 and rsi[i-1] >= 50 and volume[i] > vol_avg_20[i] * 1.5 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI drops below 50
            if kama[i] < kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI rises above 50
            if kama[i] > kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals