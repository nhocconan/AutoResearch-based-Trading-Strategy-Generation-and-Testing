#!/usr/bin/env python3
"""
12h_RSI_OverboughtOversold_1wTrend
Hypothesis: RSI extreme levels (overbought/oversold) combined with 1-week EMA trend filter identifies reversal opportunities in both bull and bear markets. In bull markets, buy when RSI < 30 and 1w trend up; in bear markets, sell when RSI > 70 and 1w trend down. Uses volume confirmation to avoid false signals.
"""

name = "12h_RSI_OverboughtOversold_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate RSI(14) on 12h timeframe
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan

    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_30[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) + 1w uptrend + volume spike
            if rsi[i] < 30 and close[i] > ema34_1w_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + 1w downtrend + volume spike
            elif rsi[i] > 70 and close[i] < ema34_1w_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 or 1w trend turns down
            if rsi[i] > 50 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 or 1w trend turns up
            if rsi[i] < 50 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals