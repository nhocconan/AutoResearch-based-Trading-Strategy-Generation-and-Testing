#!/usr/bin/env python3

# 4h_12h_KAMA_Direction_RSI_Filter_With_Trend
# Hypothesis: Use KAMA on 4h for direction, RSI(14) for momentum confirmation, and 12h EMA50 as trend filter.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI filters for momentum strength.
# 12h EMA ensures alignment with higher timeframe trend. Designed for low frequency (<30 trades/year) to minimize fee drag.
# Works in bull markets via trend-following and in bear via momentum exhaustion signals.

name = "4h_12h_KAMA_Direction_RSI_Filter_With_Trend"
timeframe = "4h"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Calculate KAMA on 4h (ER = 10, fast = 2, slow = 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    # Recalculate volatility properly: sum of absolute changes over ER period
    er_period = 10
    change_arr = np.abs(np.diff(close, prepend=close[0]))
    volatility_arr = np.zeros_like(change)
    for i in range(len(change)):
        if i < er_period:
            volatility_arr[i] = np.nan
        else:
            volatility_arr[i] = np.sum(change_arr[i-er_period+1:i+1])
    # Avoid division by zero
    er = np.where(volatility_arr != 0, change_arr / volatility_arr, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI(14) on 4h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # KAMA direction: price above KAMA = bullish, below = bearish
        bullish_kama = close[i] > kama[i]
        bearish_kama = close[i] < kama[i]

        # RSI filters: avoid overbought/oversold extremes
        rsi_ok_long = rsi[i] < 70  # not overbought
        rsi_ok_short = rsi[i] > 30  # not oversold

        if position == 0:
            # LONG: Price above KAMA with rising momentum and bullish 12h trend
            if bullish_kama and rsi_ok_long and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with falling momentum and bearish 12h trend
            elif bearish_kama and rsi_ok_short and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals