#!/usr/bin/env python3
# 4h_RSI_Engulfing_Pullback_Trend_1d
# Hypothesis: Combine RSI pullbacks with bullish/bearish engulfing candles on 4h, filtered by 1d EMA trend.
# Long: RSI < 40, bullish engulfing, price above 1d EMA50. Short: RSI > 60, bearish engulfing, price below 1d EMA50.
# Exit: RSI crosses back above 60 (long) or below 40 (short).
# Designed to work in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
# Target: 20-40 trades/year per symbol.

name = "4h_RSI_Engulfing_Pullback_Trend_1d"
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
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # Detect bullish and bearish engulfing candles
    bullish_engulfing = (close > open_) & (open_ > np.roll(close, 1)) & (close > np.roll(open_, 1))
    bearish_engulfing = (close < open_) & (open_ < np.roll(close, 1)) & (close < np.roll(open_, 1))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data is not ready
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 40, bullish engulfing, price above 1d EMA (uptrend)
            if rsi[i] < 40 and bullish_engulfing[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 60, bearish engulfing, price below 1d EMA (downtrend)
            elif rsi[i] > 60 and bearish_engulfing[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses back above 60
            if rsi[i] >= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses back below 40
            if rsi[i] <= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals