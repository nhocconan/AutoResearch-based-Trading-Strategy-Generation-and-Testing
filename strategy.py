#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_RSI_Reversal
# Hypothesis: Use 1d KAMA for trend direction, 4h RSI for mean-reversion entries, and 1d volatility filter to avoid chop.
# In bull markets: buy dips in uptrend (RSI < 30). In bear markets: sell rallies in downtrend (RSI > 70).
# Volatility filter ensures trades only during sufficient momentum, reducing whipsaws in low-volatility periods.
# Designed for 20-50 trades/year on 4h timeframe with discrete position sizing to minimize fee drag.

name = "4h_1d_KAMA_Trend_RSI_Reversal"
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

    # Get daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily KAMA for trend filter
    close_1d = df_1d['close'].values
    # Efficiency ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Calculate daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full_like(close_1d, np.nan)
    atr_1d[13] = np.mean(tr[:13])  # First ATR at index 13
    for i in range(14, len(tr)+1):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i-1]) / 14
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)

    # Calculate 4h RSI (14-period) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average at index 13
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: ATR > 0.5 * average ATR over last 50 periods
        if i >= 50:
            atr_ma = np.nanmean(atr_1d_aligned[i-50:i])
            vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma
        else:
            vol_filter = True  # Not enough data for MA, allow trade

        # Trend filter
        bullish_trend = close[i] > kama_1d_aligned[i]
        bearish_trend = close[i] < kama_1d_aligned[i]

        if position == 0:
            # LONG: Price in uptrend and RSI oversold
            if bullish_trend and rsi[i] < 30 and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Price in downtrend and RSI overbought
            elif bearish_trend and rsi[i] > 70 and vol_filter:
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