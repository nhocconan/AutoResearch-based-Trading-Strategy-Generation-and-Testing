#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Range_Exit
# Hypothesis: 4h KAMA identifies trend direction (bull/bear), RSI(2) triggers mean-reversion entries in range-bound markets,
# and exits on RSI mean reversion. Uses 1d ADX > 25 to filter for trending markets only, avoiding whipsaws in low ADX.
# Designed for 20-50 trades/year on 4h to minimize fee drag. Works in bull by following KAMA trend long,
# and in bear by fading RSI extremes against the trend when ADX confirms trend strength.

name = "4h_KAMA_Trend_RSI_Range_Exit"
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

    # Get 4h data for KAMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.abs(np.diff(close_4h))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    kama_4h = kama

    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Plus Directional Movement and Minus Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period]) if np.any(~np.isnan(arr[:period])) else 0
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed

    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth_wilder(dx, 14)

    adx_1d = adx

    # Align KAMA and ADX to 4h timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Calculate RSI(2) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Only trade when ADX > 25 (trending market)
            if adx_1d_aligned[i] > 25:
                # LONG: Price above KAMA (uptrend) and RSI < 15 (oversold)
                if close[i] > kama_4h_aligned[i] and rsi[i] < 15:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price below KAMA (downtrend) and RSI > 85 (overbought)
                elif close[i] < kama_4h_aligned[i] and rsi[i] > 85:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Low ADX: range market, no trend following
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (mean reversion)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (mean reversion)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals