#!/usr/bin/env python3
# 1h_4h1d_Momentum_Volume_Regime
# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation, using 4h RSI for entry timing.
# Uses 4h RSI(14) > 55 for long momentum, < 45 for short momentum, filtered by 1d EMA50 trend and volume spike.
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in bull markets via momentum continuation and in bear markets via trend-aligned mean reversion within regime.

name = "1h_4h1d_Momentum_Volume_Regime"
timeframe = "1h"
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

    # Get 4h data for RSI momentum signal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Calculate 4h RSI(14)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h[0] = 50  # neutral for first value

    # Align 4h RSI to 1h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume spike detection (2x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_sma20 * 2.0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h RSI > 55 (bullish momentum) + price > 1d EMA50 (uptrend) + volume spike
            if rsi_4h_aligned[i] > 55 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h RSI < 45 (bearish momentum) + price < 1d EMA50 (downtrend) + volume spike
            elif rsi_4h_aligned[i] < 45 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h RSI < 50 (loss of momentum) or price < 1d EMA50 (trend break)
            if rsi_4h_aligned[i] < 50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h RSI > 50 (loss of bearish momentum) or price > 1d EMA50 (trend break)
            if rsi_4h_aligned[i] > 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals