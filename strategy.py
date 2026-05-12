#!/usr/bin/env python3
# 6h_RSI_Trend_Confluence
# Hypothesis: RSI momentum combined with 1w EMA trend and 1d volume confirmation
# works in both bull and bear markets. RSI filters avoid extremes, while EMA
# ensures trend alignment. Volume confirms conviction. Designed for ~15-25
# trades/year to minimize fee drag.

name = "6h_RSI_Trend_Confluence"
timeframe = "6h"
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

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # 1d volume SMA20 for confirmation
    volume_series = pd.Series(volume_1d)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_sma20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after indicators need 50 bars
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(volume_sma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50 (bullish momentum) + price > 1w EMA50 + volume > 1.5x avg
            if (rsi[i] > 50 and
                close[i] > ema50_1w_aligned[i] and
                volume[i] > volume_sma20_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 (bearish momentum) + price < 1w EMA50 + volume > 1.5x avg
            elif (rsi[i] < 50 and
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > volume_sma20_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 (loss of momentum) OR price < 1w EMA50
            if rsi[i] < 40 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 60 (loss of momentum) OR price > 1w EMA50
            if rsi[i] > 60 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals