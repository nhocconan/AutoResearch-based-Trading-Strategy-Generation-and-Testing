#!/usr/bin/env python3
# 6h_RSI_EMA_Trend_Filter
# Hypothesis: Combine RSI(14) momentum with EMA(50) trend filter on 6h timeframe, using 1w trend for regime filtering.
# Long when RSI > 50 and price > EMA50 and 1w trend up; Short when RSI < 50 and price < EMA50 and 1w trend down.
# Uses volume confirmation to avoid false signals. Designed for 10-30 trades/year to minimize fee drag.

name = "6h_RSI_EMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for regime filter (trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Calculate EMA(50) on 6h for trend filter
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_6h[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50 (bullish momentum) + price > EMA50 + 1w trend up + volume confirmation
            if (rsi[i] > 50 and 
                close[i] > ema50_6h[i] and
                ema50_1w_aligned[i] > ema50_1w_aligned[max(i-1, 0)] and  # 1w trend up
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 (bearish momentum) + price < EMA50 + 1w trend down + volume confirmation
            elif (rsi[i] < 50 and 
                  close[i] < ema50_6h[i] and
                  ema50_1w_aligned[i] < ema50_1w_aligned[max(i-1, 0)] and  # 1w trend down
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or 1w trend turns down
            if rsi[i] < 50 or ema50_1w_aligned[i] < ema50_1w_aligned[max(i-1, 0)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 or 1w trend turns up
            if rsi[i] > 50 or ema50_1w_aligned[i] > ema50_1w_aligned[max(i-1, 0)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals