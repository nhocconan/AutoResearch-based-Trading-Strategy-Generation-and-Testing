#!/usr/bin/env python3
"""
4h_RSI200_Trend_Breakout_With_Volume
Hypothesis: RSI(2) extreme reversions combined with EMA200 trend filter and volume confirmation work well on 4h timeframe.
Long when RSI(2)<10, price>EMA200, volume>1.5x average; short when RSI(2)>90, price<EMA200, volume>1.5x average.
Uses 12h EMA50 trend filter to avoid counter-trend trades in strong trends.
Target: 20-50 trades/year (80-200 total over 4 years) with low turnover to minimize fee drag.
Works in bull via trend-aligned reversions and bear via counter-trend bounces at extremes.
"""

name = "4h_RSI200_Trend_Breakout_With_Volume"
timeframe = "4h"
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

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 4h bar
        ema50 = ema50_12h_aligned[i]
        rsi_val = rsi[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if np.isnan(ema50) or np.isnan(rsi_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI(2) < 10 (oversold) + price > EMA50(12h) + volume surge
            if (rsi_val < 10 and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI(2) > 90 (overbought) + price < EMA50(12h) + volume surge
            elif (rsi_val > 90 and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) > 50 or price < EMA50(12h)
            if (rsi_val > 50 or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) < 50 or price > EMA50(12h)
            if (rsi_val < 50 or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals