#!/usr/bin/env python3
"""
4h_RSI200_Trend_Breakout_With_Volume
Hypothesis: On 4h timeframe, RSI crossing above/below 50 with price above/below 200 EMA and volume surge indicates momentum shift. 
RSI > 50 + price > EMA200 + volume > 1.5x average signals long; RSI < 50 + price < EMA200 + volume > 1.5x average signals short.
Uses 1d EMA200 as higher timeframe trend filter to avoid counter-trend trades. 
Volume confirmation ensures institutional participation. 
Target: 20-50 trades/year (80-200 total over 4 years) with clear entry/exit rules.
Works in bull via momentum continuation and bear via mean-reversion at extremes with trend filter.
"""

name = "4h_RSI200_Trend_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 220:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for EMA200 calculation (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Calculate 4h EMA200 for trend
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)

    # Get 1d data for higher timeframe trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA200 for higher timeframe trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Get aligned values for current 4h bar
        ema200_4h_val = ema200_4h_aligned[i]
        ema200_1d_val = ema200_1d_aligned[i]
        rsi_val = rsi[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_val) or np.isnan(ema200_1d_val) or 
            np.isnan(rsi_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50 + price > EMA200(4h) + price > EMA200(1d) + volume surge
            if (rsi_val > 50 and 
                close[i] > ema200_4h_val and 
                close[i] > ema200_1d_val and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 + price < EMA200(4h) + price < EMA200(1d) + volume surge
            elif (rsi_val < 50 and 
                  close[i] < ema200_4h_val and 
                  close[i] < ema200_1d_val and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or price < EMA200(4h) or price < EMA200(1d)
            if (rsi_val < 50 or close[i] < ema200_4h_val or close[i] < ema200_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 or price > EMA200(4h) or price > EMA200(1d)
            if (rsi_val > 50 or close[i] > ema200_4h_val or close[i] > ema200_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals