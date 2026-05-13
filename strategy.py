#!/usr/bin/env python3
# 4h_RSI2_Close_Breakout_1dTrend_Volume
# Hypothesis: Use 2-period RSI on close for short-term momentum, with 1d EMA50 trend filter and volume spike.
# Enter long when RSI2 < 10 (oversold) and price closes above 4h EMA20 with 1d EMA50 uptrend and volume > 2x average.
# Enter short when RSI2 > 90 (overbought) and price closes below 4h EMA20 with 1d EMA50 downtrend and volume > 2x average.
# Exit when RSI2 crosses back above 50 (long) or below 50 (short).
# Target: 20-30 trades/year on 4h to minimize fee decay.

name = "4h_RSI2_Close_Breakout_1dTrend_Volume"
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
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 4h EMA20 for entry filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate 2-period RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI2 < 10 (oversold) + close > EMA20 + 1d EMA50 uptrend + volume spike
            if (rsi[i] < 10 and 
                close[i] > ema20[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # 1d EMA50 rising
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI2 > 90 (overbought) + close < EMA20 + 1d EMA50 downtrend + volume spike
            elif (rsi[i] > 90 and 
                  close[i] < ema20[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # 1d EMA50 falling
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI2 crosses back above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI2 crosses back below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals