#!/usr/bin/env python3
"""
4h_RSI_Divergence_1dTrend_VolumeFilter
Hypothesis: RSI divergence (bullish/bearish) on 4h combined with 1d EMA trend filter and volume confirmation captures reversals in both bull and bear markets. Bullish divergence + 1d uptrend + volume spike = long; bearish divergence + 1d downtrend + volume spike = short. Uses RSI divergence for early reversal signals and 1d trend for filtering.
"""

name = "4h_RSI_Divergence_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate RSI on 4h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 3:
            # Check last 3 bars for bullish divergence
            if low[i] < low[i-1] < low[i-2] and rsi[i] > rsi[i-1] > rsi[i-2]:
                bullish_div = True
            # Also check for divergence over 4 bars
            elif low[i] < low[i-3] and rsi[i] > rsi[i-3]:
                bullish_div = True

        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 3:
            # Check last 3 bars for bearish divergence
            if high[i] > high[i-1] > high[i-2] and rsi[i] < rsi[i-1] < rsi[i-2]:
                bearish_div = True
            # Also check for divergence over 4 bars
            elif high[i] > high[i-3] and rsi[i] < rsi[i-3]:
                bearish_div = True

        if position == 0:
            # LONG: Bullish RSI divergence + 1d uptrend + volume spike
            if bullish_div and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = 0.30
                position = 1
            # SHORT: Bearish RSI divergence + 1d downtrend + volume spike
            elif bearish_div and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish RSI divergence or 1d trend turns down
            if bearish_div or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Bullish RSI divergence or 1d trend turns up
            if bullish_div or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals