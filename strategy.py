#!/usr/bin/env python3
# 1d_RSI_Extreme_1wTrend_VolumeFilter
# Hypothesis: On daily timeframe, take long positions when RSI(14) < 30 (oversold) with weekly uptrend (price > EMA50) and volume > 1.5x average; take short positions when RSI(14) > 70 (overbought) with weekly downtrend (price < EMA50) and volume > 1.5x average. Exit when RSI returns to neutral zone (40-60). Uses weekly trend filter to avoid counter-trend trades in strong trends, and volume filter to ensure conviction. Designed for low trade frequency (target 10-25 trades/year) to minimize fee drag and improve generalization across bull/bear markets.

name = "1d_RSI_Extreme_1wTrend_VolumeFilter"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) + weekly uptrend (price > EMA50) + volume > 1.5x avg
            if (rsi[i] < 30 and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + weekly downtrend (price < EMA50) + volume > 1.5x avg
            elif (rsi[i] > 70 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>= 40) or weekly trend turns down
            if rsi[i] >= 40 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<= 60) or weekly trend turns up
            if rsi[i] <= 60 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals