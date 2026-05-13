#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_1dTrend
# Hypothesis: KAMA adapts to market efficiency, reducing noise in chop and catching trends. Use KAMA direction as primary trend filter (1d timeframe) and RSI for momentum confirmation. Enter long when price > KAMA and RSI > 55; short when price < KAMA and RSI < 45. Volume confirmation requires >1.3x 20-period average. Designed for low-frequency, high-conviction trades on 12H timeframe to avoid overtrading and fee drag.

name = "12h_KAMA_Trend_RSI_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def rsi(close, length=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    ru = pd.Series(up).ewm(span=length, adjust=False).mean()
    rd = pd.Series(down).ewm(span=length, adjust=False).mean()
    rs = ru / (rd + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for KAMA and RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend filter
    kama_1d = kama(df_1d['close'], er_length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Calculate 1d RSI for momentum filter
    rsi_1d = rsi(df_1d['close'], 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above KAMA (uptrend) + RSI > 55 (bullish momentum) + volume spike
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 55 and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA (downtrend) + RSI < 45 (bearish momentum) + volume spike
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 45 and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below KAMA (trend change) or RSI < 45 (loss of momentum)
            if (close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 45):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above KAMA (trend change) or RSI > 55 (loss of momentum)
            if (close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals