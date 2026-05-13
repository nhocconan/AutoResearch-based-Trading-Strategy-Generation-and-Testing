#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_1dTrend_Volume
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction, 
# combined with RSI(14) for momentum and volume confirmation. Enter long when KAMA 
# is rising and RSI > 55, short when KAMA is falling and RSI < 45, filtered by 1d 
# EMA50 trend and volume > 1.3x 20-period average. Designed for low-frequency 
# trend following with adaptive smoothing to reduce whipsaws in choppy markets.

name = "12h_KAMA_Trend_RSI_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(close - np.roll(close, length))
    change[0] = 0
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    volatility = np.full_like(close, volatility, dtype=float)
    for i in range(length, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-length+1:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama_out = np.zeros_like(close)
    kama_out[0] = close[0]
    for i in range(1, len(close)):
        kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
    return kama_out

def rsi(close, length=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).ewm(span=length, adjust=False).mean()
    roll_down = pd.Series(down).ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi_out = 100 - (100 / (1 + rs))
    return rsi_out.values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate KAMA(10,2,30) on 12h data
    kama_val = kama(close, 10, 2, 30)

    # Calculate RSI(14)
    rsi_val = rsi(close, 14)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        kama_rising = kama_val[i] > kama_val[i-1]
        kama_falling = kama_val[i] < kama_val[i-1]

        if position == 0:
            # LONG: KAMA rising + RSI > 55 + price above 1d EMA50 + volume spike
            if (kama_rising and 
                rsi_val[i] > 55 and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + RSI < 45 + price below 1d EMA50 + volume spike
            elif (kama_falling and 
                  rsi_val[i] < 45 and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI < 50 or price below 1d EMA50
            if (not kama_rising or rsi_val[i] < 50 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI > 50 or price above 1d EMA50
            if (kama_falling or rsi_val[i] > 50 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals