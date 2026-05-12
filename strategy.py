#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion
Hypothesis: On daily timeframe, use KAMA to determine trend direction (bullish when price > KAMA, bearish when price < KAMA).
In bullish regime, enter long when RSI(14) < 30 (oversold pullback) with volume > 1.5x average.
In bearish regime, enter short when RSI(14) > 70 (overbought bounce) with volume > 1.5x average.
Exit when price crosses back across KAMA or RSI returns to neutral (40-60).
This mean-reversion-within-trend approach aims to capture swings in both bull and bear markets while avoiding chop.
Target: 15-25 trades per year to minimize fee drag.
"""

name = "1d_KAMA_Trend_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # KAMA (Kaufman Adaptive Moving Average) - trend identifier
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, prepend=close[0]))  # actually need net change
    # Correct calculation:
    net_change = np.abs(close - np.roll(close, 1))
    net_change[0] = 0
    sum_abs_change = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_change[0] = 0
    
    # Avoid division by zero
    er = np.where(sum_abs_change != 0, net_change / sum_abs_change, 0)
    
    # Smoothing constants
    sc = (er * (0.0645 - 0.06) + 0.06) ** 2  # 2 and 30 period constants
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume average (20-day)
    vol_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_avg[i] = np.mean(volume[:i+1]) if i >= 0 else 0
        else:
            vol_avg[i] = np.mean(volume[i-19:i+1])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup period
        # Skip if any required value is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish trend (price > KAMA) + RSI oversold + volume confirmation
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                volume[i] > vol_avg[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish trend (price < KAMA) + RSI overbought + volume confirmation
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  volume[i] > vol_avg[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA OR RSI returns to neutral
            if close[i] < kama[i] or (rsi[i] >= 40 and rsi[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA OR RSI returns to neutral
            if close[i] > kama[i] or (rsi[i] >= 40 and rsi[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals