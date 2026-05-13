#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Pullback
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on daily timeframe.
# Enter long when price pulls back to KAMA during uptrend with RSI < 40 and volume above average.
# Enter short when price pulls back to KAMA during downtrend with RSI > 60 and volume above average.
# Exit when price crosses KAMA in opposite direction.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# Combined with RSI for mean reversion entries and volume confirmation to avoid low-quality signals.
# Target: 15-25 trades/year on 1d to minimize fee drag while capturing meaningful swings.

name = "1d_KAMA_Trend_RSI_Pullback"
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

    # Calculate weekly EMA40 for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)

    # Calculate daily KAMA (ER=10, Fast=2, Slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix volatility calculation: sum of absolute changes over ER period
    volatility_sum = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - 9)
        volatility_sum[i] = np.sum(np.abs(np.diff(close[start:i+1], prepend=close[start])))
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already on daily timeframe

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
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near KAMA in uptrend, RSI oversold, volume confirmation
            if (close[i] <= kama[i] * 1.005 and  # within 0.5% above KAMA
                ema40_1w_aligned[i] > ema40_1w_aligned[i-1] and  # weekly uptrend
                rsi[i] < 40 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near KAMA in downtrend, RSI overbought, volume confirmation
            elif (close[i] >= kama[i] * 0.995 and  # within 0.5% below KAMA
                  ema40_1w_aligned[i] < ema40_1w_aligned[i-1] and  # weekly downtrend
                  rsi[i] > 60 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals