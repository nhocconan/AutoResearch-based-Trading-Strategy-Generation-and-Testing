#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_1dTrend_With_Volume
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 1d, combined with RSI for momentum and volume confirmation. Enter long when KAMA is rising, RSI < 30 (oversold), and volume > 1.5x average. Enter short when KAMA is falling, RSI > 70 (overbought), and volume > 1.5x average. Exit when RSI returns to neutral range (40-60). This strategy aims to catch mean-reversion moves within the dominant trend, reducing false signals in strong trends. Designed for low frequency to minimize fee drag, suitable for 1d timeframe with expected 10-30 trades per year.

name = "1d_KAMA_Direction_RSI_1dTrend_With_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Weekly trend: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Daily KAMA (10, 2, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first 10 values
    er = np.full_like(change, np.nan, dtype=np.float64)
    er[10:] = change[10:] / volatility[10:]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # Start at index 9 for 10-period lookback
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Daily RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first value
    rsi = np.insert(rsi, 0, 50.0)

    # Volume average (20-day)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup for indicators
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (today > yesterday), RSI < 30, volume spike, price above weekly EMA50
            if kama[i] > kama[i-1] and rsi[i] < 30 and volume[i] > 1.5 * vol_ma_20[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (today < yesterday), RSI > 70, volume spike, price below weekly EMA50
            elif kama[i] < kama[i-1] and rsi[i] > 70 and volume[i] > 1.5 * vol_ma_20[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (40-60) or KAMA turns down
            if rsi[i] >= 40 and rsi[i] <= 60 or kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (40-60) or KAMA turns up
            if rsi[i] >= 40 and rsi[i] <= 60 or kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals