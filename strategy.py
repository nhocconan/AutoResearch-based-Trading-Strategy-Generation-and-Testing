#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Filter_Trend_200
# Hypothesis: Use daily KAMA direction (trend filter) with RSI(14) for entry timing and 200-period EMA for long-term trend filter.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI identifies overbought/oversold conditions within the trend.
# The 200 EMA ensures we only trade in the direction of the long-term trend, improving performance in both bull and bear markets.
# Entry: Long when KAMA trending up, RSI < 30 (oversold), and close > EMA200.
# Entry: Short when KAMA trending down, RSI > 70 (overbought), and close < EMA200.
# Exit: Reverse signal or trend change.
# Target: 20-60 total trades over 4 years = 5-15/year.

name = "1d_KAMA_Direction_RSI_Filter_Trend_200"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute daily changes
    
    # Handle edge cases for ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA200 for long-term trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get weekly trend filter (price vs weekly EMA50)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_200[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA trending up (current > previous), RSI < 30 (oversold), close > EMA200
            if (kama[i] > kama[i-1] and 
                rsi[i] < 30 and
                close[i] > ema_200[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down (current < previous), RSI > 70 (overbought), close < EMA200
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 70 and
                  close[i] < ema_200[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI > 70 (overbought) OR close < EMA200
            if (kama[i] < kama[i-1] or rsi[i] > 70 or close[i] < ema_200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI < 30 (oversold) OR close > EMA200
            if (kama[i] > kama[i-1] or rsi[i] < 30 or close[i] > ema_200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals